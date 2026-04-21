import os
import sys
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from loguru import logger

# 确保能找到项目根目录下的模块
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.crowd_drone import Crowd_Drone
from datasets.crowd_rgbtcc import Crowd_RGBTCC
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob
from losses.bpl import BPL_Loss
from models.RaBiT_Model import fusion_model
from utils.evaluation import eval_game, eval_relative

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    if type(transposed_batch[0][0]) == list:
        rgb_list = [item[0] for item in transposed_batch[0]]
        t_list = [item[1] for item in transposed_batch[0]]
        rgb = torch.stack(rgb_list, 0)
        t = torch.stack(t_list, 0)
        images = [rgb, t]
    else:
        images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    st_sizes = torch.FloatTensor(transposed_batch[2])
    return images, points, st_sizes


class CrowdCountingLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        # 1. 初始化模型
        self.model = fusion_model(
            construct_method=args.constr_hg,
            k_interact=args.constr_k,
            thresh_interact=args.constr_threshold
        )

        params_m = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f'Model params (M): {params_m:.2f}')

        self.lambda_aux_bayes = 0.5  
        self.lambda_bpl = 0.1

    def on_fit_start(self):
        # 2. 初始化损失函数 (此时 self.device 已经是分配好的 GPU，例如 cuda:0 或 cuda:1)
        # 注意：这里我们移除了 self.device 参数，因为 Post_Prob 已经使用了 register_buffer
        self.post_prob = Post_Prob(self.args.sigma, self.args.crop_size, self.args.downsample_ratio, 
                                   self.args.background_ratio, self.args.use_background)
        self.criterion = Bay_Loss(self.args.use_background, self.device)
        self.bpl_criterion = BPL_Loss(
            window_size=8, stride=8, delta=0.5, downsample_ratio=self.args.downsample_ratio
        )
        
        # 确保这些组件都在当前卡上
        if hasattr(self.post_prob, 'to'): self.post_prob.to(self.device)
        if hasattr(self.criterion, 'to'): self.criterion.to(self.device)
        if hasattr(self.bpl_criterion, 'to'): self.bpl_criterion.to(self.device)

    def forward(self, inputs):
        return self.model(inputs)

    def configure_optimizers(self):
        params = []
        for key, value in dict(self.model.named_parameters()).items():
            if 'pvt_backbone_rgb' in key or 'pvt_backbone_t' in key:
                params += [{'params': value, 'lr': self.args.lr * 0.1}]
            else:
                params += [{'params': value}]
        
        optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 800, 1200], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs, points, st_sizes = batch
        
        # 复杂结构挂载到设备
        if isinstance(inputs, list):
            inputs = [x.to(self.device) for x in inputs]
        else:
            inputs = inputs.to(self.device)
            
        points = [p.to(self.device) for p in points]
        st_sizes = st_sizes.to(self.device)
        
        self.post_prob.device = self.device
        self.criterion.device = self.device

        D_F, D_R, D_T, reli_maps_R, reli_maps_T = self(inputs)

        prob_list = self.post_prob(points, st_sizes)
        loss_bayes_F = self.criterion(prob_list, D_F)
        loss_bayes_R = self.criterion(prob_list, D_R)
        loss_bayes_T = self.criterion(prob_list, D_T)

        r_R_bpl = F.adaptive_avg_pool2d(reli_maps_R[0], output_size=D_R.shape[2:])
        r_T_bpl = F.adaptive_avg_pool2d(reli_maps_T[0], output_size=D_T.shape[2:])
        loss_bpl = self.bpl_criterion(D_R, D_T, r_R_bpl, r_T_bpl, points)

        total_loss = loss_bayes_F + \
                     self.lambda_aux_bayes * (loss_bayes_R + loss_bayes_T) + \
                     self.lambda_bpl * loss_bpl

        # --- 计算训练集全指标并记录到 TensorBoard ---
        with torch.no_grad():
            N = D_F.shape[0]
            train_metrics = {f'Train/GAME{i}': 0.0 for i in range(4)}
            train_metrics['Train/MSE_sq'] = 0.0 
            train_metrics['Train/RE'] = 0.0

            for i in range(N):
                target_shape = D_F.shape[2:]
                sample_target = torch.zeros(target_shape, device=self.device)
                
                # 下采样映射
                sample_points = (points[i] / self.args.downsample_ratio).long()
                valid = (sample_points[:, 0] < target_shape[1]) & (sample_points[:, 1] < target_shape[0])
                sample_points = sample_points[valid]
                sample_target[sample_points[:, 1], sample_points[:, 0]] = 1.0
                
                sample_output = D_F[i:i+1]
                
                for L in range(4):
                    abs_err, sq_err = eval_game(sample_output, sample_target, L)
                    train_metrics[f'Train/GAME{L}'] += abs_err
                    if L == 0:
                        train_metrics['Train/MSE_sq'] += sq_err
                train_metrics['Train/RE'] += eval_relative(sample_output, sample_target)

            # 自动同步记录
            for k, v in train_metrics.items():
                self.log(k, v / N, on_step=False, on_epoch=True, sync_dist=True,batch_size=N)

        self.log('Train/Loss_Total', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss

    def on_train_epoch_end(self):
        # 仅主卡打印
        if self.trainer.is_global_zero:
            metrics = self.trainer.callback_metrics
            loss = metrics.get('Train/Loss_Total', torch.tensor(0.0)).item()
            g0 = metrics.get('Train/GAME0', torch.tensor(0.0)).item()
            g1 = metrics.get('Train/GAME1', torch.tensor(0.0)).item()
            g2 = metrics.get('Train/GAME2', torch.tensor(0.0)).item()
            g3 = metrics.get('Train/GAME3', torch.tensor(0.0)).item()
            rmse = np.sqrt(metrics.get('Train/MSE_sq', torch.tensor(0.0)).item())
            re = metrics.get('Train/RE', torch.tensor(0.0)).item()
            
            logger.info(f"[Epoch {self.current_epoch} Train] Loss: {loss:.4f} | GAME0: {g0:.3f} | GAME1: {g1:.3f} | GAME2: {g2:.3f} | GAME3: {g3:.3f} | RMSE: {rmse:.3f} | RE: {re:.4f}")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        prefix = "Test"
        if self.args.dataset == 'RGBTCC' and dataloader_idx == 0:
            prefix = "Val"

        # 延迟评估逻辑：记录 inf 以防止 Checkpoint 报错
        if self.current_epoch < self.args.val_start:
            dummy_metrics = {f'{prefix}/GAME{i}': float('inf') for i in range(4)}
            dummy_metrics.update({f'{prefix}/MSE': float('inf'), f'{prefix}/RelativeError': float('inf')})
            self.log_dict(dummy_metrics, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False,batch_size=1)
            return
            
        inputs, target, name = batch
        if isinstance(inputs, list):
            inputs = [x.to(self.device) for x in inputs]
        else:
            inputs = inputs.to(self.device)
        target = target.to(self.device)

        outputs = self(inputs)
        
        metrics = {}
        for L in range(4):
            abs_error, square_error = eval_game(outputs, target, L)
            metrics[f'{prefix}/GAME{L}'] = torch.tensor(abs_error, dtype=torch.float32, device=self.device)
            if L == 0:
                metrics[f'{prefix}/MSE'] = torch.tensor(square_error, dtype=torch.float32, device=self.device)

        metrics[f'{prefix}/RelativeError'] = torch.tensor(eval_relative(outputs, target), dtype=torch.float32, device=self.device)
        # 记录真实评估指标
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False, batch_size=1)
    
    def on_validation_epoch_end(self):
        # 排除模型启动前默认的 Sanity Check 阶段
        if self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
            
        metrics = self.trainer.callback_metrics
        for stage in ["Val", "Test"]:
            g0 = metrics.get(f'{stage}/GAME0')
            # 过滤掉填充的 inf 指标
            if g0 is not None and g0.item() != float('inf'):
                g1 = metrics.get(f'{stage}/GAME1').item()
                g2 = metrics.get(f'{stage}/GAME2').item()
                g3 = metrics.get(f'{stage}/GAME3').item()
                rmse = np.sqrt(metrics.get(f'{stage}/MSE').item())
                re = metrics.get(f'{stage}/RelativeError').item()
                
                logger.info(f"[Epoch {self.current_epoch} {stage}] GAME0: {g0.item():.3f} | GAME1: {g1:.3f} | GAME2: {g2:.3f} | GAME3: {g3:.3f} | RMSE: {rmse:.3f} | RE: {re:.4f}")

    def setup(self, stage=None):
        args = self.args
        if args.dataset == 'RGBTCC':
            self.datasets = {x: Crowd_RGBTCC(os.path.join(args.data_dir, x), args.crop_size, args.downsample_ratio, x) for x in ['train', 'val', 'test']}
        elif args.dataset == 'DroneRGBT':
            self.datasets = {
                'train': Crowd_Drone(os.path.join(args.data_dir, 'Train'), args.crop_size, args.downsample_ratio, 'train'),
                'test': Crowd_Drone(os.path.join(args.data_dir, 'Test'), args.crop_size, args.downsample_ratio, 'test')
            }

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], collate_fn=train_collate, batch_size=self.args.batch_size, 
                          shuffle=True, num_workers=self.args.num_workers, pin_memory=True)

    def val_dataloader(self):
        loaders = []
        if self.args.dataset == 'RGBTCC':
            loaders.append(DataLoader(self.datasets['val'], batch_size=1, shuffle=False, num_workers=self.args.num_workers))
        loaders.append(DataLoader(self.datasets['test'], batch_size=1, shuffle=False, num_workers=self.args.num_workers))
        return loaders