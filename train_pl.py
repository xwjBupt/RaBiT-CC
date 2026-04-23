import argparse
import os
import torch
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from loguru import logger
from pytorch_lightning.strategies import DDPStrategy
import subprocess  # <--- 新增：用于执行系统命令
import torch.distributed as dist
from pytorch_lightning.callbacks import TQDMProgressBar

# === 激活 Ampere/Ada 架构的 TF32 矩阵乘法加速 (L40 算力解放) ===
torch.set_float32_matmul_precision('high') 

# 导入上面的 LightningModule
from lightning_trainer import CrowdCountingLightningModule 

# === 新增：自定义纯净版进度条 ===
class CleanProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
            # 默认会返回各种 loss 和 v_num，我们直接返回空字典，截断所有文字！
        return {}


def parse_args():
    parser = argparse.ArgumentParser(description='Train with PyTorch Lightning')
    parser.add_argument('--exp-tag', default='rabbit', help='实验标签')
    parser.add_argument('--dataset', default='RGBTCC', choices=['RGBTCC', 'DroneRGBT'])
    parser.add_argument('--data-dir', default='/home/wjx/data/CrowdCounting/RGBTCC_Pro')    
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--resume', default='', help='断点续传路径 (ckpt)')
    
    # 支持多卡，默认 [0, 1] 等。可根据实际需要修改
    parser.add_argument('--devices', default=[0], nargs='+', type=int, help='GPU ID列表')
    
    parser.add_argument('--crop-size', type=int, default=256)
    
    # model setting
    parser.add_argument('--constr-hg', type=str, default='threshold', choices=['threshold', 'knn'])
    parser.add_argument('--constr-k', type=int, default=4)
    parser.add_argument('--constr-threshold', type=float, default=0.8)

    # training setting
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--val-epoch', type=int, default=1)
    parser.add_argument('--val-start', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--downsample-ratio', type=int, default=8)
    parser.add_argument('--use-background', type=bool, default=True)
    parser.add_argument('--sigma', type=float, default=8.0)
    parser.add_argument('--background-ratio', type=float, default=0.15)
    parser.add_argument('--nodebug', action='store_true', help='如果不加此参数，则为debug模式不记录代码；加上则自动进行 git commit')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if "EXP_TIMESTAMP" not in os.environ:
        os.environ["EXP_TIMESTAMP"] = datetime.now().strftime('%y%m%d-%H%M%S')
    if args.nodebug:
        root = os.path.dirname(os.path.realpath(__file__)) + '/output'
    else:
        root = os.path.dirname(os.path.realpath(__file__)) + '/output_debug'
        args.val_start = 1

    # 1. 统一构建输出路径 (保持原有格式: exp_tag@时间戳)
    sub_dir = f"{args.exp_tag}_{args.dataset}@{datetime.now().strftime('%y%m%d-%H%M%S')}"
    args.exp_tag = sub_dir
    final_save_dir = os.path.join(root, sub_dir)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        os.makedirs(final_save_dir, exist_ok=True)
        # 2. 配置 loguru
        # enqueue=True 用于确保在 DDP 多卡/多进程模型下日志写入的安全
        logger.add(
            os.path.join(final_save_dir, 'train.log'), 
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", 
            level="INFO",
            enqueue=True 
        )
        if not args.nodebug:
            # Debug 模式，使用 warning 级别醒目提示
            logger.warning("⚠️ [当前为 Debug 模式] 不会对代码进行 Git 记录。")
        else:
            logger.info(f"🚀 [非 Debug 模式] 正在进行 Git 代码版本记录 (Commit: {args.exp_tag})...")
            try:
                status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
                
                if status.stdout.strip():
                    subprocess.run(["git", "add", "."], check=True)
                    subprocess.run(["git", "commit", "-m", f"Auto-commit [Exp]: {args.exp_tag}"], check=True)
                    logger.info("✅ Git 代码版本已成功记录！")
                else:
                    logger.info("ℹ️ Git 工作区干净，没有需要 Commit 的更改。")
            except subprocess.CalledProcessError as e:
                # 使用 error 级别记录异常
                logger.error(f"❌ Git 记录失败 (可能是没有初始化 Git 或发生冲突)。错误: {e}")
            except Exception as e:
                logger.error(f"❌ 发生未知错误，跳过 Git 记录: {e}")
        # ==========================================================

        logger.info(f"\n\n\n>>>>>>>>>> Starting experiment: {sub_dir} <<<<<<<<<<\n\n\n")
        logger.info("================ Configurations ================")
        for k, v in args.__dict__.items():
            logger.info(f"{k}: {v}")
        logger.info("================================================")

    # 3. 设定随机种子和开启 CuDNN benchmark 以提升训练速度
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.benchmark = True
    
    # 4. 初始化 Lightning 模型
    model = CrowdCountingLightningModule(args)
    
    # 5. 配置 TensorBoard Logger
    # version='' 使得日志文件直接写在 final_save_dir 目录下
    tb_logger = TensorBoardLogger(save_dir=final_save_dir, name=sub_dir, version='')
    # 监控学习率
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 6. 配置 Model Checkpoints，将其直接存入指定的最终目录中
    checkpoint_callbacks = [
        ModelCheckpoint(dirpath=final_save_dir, monitor='Test/GAME0', mode='min', filename='best-game0-{epoch:02d}', save_top_k=1),
        ModelCheckpoint(dirpath=final_save_dir, monitor='Test/GAME1', mode='min', filename='best-game1-{epoch:02d}', save_top_k=1),
        ModelCheckpoint(dirpath=final_save_dir, monitor='Test/GAME2', mode='min', filename='best-game2-{epoch:02d}', save_top_k=1),
        ModelCheckpoint(dirpath=final_save_dir, monitor='Test/GAME3', mode='min', filename='best-game3-{epoch:02d}', save_top_k=1),
        ModelCheckpoint(dirpath=final_save_dir, monitor='Test/MSE',   mode='min', filename='best-mse-{epoch:02d}', save_top_k=1),
        ModelCheckpoint(dirpath=final_save_dir, save_last=True),
    ]
    
    
    callbacks_list = checkpoint_callbacks + [lr_monitor, CleanProgressBar()]
    # 7. 初始化 Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        # 强制开启未使用参数的探测，解决复杂模型 DDP 报错
        strategy=DDPStrategy(
            find_unused_parameters=True, 
            gradient_as_bucket_view=True
        ) if len(args.devices) > 1 or args.devices == [-1] else "auto",
        max_epochs=args.max_epoch,
        check_val_every_n_epoch=args.val_epoch,
        num_sanity_val_steps=0,  # 屏蔽训练前默认跑 2 个 val_batch 的检查
        log_every_n_steps=5,
        logger=tb_logger,
        callbacks=callbacks_list,
        benchmark=True,
    )
    
    # 若提供断点，进行续传
    trainer.fit(model, ckpt_path=args.resume if args.resume else None)

    if torch.cuda.is_available() and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        
    if local_rank == 0:
        logger.info(f"\n\n\n>>>>>>>>>> Ending experiment: {sub_dir} <<<<<<<<<<\n\n\n")