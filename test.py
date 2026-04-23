import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from loguru import logger
import sys
from torch.utils.data import DataLoader
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# 引入已有的模块
from lightning_trainer import CrowdCountingLightningModule
from utils.evaluation import eval_game, eval_relative
from datasets.crowd_drone import Crowd_Drone
from datasets.crowd_rgbtcc import Crowd_RGBTCC

def parse_args():
    parser = argparse.ArgumentParser(description='Test Crowd Counting Model on Full Dataset')
    # 必填项：指定的单个模型权重
    parser.add_argument('--ckpt-path', type=str, required=True, help='训练好的模型权重路径 (.ckpt)')
    parser.add_argument('--data-dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--dataset', type=str, default='DroneRGBT', choices=['RGBTCC', 'DroneRGBT'])
    
    # 【新增】：测试时的 batch_size，默认 8（L40 可以轻松跑到 16）
    parser.add_argument('--batch-size', type=int, default=1, help='测试时的批量大小')
    
    # 选填项：默认开启可视化保存
    parser.add_argument('--no-vis', action='store_true', help='加上此参数则不保存可视化的密度图')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 确定保存目录 (就在传入的 ckpt 所在的目录)
    save_dir = os.path.dirname(os.path.abspath(args.ckpt_path))
    ckpt_name = os.path.basename(args.ckpt_path)
    
    # 建立当前权重的专属可视化文件夹，避免多次测试图片混在一起
    vis_dir = os.path.join(save_dir, f'visualizations_{os.path.splitext(ckpt_name)[0]}')
    if not args.no_vis:
        os.makedirs(vis_dir, exist_ok=True)
        
    # 初始化 Logger
    logger.add(os.path.join(save_dir, 'test_evaluation.log'), level="INFO")
    logger.info(f"🚀 开始加载模型权重: {args.ckpt_path}")
    
    # 2. 从 Checkpoint 加载模型 (strict=False 避免坐标缓存报错)
    model = CrowdCountingLightningModule.load_from_checkpoint(args.ckpt_path, strict=False)
    model.eval()
    
    # 挂载到 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 3. 准备测试数据集
    logger.info(f"📂 正在加载数据集: {args.dataset}")
    if args.dataset == 'DroneRGBT':
        test_dataset = Crowd_Drone(args.data_dir, model.args.crop_size, model.args.downsample_ratio, 'test')
    elif args.dataset == 'RGBTCC':
        test_dataset = Crowd_RGBTCC(args.data_dir, model.args.crop_size, model.args.downsample_ratio, 'test')
        
    # 强力防空军拦截器
    if len(test_dataset) == 0:
        logger.error(f"❌ 在路径 [{args.data_dir}] 下连一张测试图片都没找到！请检查路径大小写。")
        raise ValueError("Dataset is empty.")

    # 使用传入的 batch_size 进行极速读取
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    num_samples = len(test_dataset)
    
    # 4. 初始化指标累加器
    metrics = {f'GAME{i}': 0.0 for i in range(4)}
    metrics['MSE_sq'] = 0.0
    metrics['RE'] = 0.0
    
    logger.info(f"🔍 开始对 {num_samples} 张图像进行推理测试 (Batch Size: {args.batch_size})...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Images"):
            inputs, target, names = batch 
            
            # 设备挂载
            if isinstance(inputs, list):
                inputs_dev = [x.to(device) for x in inputs]
            else:
                inputs_dev = inputs.to(device)
            target_dev = target.to(device)
            
            # ==================== 1. 批量极致推理 ====================
            outputs = model(inputs_dev)
            
            # 剥离出最终预测的高精度密度图 D_F
            pred_densities = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            
            # 获取当前 batch 的实际大小
            B = target_dev.shape[0]
            
            # ==================== 2. 逐图解包算指标和保存 ====================
            for i in range(B):
                single_name = names[i]
                
                # 切片取出单张图的预测和 target，保持维度
                single_pred = pred_densities[i:i+1]
                single_target = target_dev[i:i+1]
                
                # 计算单张图的指标并累加
                for L in range(4):
                    abs_err, sq_err = eval_game(single_pred, single_target, L)
                    metrics[f'GAME{L}'] += abs_err
                    if L == 0:
                        metrics['MSE_sq'] += sq_err
                metrics['RE'] += eval_relative(single_pred, single_target)
                
                # 可视化单张图
                if not args.no_vis:
                    # 提取单张密度图并转为 numpy
                    pred_map = single_pred.squeeze().cpu().numpy()
                    
                    pred_count = np.sum(pred_map)
                    gt_count = single_target.sum().item()
                    
                    # 提取对应的 RGB 原图并反归一化
                    rgb_tensor = inputs[0][i].cpu() if isinstance(inputs, list) else inputs[i].cpu()
                    mean = torch.tensor([0.407, 0.389, 0.396]).view(3, 1, 1)
                    std = torch.tensor([0.241, 0.246, 0.242]).view(3, 1, 1)
                    
                    rgb_img = rgb_tensor * std + mean
                    rgb_img = rgb_img.numpy().transpose(1, 2, 0)
                    rgb_img = np.clip(rgb_img, 0, 1)
                    
                    # 调整密度图尺寸以匹配原图
                    H, W = rgb_img.shape[:2]
                    pred_map_resized = cv2.resize(pred_map, (W, H), interpolation=cv2.INTER_CUBIC)
                    
                    # 绘制图像
                    plt.figure(figsize=(10, 8))
                    plt.imshow(rgb_img)
                    plt.imshow(pred_map_resized, cmap='jet', alpha=0.5)
                    plt.title(f"Image: {single_name} | GT: {gt_count:.1f} | Pred: {pred_count:.1f}", fontsize=14)
                    plt.axis('off')
                    
                    # 保存图像
                    save_path = os.path.join(vis_dir, f"{single_name}_Pred_{pred_count:.1f}_GT_{gt_count:.1f}.png")
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
                    plt.close()
                
    # 6. 汇总计算最终平均指标
    final_metrics = {}
    for L in range(4):
        final_metrics[f'GAME{L}'] = metrics[f'GAME{L}'] / num_samples
    final_metrics['RMSE'] = np.sqrt(metrics['MSE_sq'] / num_samples)
    final_metrics['RE'] = metrics['RE'] / num_samples
    
    # 7. 打印并记录最终成绩单
    logger.info("=" * 50)
    logger.info(f"🏆 Final Test Results on [{args.dataset}]")
    logger.info(f"GAME0 (MAE): {final_metrics['GAME0']:.3f}")
    logger.info(f"GAME1:       {final_metrics['GAME1']:.3f}")
    logger.info(f"GAME2:       {final_metrics['GAME2']:.3f}")
    logger.info(f"GAME3:       {final_metrics['GAME3']:.3f}")
    logger.info(f"RMSE:        {final_metrics['RMSE']:.3f}")
    logger.info(f"RelativeErr: {final_metrics['RE']:.4f}")
    logger.info("=" * 50)
    
    # 将汇总结果写入专属的 txt 报告文件
    res_file = os.path.join(save_dir, f"test_summary_{os.path.splitext(ckpt_name)[0]}.txt")
    with open(res_file, "w", encoding='utf-8') as f:
        f.write(f"测试模型: {ckpt_name}\n")
        f.write(f"测试集:   {args.dataset}\n")
        f.write("-" * 30 + "\n")
        for k, v in final_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
            
    logger.info(f"✅ 数值测试报告已保存至: {res_file}")
    if not args.no_vis:
        logger.info(f"🎨 {num_samples} 张测试图片的预测密度图已全部保存至: {vis_dir}")
    
    # ==================== 8. 写入 Excel 总表核心逻辑 ====================
    # 提取实验目录名称作为更清晰的标记 (例如: rabbit_drone@260421-200152)
    exp_dir_name = os.path.basename(save_dir.split('/')[-1])
    experiment_name = f"{exp_dir_name} # {ckpt_name}"
    
    excel_path =root = os.path.dirname(os.path.realpath(__file__))+ "/all_experiments_results.xlsx"  # 保存在项目根目录
    
    # 准备要写入的一行新数据
    new_data = {
        "Experiment Name": [experiment_name],
        "Dataset": [args.dataset],
        "GAME0": [round(final_metrics['GAME0'], 3)],
        "GAME1": [round(final_metrics['GAME1'], 3)],
        "GAME2": [round(final_metrics['GAME2'], 3)],
        "GAME3": [round(final_metrics['GAME3'], 3)],
        "RMSE": [round(final_metrics['RMSE'], 3)],
        "RelativeErr": [round(final_metrics['RE'], 4)]
    }
    df_new = pd.DataFrame(new_data)
    
    try:
        if os.path.exists(excel_path):
            # 如果文件存在，先读取原有数据，再追加新数据
            df_existing = pd.read_excel(excel_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(excel_path, index=False)
            logger.info(f"📊 测试结果已成功追加到 Excel 总表: {excel_path}")
        else:
            # 如果文件不存在，直接创建并写入表头和第一行数据
            df_new.to_excel(excel_path, index=False)
            logger.info(f"📊 已创建全新 Excel 总表并写入结果: {excel_path}")
    except PermissionError:
        logger.error(f"❌ 写入 Excel 失败！文件 {excel_path} 可能正在被另一个程序（如 Wps/Excel）打开，请关闭后重试。")
    except Exception as e:
        logger.error(f"❌ 写入 Excel 时发生未知错误: {e}")
    # ====================================================================

if __name__ == '__main__':
    main()