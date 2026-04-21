import cv2
import numpy as np
import torch

def eval_game(output, target, L=0):
    # ================= 修改 1：鲁棒的维度解析 =================
    # 原代码: output[0][0][0] 非常脆弱。当传入 [1, 1, H, W] 时，它会切片出 1D 向量导致报错。
    # 修复：判断是否为 tuple (模型返回多分支时)，只取第一个(D_F)。然后用 squeeze() 智能去除多余维度。
    if isinstance(output, (tuple, list)):
        pred = output[0]  # 取出 D_F
    else:
        pred = output
    # squeeze() 会把 [1, 1, H, W] 或 [1, H, W] 统一压平为 2D 的 (H_pred, W_pred)
    pred = pred.detach().cpu().squeeze().numpy()

    # 同样鲁棒地处理 target
    if isinstance(target, (tuple, list)):
        gt = target[0]
    else:
        gt = target
    gt = gt.detach().cpu().squeeze().numpy()

    H, W = gt.shape
    H_pred, W_pred = pred.shape

    # ================= 修改 2：修复长宽缩放比例 Bug =================
    # 原代码: ratio = H / output.shape[1] 
    # Bug分析: output.shape[1] 是宽度 W！用高度 H 去除以宽度 W 是错的（除非你的图全是正方形）。
    # 修复：分别计算高和宽的缩放比例。
    ratio_h = H / H_pred
    ratio_w = W / W_pred
    
    # 保持密度图积分(人头数)不变：上采样后除以面积放大的倍数
    pred_resized = cv2.resize(pred, (W, H), interpolation=cv2.INTER_CUBIC) / (ratio_h * ratio_w)

    assert pred_resized.shape == gt.shape

    p = pow(2, L)
    abs_error = 0
    square_error = 0
    for i in range(p):
        for j in range(p):
            output_block = pred_resized[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]
            target_block = gt[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]

            # 为了避免精度溢出，提前计算差值
            err = output_block.sum() - target_block.sum()
            abs_error += abs(err)
            square_error += err ** 2

    return abs_error, square_error


def eval_relative(output, target):
    # ================= 修改 3：修复指标暴涨的致命 BUG =================
    # 原代码: for i in output: output_num = output_num+i.cpu().sum().item()
    # Bug分析: 你的模型返回 (D_F, D_R, D_T, reli_maps_R, reli_maps_T)。
    # 原代码用 for 循环把这些辅助密度图和可靠性权重图 全 部 加 起 来 算作预测人头数！
    # 这会导致你的 RE 指标极高且毫无意义。
    # 修复：和 GAME 指标一样，只取 D_F 融合特征图计算。
    if isinstance(output, (tuple, list)):
        pred = output[0]
    else:
        pred = output
        
    output_num = pred.detach().cpu().sum().item()
    target_num = target.detach().cpu().sum().item()
    
    # ================= 修改 4：增加除零保护 =================
    # 如果某张测试图片里刚好没有行人（target_num = 0），原代码直接触发 ZeroDivisionError 崩溃
    if target_num == 0:
        if output_num == 0:
            return 0.0
        else:
            target_num = 1e-4  # 给予一个极小值避免崩溃
            
    relative_error = abs(output_num - target_num) / target_num
    return relative_error