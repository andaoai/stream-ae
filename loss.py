"""
损失函数模块

该模块包含了流式视频自编码器中使用的各种损失函数。
主要包括全局损失函数，用于衡量重建质量。
"""

import torch
import torch.nn.functional as F


def compute_global_loss(curr_frame, reconstruction):
    """
    全局损失：整体图像重建质量
    结合多种损失函数确保整体embedding质量
    
    Args:
        curr_frame (torch.Tensor): 当前帧，形状[B,C,H,W]
        reconstruction (torch.Tensor): 重建帧，形状[B,C,H,W]
    
    Returns:
        tuple: (global_loss, mse_loss, l1_loss, ssim_loss)
    """
    # 1. MSE损失 - 使用mean而不是sum，避免损失值过大
    mse_loss = F.mse_loss(reconstruction, curr_frame, reduction='mean')
    
    # 2. L1损失 - 提供更稳定的梯度
    l1_loss = F.l1_loss(reconstruction, curr_frame, reduction='mean')
    
    # 3. 简化的SSIM损失（结构相似性）
    def compute_ssim_loss(img1, img2, window_size=11):
        try:
            # 简化的SSIM计算
            mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
            mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
            sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
            sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
            
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return 1 - torch.mean(ssim)
        except:
            # 如果SSIM计算失败，返回0
            return torch.tensor(0.0, device=img1.device)
    
    ssim_loss = compute_ssim_loss(curr_frame, reconstruction)
    
    # 组合全局损失 - 平衡不同损失项
    global_loss = mse_loss + 0.5 * l1_loss + 0.1 * ssim_loss
    
    return global_loss, mse_loss, l1_loss, ssim_loss