"""
数据预处理工具模块
负责帧预处理、数据格式转换等工具函数
"""

import cv2
import numpy as np
import torch
from config import DEVICE, DATA_CONFIG


def preprocess_frame(frame, target_size=None):
    """
    预处理游戏帧

    Args:
        frame: 原始帧数据
        target_size: 目标尺寸 (width, height)

    Returns:
        torch.Tensor: 预处理后的张量 [1, 3, H, W]
    """
    # 使用配置中的目标尺寸
    if target_size is None:
        target_size = DATA_CONFIG['frame_size']

    # 转换为RGB格式
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 调整尺寸
    frame = cv2.resize(frame, target_size)

    # 归一化到[0, 1]
    min_val, max_val = DATA_CONFIG['normalization_range']
    frame = frame.astype(np.float32) / 255.0

    # 转换为张量并添加批次维度
    tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)

    # 移动到GPU
    tensor = tensor.to(DEVICE)

    return tensor


def postprocess_output(tensor):
    """
    后处理模型输出
    
    Args:
        tensor: 模型输出张量 [1, 3, H, W]
    
    Returns:
        numpy.ndarray: 处理后的图像数组 [H, W, 3]
    """
    # 移除批次维度并转置
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 转换为numpy数组
    array = tensor.permute(1, 2, 0).cpu().numpy()
    
    # 确保值在[0, 1]范围内
    array = np.clip(array, 0, 1)
    
    # 转换为[0, 255]范围
    array = (array * 255).astype(np.uint8)
    
    return array


def compute_frame_difference(frame1, frame2):
    """
    计算两帧之间的差异
    
    Args:
        frame1: 第一帧
        frame2: 第二帧
    
    Returns:
        float: 帧差异值
    """
    if frame1 is None or frame2 is None:
        return 1.0
    
    # 计算MSE差异
    diff = torch.mean((frame1 - frame2) ** 2)
    return diff.item()


def normalize_tensor(tensor, min_val=0.0, max_val=1.0):
    """
    归一化张量到指定范围
    
    Args:
        tensor: 输入张量
        min_val: 最小值
        max_val: 最大值
    
    Returns:
        torch.Tensor: 归一化后的张量
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    
    if tensor_max - tensor_min > 0:
        normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
        return normalized * (max_val - min_val) + min_val
    else:
        return torch.zeros_like(tensor) + min_val


def create_gaussian_kernel(kernel_size=5, sigma=1.0):
    """
    创建高斯核
    
    Args:
        kernel_size: 核大小
        sigma: 标准差
    
    Returns:
        torch.Tensor: 高斯核
    """
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.0
    
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    
    kernel = torch.outer(g, g)
    return kernel.unsqueeze(0).unsqueeze(0)