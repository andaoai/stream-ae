"""
模型架构模块

该模块包含了流式视频自编码器的所有模型组件。
包括编码器、解码器、变化检测器等核心组件。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import time


class LayerNormalization(nn.Module):
    """
    自定义层归一化模块

    实现了动态形状的层归一化，适用于流式处理中可能变化的输入尺寸。
    相比标准的LayerNorm，该实现：
    - 自动适应输入张量的形状
    - 无需预先指定归一化维度
    - 适合流式数据处理场景

    在streaming-drl项目的启发下，该归一化方法有助于：
    - 稳定在线学习过程
    - 减少内部协变量偏移
    - 提高模型对不同输入分布的适应性
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        """
        前向传播

        Args:
            input (torch.Tensor): 输入张量，任意形状

        Returns:
            torch.Tensor: 归一化后的张量，形状与输入相同
        """
        return F.layer_norm(input, input.size())

    def extra_repr(self) -> str:
        return "Dynamic Layer Normalization for Streaming Data"


class PixelChangeDetector:
    """
    像素变化检测器

    该类实现了智能的帧间变化检测，是流式视频处理的核心组件。
    借鉴streaming-drl项目中的变化检测思想，用于：

    1. 运动检测 - 识别视频中的运动区域
    2. 注意力机制 - 引导模型关注变化区域
    3. 计算优化 - 减少对静态区域的重复计算
    4. 损失加权 - 为变化区域分配更高的学习权重

    技术特性：
    - 多通道差异计算
    - 自适应阈值检测
    - 空间平滑滤波
    - 变化强度量化
    """

    def __init__(self, threshold=0.1, spatial_kernel_size=3):
        """
        初始化像素变化检测器

        Args:
            threshold (float): 变化检测阈值，范围[0,1]
                - 较小值：检测微小变化，适合精细场景
                - 较大值：仅检测显著变化，适合粗糙场景
            spatial_kernel_size (int): 空间平滑核大小
                - 用于减少噪声影响
                - 奇数值，推荐3或5
        """
        self.threshold = threshold
        self.spatial_kernel_size = spatial_kernel_size

    def detect_changes(self, prev_frame, curr_frame):
        """
        检测两帧之间的像素变化

        该方法实现了多层次的变化检测：
        1. 像素级差异计算
        2. 多通道信息融合
        3. 阈值化二值检测
        4. 空间平滑处理

        Args:
            prev_frame (torch.Tensor): 前一帧，形状[C,H,W]
            curr_frame (torch.Tensor): 当前帧，形状[C,H,W]

        Returns:
            tuple: (change_mask, change_intensity)
                - change_mask: 二值变化掩码，形状[1,H,W]
                - change_intensity: 变化强度图，形状[1,H,W]
        """
        if prev_frame is None:
            return torch.ones_like(curr_frame[:1]), torch.ones_like(curr_frame[:1])
        
        # 计算像素差异
        pixel_diff = torch.abs(curr_frame - prev_frame)
        
        # 计算变化强度（所有通道的平均差异）
        change_intensity = torch.mean(pixel_diff, dim=0, keepdim=True)
        
        # 生成变化掩码
        change_mask = (change_intensity > self.threshold).float()
        
        # 空间平滑（考虑邻域像素）
        if self.spatial_kernel_size > 1:
            # 确保change_mask是单通道的
            if change_mask.shape[1] > 1:
                change_mask = torch.mean(change_mask, dim=1, keepdim=True)

            kernel = torch.ones(1, 1, self.spatial_kernel_size, self.spatial_kernel_size) / (self.spatial_kernel_size ** 2)
            kernel = kernel.to(change_mask.device)
            change_mask = F.conv2d(change_mask, kernel, padding=self.spatial_kernel_size//2)
            change_mask = (change_mask > 0.3).float()  # 重新二值化
        
        return change_mask, change_intensity


class ResidualBlock(nn.Module):
    """ResNet风格的残差块"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = LayerNormalization()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = LayerNormalization()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual  # 残差连接
        return self.relu(out)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块 - 用于深层特征处理"""
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    """将特征图转换为patch序列"""
    def __init__(self, in_channels, embed_dim, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, embed_dim, H//patch_size, W//patch_size]
        x = self.proj(x)
        B, C, H, W = x.shape
        # Flatten to sequence: [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)
        return x, (H, W)


class PatchReconstruction(nn.Module):
    """将patch序列重建为特征图"""
    def __init__(self, embed_dim, out_channels, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, out_channels, patch_size, patch_size)

    def forward(self, x, hw_shape):
        H, W = hw_shape
        B, N, C = x.shape
        # Reshape to feature map: [B, N, C] -> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)
        return x


class Encoder(nn.Module):
    """
    简化的单一路径编码器

    该编码器实现了三个并行分支，每个分支使用不同大小的卷积核来捕获不同类型的特征：
    - 小卷积分支 (3×3): 专注于纹理特征
    - 中卷积分支 (5×5): 专注于平衡特征  
    - 大卷积分支 (7×7): 专注于结构特征和小目标

    核心设计原则：
    1. 零填充策略：所有卷积层不使用padding，保持边缘信息的真实性
    2. 灵活Embedding尺寸：每个分支的embedding尺寸独立计算，无需完全一致
    3. 整数下采样：确保输出尺寸为整数，避免特征图变形
    4. 独立通道设计：每个分支的通道数可以根据其特点进行调整

    输出尺寸：
    - 小卷积分支: 224×224×3 → 27×27×3 (压缩比 92:1)
    - 中卷积分支: 224×224×3 → 25×25×2 (压缩比 161:1)
    - 大卷积分支: 224×224×3 → 23×23×2 (压缩比 190:1)
    - 最终输出: 224×224×3 → 128维embedding (压缩比 1176:1)

    单一路径设计：
    - 三个并行分支捕获多尺度特征
    - 扁平化拼接为单一特征向量
    - 通过MLP投影到128维紧凑表示
    """

    def __init__(self, input_channels=3, latent_channels=4):
        super().__init__()
        
        # 小卷积分支 - 纹理特征 (3×3, 无padding)
        self.small_kernel_branch = nn.Sequential(
            # 224×224×3 → 111×111×16
            nn.Conv2d(3, 16, 3, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),
            
            # 111×111×16 → 55×55×12
            nn.Conv2d(16, 12, 3, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),
            
            # 55×55×12 → 27×27×3
            nn.Conv2d(12, 3, 3, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU()
        )  # 输出: 27×27×3
        
        # 中卷积分支 - 平衡特征 (5×5, 无padding)
        self.medium_kernel_branch = nn.Sequential(
            # 224×224×3 → 110×110×16
            nn.Conv2d(3, 16, 5, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),
            
            # 110×110×16 → 53×53×12
            nn.Conv2d(16, 12, 5, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),
            
            # 53×53×12 → 25×25×2
            nn.Conv2d(12, 2, 5, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU()
        )  # 输出: 25×25×2
        
        # 大卷积分支 - 结构特征 (7×7, 无padding)
        self.large_kernel_branch = nn.Sequential(
            # 224×224×3 → 109×109×16
            nn.Conv2d(3, 16, 7, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),
            
            # 109×109×16 → 52×52×12
            nn.Conv2d(16, 12, 7, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),
            
            # 52×52×12 → 23×23×2
            nn.Conv2d(12, 2, 7, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU()
        )  # 输出: 23×23×2
        
        # MLP投影层：将多尺度特征投影到128维
        self.embedding_mlp = nn.Sequential(
            nn.Linear(4495, 1024),  # 27*27*3 + 25*25*2 + 23*23*2 = 2187 + 1250 + 1058 = 4495
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128)
        )
    
    def forward(self, x):
        # 三个并行分支提取多尺度特征
        small_emb = self.small_kernel_branch(x)    # 27×27×3
        medium_emb = self.medium_kernel_branch(x)  # 25×25×2
        large_emb = self.large_kernel_branch(x)    # 23×23×2
        
        # 扁平化三个分支的embedding
        small_flat = small_emb.flatten(start_dim=1)  # B x 2187
        medium_flat = medium_emb.flatten(start_dim=1)  # B x 1250
        large_flat = large_emb.flatten(start_dim=1)  # B x 1058
        
        # 拼接所有特征
        combined_flat = torch.cat([small_flat, medium_flat, large_flat], dim=1)  # B x 4495
        
        # MLP投影到128维
        embedding_128 = self.embedding_mlp(combined_flat)  # B x 128
        
        return embedding_128


class Decoder(nn.Module):
    """
    简化的单一路径解码器

    该解码器从128维embedding重建原始图像：
    - MLP扩展层：将128维embedding扩展到多尺度特征
    - 三个并行解码分支：对应编码器的三个分支
    - 特征融合：将多尺度重建结果融合为最终图像

    核心特性：
    1. 单一输入：只接受128维embedding作为输入
    2. 对称设计：解码分支与编码器分支结构对称
    3. 无填充策略：所有转置卷积不使用padding，保持边缘真实性
    4. 最终sigmoid：只在最终输出层应用sigmoid激活函数
    5. 尺寸统一：使用双线性插值将不同尺寸的重建图像统一到224×224

    重建策略：
    1. MLP扩展：将128维embedding扩展到4495维
    2. 特征分割：分割为三个分支的特征
    3. 并行解码：三个分支同时解码
    4. 尺寸统一：将三个重建图像上采样到224×224
    5. 特征融合：使用1×1卷积进行特征融合和降维
    6. 最终激活：应用sigmoid确保输出范围[0,1]
    """

    def __init__(self, latent_channels=4, output_channels=3):
        super().__init__()
        
        # MLP扩展层：将128维embedding扩展到多尺度特征
        self.embedding_mlp_expand = nn.Sequential(
            nn.Linear(128, 256),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 512),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 1024),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 4495)  # 扩展到拼接维度
        )
        
        # 小卷积分支解码器 (纹理特征)
        self.small_decoder = nn.Sequential(
            # 27×27×3 → 55×55×12
            nn.ConvTranspose2d(3, 12, 3, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),
            
            # 55×55×12 → 111×111×16
            nn.ConvTranspose2d(12, 16, 3, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),
            
            # 111×111×16 → 223×223×3
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=0)  # 无padding，无sigmoid
        )  # 输出: 223×223×3
        
        # 中卷积分支解码器 (平衡特征)
        self.medium_decoder = nn.Sequential(
            # 25×25×2 → 53×53×12
            nn.ConvTranspose2d(2, 12, 5, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),
            
            # 53×53×12 → 109×109×16
            nn.ConvTranspose2d(12, 16, 5, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),
            
            # 109×109×16 → 221×221×3
            nn.ConvTranspose2d(16, 3, 5, stride=2, padding=0)  # 无padding，无sigmoid
        )  # 输出: 221×221×3
        
        # 大卷积分支解码器 (结构特征)
        self.large_decoder = nn.Sequential(
            # 23×23×2 → 51×51×12
            nn.ConvTranspose2d(2, 12, 7, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),
            
            # 51×51×12 → 107×107×16
            nn.ConvTranspose2d(12, 16, 7, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),
            
            # 107×107×16 → 219×219×3
            nn.ConvTranspose2d(16, 3, 7, stride=2, padding=0)  # 无padding，无sigmoid
        )  # 输出: 219×219×3
        
        # 自适应尺寸融合模块 - 直接合并特征，最后应用sigmoid
        self.adaptive_fusion = nn.Sequential(
            nn.Conv2d(9, 6, 1),  # 融合三个重建结果
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Conv2d(6, 3, 1),  # 最终输出，无sigmoid
        )
    
    def forward(self, embedding_128):
        # 从128维embedding重建三个分支特征
        expanded_flat = self.embedding_mlp_expand(embedding_128)  # B x 4495
        
        # 分割为三个分支
        small_flat = expanded_flat[:, :2187]  # B x 2187 (27*27*3)
        medium_flat = expanded_flat[:, 2187:2187+1250]  # B x 1250 (25*25*2)
        large_flat = expanded_flat[:, 2187+1250:]  # B x 1058 (23*23*2)
        
        # Reshape到原始的spatial dimensions
        batch_size = embedding_128.shape[0]
        small_emb = small_flat.view(batch_size, 3, 27, 27)  # B x 3 x 27 x 27
        medium_emb = medium_flat.view(batch_size, 2, 25, 25)  # B x 2 x 25 x 25
        large_emb = large_flat.view(batch_size, 2, 23, 23)  # B x 2 x 23 x 23
        
        # 并行解码
        small_recon = self.small_decoder(small_emb)
        medium_recon = self.medium_decoder(medium_emb)
        large_recon = self.large_decoder(large_emb)
        
        # 统一尺寸到224×224
        small_recon_up = F.interpolate(small_recon, size=(224, 224), mode='bilinear', align_corners=False)
        medium_recon_up = F.interpolate(medium_recon, size=(224, 224), mode='bilinear', align_corners=False)
        large_recon_up = F.interpolate(large_recon, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 通道拼接 (224×224×9)
        combined = torch.cat([small_recon_up, medium_recon_up, large_recon_up], dim=1)
        final_output = self.adaptive_fusion(combined)
        
        # 只在最终输出应用sigmoid
        final_output = torch.sigmoid(final_output)
        
        return final_output