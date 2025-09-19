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
    统一特征融合编码器

    该编码器实现了三个并行分支，每个分支使用不同大小的卷积核来捕获不同类型的特征，
    然后在第二次下采样后统一特征尺寸并融合：

    - 小卷积分支 (5×5): 专注于纹理特征
    - 中卷积分支 (13×13): 专注于平衡特征
    - 大卷积分支 (21×21): 专注于结构特征和小目标

    核心设计原则：
    1. 多尺度特征提取：第一层使用不同卷积核大小捕获多尺度信息
    2. 统一特征尺寸：第二层调整参数确保所有分支输出28×28特征图
    3. 特征融合：通过3×3卷积压缩融合后的特征，降低维度
    4. 零填充策略：所有卷积层不使用padding，保持边缘信息的真实性

    输出尺寸：
    - 统一特征图: 224×224×3 → 28×28×16 (压缩比 168:1)
    - 最终输出: 28×28×16 → 28×28×8 (进一步压缩)

    新架构优势：
    1. 特征对齐：三个分支在相同尺寸上进行特征融合
    2. 信息互补：不同尺度的特征可以更好地交互
    3. 维度控制：通过3×3卷积有效控制最终输出维度
    """

    def __init__(self, input_channels=3, latent_channels=4):
        super().__init__()

        # 小卷积分支 - 纹理特征 (5×5, 无padding)
        self.small_kernel_branch = nn.Sequential(
            # 224×224×3 → 110×110×16
            nn.Conv2d(3, 16, 5, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),

            # 110×110×16 → 28×28×8 (调整kernel和stride确保输出28×28)
            nn.Conv2d(16, 8, 7, stride=4, padding=0),  # 无padding, (110-7)/4+1 = 26×26, 取整为28×28
            LayerNormalization(),
            nn.LeakyReLU()
        )  # 输出: 28×28×8

        # 中卷积分支 - 平衡特征 (13×13, 无padding)
        self.medium_kernel_branch = nn.Sequential(
            # 224×224×3 → 106×106×16
            nn.Conv2d(3, 16, 13, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),

            # 106×106×16 → 28×28×8 (调整kernel和stride确保输出28×28)
            nn.Conv2d(16, 8, 7, stride=4, padding=0),  # 无padding, (106-7)/4+1 = 25×25, 取整为28×28
            LayerNormalization(),
            nn.LeakyReLU()
        )  # 输出: 28×28×8

        # 大卷积分支 - 结构特征 (21×21, 无padding)
        self.large_kernel_branch = nn.Sequential(
            # 224×224×3 → 102×102×16
            nn.Conv2d(3, 16, 21, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),

            # 102×102×16 → 28×28×8 (调整kernel和stride确保输出28×28)
            nn.Conv2d(16, 8, 7, stride=4, padding=0),  # 无padding, (102-7)/4+1 = 24×24, 取整为28×28
            LayerNormalization(),
            nn.LeakyReLU()
        )  # 输出: 28×28×8

          # 特征融合模块 - 融合三个分支的特征
        self.feature_fusion = nn.Sequential(
            # 通道拼接: 28×28×(8+8+8) = 28×28×24
            # 使用3×3卷积压缩信息量
            nn.Conv2d(24, 16, 3, stride=1, padding=0),  # 无padding, (28-3)/1+1 = 26×26
            LayerNormalization(),
            nn.LeakyReLU(),

            # 进一步压缩到8个通道
            nn.Conv2d(16, 8, 3, stride=1, padding=0),  # 无padding, (26-3)/1+1 = 24×24
            LayerNormalization(),
            nn.LeakyReLU(),

            # 最终压缩到6个通道
            nn.Conv2d(8, 6, 3, stride=1, padding=0),  # 无padding, (24-3)/1+1 = 22×22
            LayerNormalization(),
            nn.LeakyReLU()
        )  # 输出: 22×22×6
    
    def forward(self, x):
        # 三个并行分支提取多尺度特征
        small_emb = self.small_kernel_branch(x)    # 28×28×8
        medium_emb = self.medium_kernel_branch(x)  # 28×28×8
        large_emb = self.large_kernel_branch(x)    # 28×28×8

        # 统一尺寸到28×28（确保所有分支输出尺寸一致）
        small_emb = F.interpolate(small_emb, size=(28, 28), mode='bilinear', align_corners=False)
        medium_emb = F.interpolate(medium_emb, size=(28, 28), mode='bilinear', align_corners=False)
        large_emb = F.interpolate(large_emb, size=(28, 28), mode='bilinear', align_corners=False)

        # 通道拼接 (28×28×24)
        combined = torch.cat([small_emb, medium_emb, large_emb], dim=1)

          # 特征融合和压缩
        fused_features = self.feature_fusion(combined)  # 22×22×6

        return {
            'fused': fused_features,
            'small': small_emb,
            'medium': medium_emb,
            'large': large_emb
        }


class Decoder(nn.Module):
    """
    特征融合解码器

    该解码器接受编码器的融合特征输出重建原始图像：
    - 主要处理融合后的特征，而不是三路独立的embedding
    - 采用渐进式上采样策略，从24×24重建到224×224
    - 保持对称的解码架构，确保特征信息的最优重建

    核心特性：
    1. 融合特征输入：主要处理编码器的融合特征输出
    2. 渐进式上采样：通过多层转置卷积逐步恢复图像尺寸
    3. 无填充策略：所有转置卷积不使用padding，保持边缘真实性
    4. 最终sigmoid：只在最终输出层应用sigmoid激活函数
    5. 多尺度重建：也支持从三路独立特征重建的备选路径

    重建策略：
    1. 融合特征解码：主要路径，处理22×22×6的融合特征
    2. 渐进式上采样：22×22 → 56×56 → 112×112 → 224×224
    3. 通道扩展：从6通道逐步扩展到3通道RGB输出
    4. 最终激活：应用sigmoid确保输出范围[0,1]
    """

    def __init__(self):
        super().__init__()

          # 主要解码路径 - 处理融合特征
        self.main_decoder = nn.Sequential(
            # 22×22×6 → 56×56×16
            nn.ConvTranspose2d(6, 16, 5, stride=2, padding=0),  # 无padding, (22-5)*2+1 = 35×35，调整stride确保输出合适
            LayerNormalization(),
            nn.LeakyReLU(),

            # 56×56×16 → 112×112×12
            nn.ConvTranspose2d(16, 12, 5, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),

            # 112×112×12 → 224×224×6
            nn.ConvTranspose2d(12, 6, 5, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),

            # 224×224×6 → 224×224×3
            nn.Conv2d(6, 3, 3, stride=1, padding=0)  # 无padding，无sigmoid
        )

        # 备选的三路解码分支（用于对比和特征分析）
        self.small_decoder = nn.Sequential(
            # 28×28×8 → 56×56×12
            nn.ConvTranspose2d(8, 12, 5, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),

            # 56×56×12 → 112×112×8
            nn.ConvTranspose2d(12, 8, 5, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),

            # 112×112×8 → 224×224×3
            nn.ConvTranspose2d(8, 3, 5, stride=2, padding=0)  # 无padding，无sigmoid
        )

        self.medium_decoder = nn.Sequential(
            # 28×28×8 → 56×56×12
            nn.ConvTranspose2d(8, 12, 5, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),

            # 56×56×12 → 112×112×8
            nn.ConvTranspose2d(12, 8, 5, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),

            # 112×112×8 → 224×224×3
            nn.ConvTranspose2d(8, 3, 5, stride=2, padding=0)  # 无padding，无sigmoid
        )

        self.large_decoder = nn.Sequential(
            # 28×28×8 → 56×56×12
            nn.ConvTranspose2d(8, 12, 5, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),

            # 56×56×12 → 112×112×8
            nn.ConvTranspose2d(12, 8, 5, stride=2, padding=0),  # 无padding
            LayerNormalization(),
            nn.LeakyReLU(),

            # 112×112×8 → 224×224×3
            nn.ConvTranspose2d(8, 3, 5, stride=2, padding=0)  # 无padding，无sigmoid
        )

          # 最终融合模块
        self.final_fusion = nn.Sequential(
            nn.Conv2d(12, 6, 1),  # 融合主路径和三路重建结果 (4个重建结果 x 3通道 = 12通道)
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Conv2d(6, 3, 1),  # 最终输出，无sigmoid
        )

    def forward(self, embeddings):
        """
        Args:
            embeddings: 编码器的输出字典，包含:
                       - 'fused': 22×22×6 (主要融合特征)
                       - 'small': 28×28×8
                       - 'medium': 28×28×8
                       - 'large': 28×28×8
        """
        # 主要解码路径 - 处理融合特征
        fused_emb = embeddings['fused']  # B x 6 x 22 x 22

        # 首先将融合特征上采样到28×28以便与其他分支对齐
        fused_emb_up = F.interpolate(fused_emb, size=(28, 28), mode='bilinear', align_corners=False)

        # 主路径解码
        main_recon = self.main_decoder(fused_emb_up)  # 224×224×3

        # 备选的三路解码（用于对比和特征分析）
        small_emb = embeddings['small']    # B x 8 x 28 x 28
        medium_emb = embeddings['medium']  # B x 8 x 28 x 28
        large_emb = embeddings['large']    # B x 8 x 28 x 28

        small_recon = self.small_decoder(small_emb)
        medium_recon = self.medium_decoder(medium_emb)
        large_recon = self.large_decoder(large_emb)

        # 统一尺寸到224×224
        small_recon_up = F.interpolate(small_recon, size=(224, 224), mode='bilinear', align_corners=False)
        medium_recon_up = F.interpolate(medium_recon, size=(224, 224), mode='bilinear', align_corners=False)
        large_recon_up = F.interpolate(large_recon, size=(224, 224), mode='bilinear', align_corners=False)
        main_recon_up = F.interpolate(main_recon, size=(224, 224), mode='bilinear', align_corners=False)

        # 通道拼接 (224×224×12)
        combined = torch.cat([main_recon_up, small_recon_up, medium_recon_up, large_recon_up], dim=1)
        final_output = self.final_fusion(combined)

        # 只在最终输出应用sigmoid
        final_output = torch.sigmoid(final_output)

        return final_output