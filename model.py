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

        return fused_features


class Decoder(nn.Module):
    """
    对称特征分解解码器

    该解码器与编码器完全对称，反向实现编码器的每一个步骤：
    - 特征分解：22×22×6 → 28×28×24 (反向融合过程)
    - 三路并行解码：28×28×8 (三路) → 224×224×3 (反向编码过程)

    对称架构：
    编码器: 224×224×3 → [三路分支] → 28×28×24 → 22×22×6
    解码器: 22×22×6 → 28×28×24 → [三路分支] → 224×224×3

    核心特性：
    1. 完全对称：与编码器结构镜像对称
    2. 特征分解：反向融合过程，恢复三路特征
    3. 并行解码：三路独立解码分支
    4. 无填充设计：保持边缘真实性
    5. 真正压缩：只使用22×22×6融合特征

    解码策略：
    1. 特征分解: 22×22×6 → 24×24×8 → 26×26×16 → 28×28×24
    2. 通道分离: 28×28×24 → 28×28×8 (三路)
    3. 并行上采样: 三路分别从28×28×8 → 224×224×3
    4. 结果融合: 融合三路重建结果
    """

    def __init__(self):
        super().__init__()

        # 特征分解模块 - 反向融合过程
        self.feature_decomposition = nn.Sequential(
            # 22×22×6 → 24×24×8 (反向编码器的最后一步)
            nn.ConvTranspose2d(6, 8, 3, stride=1, padding=0),  # (22-3)+1 = 20×20, 需要调整
            LayerNormalization(),
            nn.LeakyReLU(),

            # 使用插值确保精确尺寸
            nn.Upsample(size=(24, 24), mode='bilinear', align_corners=False),
            nn.Conv2d(8, 8, 3, stride=1, padding=0),
            LayerNormalization(),
            nn.LeakyReLU(),

            # 24×24×8 → 26×26×16 (反向编码器的中间步骤)
            nn.ConvTranspose2d(8, 16, 3, stride=1, padding=0),  # (24-3)+1 = 22×22
            LayerNormalization(),
            nn.LeakyReLU(),

            nn.Upsample(size=(26, 26), mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, 3, stride=1, padding=0),
            LayerNormalization(),
            nn.LeakyReLU(),

            # 26×26×16 → 28×28×24 (反向编码器的第一步)
            nn.ConvTranspose2d(16, 24, 3, stride=1, padding=0),  # (26-3)+1 = 24×24
            LayerNormalization(),
            nn.LeakyReLU(),

            nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False),
            nn.Conv2d(24, 24, 3, stride=1, padding=0),
            LayerNormalization(),
            nn.LeakyReLU()
        )  # 输出: 28×28×24

        # 通道分离 - 将24通道分离为三路8通道
        self.channel_split = nn.Conv2d(24, 24, 1)  # 1×1卷积用于通道重排

        # 三路并行解码分支 (与编码器镜像对称)

        # 小卷积分支解码 - 反向编码器的small_kernel_branch
        self.small_decoder = nn.Sequential(
            # 28×28×8 → 106×106×16 (反向编码器的第二步)
            nn.ConvTranspose2d(8, 16, 7, stride=4, padding=0, output_padding=1),  # (28-7)*4+1+1 = 89×89, 需要调整
            LayerNormalization(),
            nn.LeakyReLU(),

            nn.Upsample(size=(106, 106), mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, 3, stride=1, padding=0),
            LayerNormalization(),
            nn.LeakyReLU(),

            # 106×106×16 → 224×224×3 (反向编码器的第一步)
            nn.ConvTranspose2d(16, 3, 5, stride=2, padding=0, output_padding=1),  # (106-5)*2+1+1 = 205×205
            LayerNormalization(),
            nn.LeakyReLU(),

            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            nn.Conv2d(3, 3, 3, stride=1, padding=0),
        )

        # 中卷积分支解码 - 反向编码器的medium_kernel_branch
        self.medium_decoder = nn.Sequential(
            # 28×28×8 → 102×102×16 (反向编码器的第二步)
            nn.ConvTranspose2d(8, 16, 7, stride=4, padding=0, output_padding=1),  # (28-7)*4+1+1 = 89×89
            LayerNormalization(),
            nn.LeakyReLU(),

            nn.Upsample(size=(102, 102), mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, 3, stride=1, padding=0),
            LayerNormalization(),
            nn.LeakyReLU(),

            # 102×102×16 → 224×224×3 (反向编码器的第一步)
            nn.ConvTranspose2d(16, 3, 13, stride=2, padding=0, output_padding=1),  # (102-13)*2+1+1 = 181×181
            LayerNormalization(),
            nn.LeakyReLU(),

            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            nn.Conv2d(3, 3, 3, stride=1, padding=0),
        )

        # 大卷积分支解码 - 反向编码器的large_kernel_branch
        self.large_decoder = nn.Sequential(
            # 28×28×8 → 98×98×16 (反向编码器的第二步)
            nn.ConvTranspose2d(8, 16, 7, stride=4, padding=0, output_padding=1),  # (28-7)*4+1+1 = 89×89
            LayerNormalization(),
            nn.LeakyReLU(),

            nn.Upsample(size=(98, 98), mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, 3, stride=1, padding=0),
            LayerNormalization(),
            nn.LeakyReLU(),

            # 98×98×16 → 224×224×3 (反向编码器的第一步)
            nn.ConvTranspose2d(16, 3, 21, stride=2, padding=0, output_padding=1),  # (98-21)*2+1+1 = 157×157
            LayerNormalization(),
            nn.LeakyReLU(),

            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            nn.Conv2d(3, 3, 3, stride=1, padding=0),
        )

        # 最终融合模块 - 融合三路重建结果并修正尺寸
        self.final_fusion = nn.Sequential(
            nn.Conv2d(9, 6, 1),  # 融合三路结果 (3×3=9通道)
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Conv2d(6, 6, 1),  # 中间层
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),  # 确保精确尺寸
            nn.Conv2d(6, 3, 3, stride=1, padding=1),  # 最终输出，保持224×224
        )

    def forward(self, fused_features):
        """
        Args:
            fused_features: 编码器的输出张量，形状为 B x 6 x 22 x 22
                           这是压缩的融合特征
        """
        # 特征分解：反向融合过程
        decomposed_features = self.feature_decomposition(fused_features)  # 28×28×24

        # 通道重排以便分离
        rearranged_features = self.channel_split(decomposed_features)  # 28×28×24

        # 将24通道分离为三路8通道
        small_features = rearranged_features[:, 0:8, :, :]    # 28×28×8
        medium_features = rearranged_features[:, 8:16, :, :]  # 28×28×8
        large_features = rearranged_features[:, 16:24, :, :]  # 28×28×8

        # 三路并行解码
        small_recon = self.small_decoder(small_features)    # 224×224×3
        medium_recon = self.medium_decoder(medium_features)  # 224×224×3
        large_recon = self.large_decoder(large_features)    # 224×224×3

        # 融合三路重建结果
        combined = torch.cat([small_recon, medium_recon, large_recon], dim=1)  # 224×224×9
        final_output = self.final_fusion(combined)  # 224×224×3

        # 只在最终输出应用sigmoid
        final_output = torch.sigmoid(final_output)

        return final_output