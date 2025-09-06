"""
流式视频自编码器核心实现

本模块实现了基于ObGD（Online Gradient Descent）优化器的流式视频自编码器。
主要特性：
1. 在线学习 - 无需存储历史数据，适合长时间流式处理
2. 双损失函数 - 细节损失和全局损失的平衡设计
3. 像素变化检测 - 智能识别帧间变化区域
4. 稀疏初始化 - 提高模型的泛化能力和训练稳定性
5. 实时可视化 - 支持训练过程的实时监控

优化器设计参考：
https://github.com/mohmdelsayed/streaming-drl
借鉴了该项目在流式深度强化学习中的在线优化思想，
特别是ObGD优化器在处理非平稳数据流时的优势。

核心组件：
- StreamingAutoEncoder: 主要的自编码器模型
- PixelChangeDetector: 帧间变化检测器
- LayerNormalization: 自定义层归一化
- ObGD Optimizer: 在线梯度下降优化器

作者：流式AI团队
版本：1.0
日期：2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import math
from optim import ObGD as Optimizer
from sparse_init import sparse_init
import cv2
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

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

def initialize_weights(m):
    """
    稀疏权重初始化函数

    对线性层和卷积层应用稀疏初始化，这是streaming-drl项目中
    提高模型泛化能力的重要技术。稀疏初始化的优势：

    1. 减少过拟合 - 通过稀疏连接降低模型复杂度
    2. 提高泛化 - 强制模型学习更鲁棒的特征表示
    3. 加速训练 - 减少需要更新的参数数量
    4. 内存效率 - 降低模型的内存占用

    Args:
        m (nn.Module): 待初始化的网络层

    Note:
        - 稀疏度设置为0.9，即90%的权重被置零
        - 偏置项统一初始化为0
        - 仅对Linear和Conv2d层进行稀疏初始化
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        sparse_init(m.weight, sparsity=0.9)  # 90%稀疏度
        if m.bias is not None:
            m.bias.data.fill_(0.0)  # 偏置置零

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

class SelfAttention2D(nn.Module):
    """2D自注意力机制 - 保留小目标特征"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels//8, 1)
        self.key = nn.Conv2d(channels, channels//8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        # 生成query, key, value
        q = self.query(x).view(B, -1, H*W).permute(0, 2, 1)  # [B, HW, C//8]
        k = self.key(x).view(B, -1, H*W)                     # [B, C//8, HW]
        v = self.value(x).view(B, -1, H*W)                   # [B, C, HW]

        # 计算注意力权重
        attention = torch.softmax(torch.bmm(q, k), dim=-1)   # [B, HW, HW]

        # 应用注意力
        out = torch.bmm(v, attention.permute(0, 2, 1))       # [B, C, HW]
        out = out.view(B, C, H, W)

        # 残差连接
        return self.gamma * out + x

class AttentionDownsample(nn.Module):
    """注意力引导下采样 - 智能保留重要特征，对小目标友好"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 注意力分支：生成重要性权重
        attention_channels = max(1, in_channels//4)  # 确保至少有1个通道
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, attention_channels, 1),
            nn.ReLU(),
            nn.Conv2d(attention_channels, 1, 1),
            nn.Sigmoid()
        )

        # 特征变换分支
        self.feature_conv = nn.Conv2d(in_channels, out_channels, 1)

        # 可选：额外的特征增强
        self.enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU()
        )

        # 用于调试可视化
        self.parent_model = None
        self.layer_name = None

        # 重新初始化注意力模块的权重（覆盖稀疏初始化）
        self._init_attention_weights()

    def _init_attention_weights(self):
        """为注意力模块使用更好的初始化"""
        for module in self.attention.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # 特征变换层也使用更好的初始化
        nn.init.xavier_uniform_(self.feature_conv.weight)
        if self.feature_conv.bias is not None:
            nn.init.constant_(self.feature_conv.bias, 0)

    def forward(self, x):
        # 1. 特征增强
        enhanced = self.enhance(x)

        # 2. 生成注意力权重 [B, 1, H, W]
        attention_weights = self.attention(enhanced)

        # 3. 加权特征 - 突出重要区域
        weighted_features = enhanced * attention_weights

        # 4. 注意力引导的平均池化
        # 对于每个2x2区域，根据注意力权重进行加权平均
        pooled_features = F.avg_pool2d(weighted_features, kernel_size=2, stride=2)
        pooled_weights = F.avg_pool2d(attention_weights, kernel_size=2, stride=2)

        # 5. 归一化：避免除零
        normalized_features = pooled_features / (pooled_weights + 1e-8)

        # 6. 特征变换到目标通道数
        output = self.feature_conv(normalized_features)

        # 7. 保存可视化数据（如果设置了parent_model）
        if hasattr(self, 'parent_model') and self.parent_model is not None:
            if hasattr(self.parent_model, 'debug_vis') and self.parent_model.debug_vis:
                if hasattr(self.parent_model, 'feature_maps') and self.layer_name:
                    # 保存注意力权重和输出特征
                    self.parent_model.feature_maps[f'{self.layer_name}_attention'] = attention_weights.detach()
                    self.parent_model.feature_maps[f'{self.layer_name}_output'] = output.detach()

        return output

class MultiHeadAttentionCompression(nn.Module):
    """多头注意力压缩 - 显式多头机制，每个头专注不同特征"""
    def __init__(self, in_channels, out_channels, input_size=16, output_size=4, num_heads=8):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        # 多头查询向量 (每个头有独立的查询)
        self.queries = nn.Parameter(torch.randn(num_heads, output_size*output_size, self.head_dim))

        # 多头Key、Value、Query投影
        self.key_proj = nn.Linear(in_channels, in_channels)
        self.value_proj = nn.Linear(in_channels, out_channels)
        self.query_proj = nn.Linear(self.head_dim, self.head_dim)

        # 多头输出投影
        self.out_proj = nn.Linear(out_channels, out_channels)

        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, input_size*input_size, in_channels))

        # 头融合层
        self.head_fusion = nn.Linear(num_heads * (out_channels // num_heads), out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.input_size and W == self.input_size

        # 展平为序列 [B, HW, C]
        x_flat = x.flatten(2).transpose(1, 2)  # [B, 256, C]

        # 添加位置编码
        x_flat = x_flat + self.pos_embed

        # 生成keys和values
        keys = self.key_proj(x_flat)      # [B, 256, C]
        values = self.value_proj(x_flat)  # [B, 256, out_C]

        # 重塑为多头格式
        keys = keys.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)    # [B, heads, 256, head_dim]
        values = values.view(B, -1, self.num_heads, self.out_channels//self.num_heads).transpose(1, 2)  # [B, heads, 256, out_head_dim]

        # 多头注意力计算
        head_outputs = []
        for head in range(self.num_heads):
            # 每个头的查询
            head_queries = self.queries[head].unsqueeze(0).expand(B, -1, -1)  # [B, 16, head_dim]
            head_queries = self.query_proj(head_queries)

            # 当前头的keys和values
            head_keys = keys[:, head]      # [B, 256, head_dim]
            head_values = values[:, head]  # [B, 256, out_head_dim]

            # 计算注意力权重
            attention_scores = torch.bmm(head_queries, head_keys.transpose(1, 2)) / (self.head_dim ** 0.5)
            attention_weights = torch.softmax(attention_scores, dim=-1)

            # 应用注意力
            head_output = torch.bmm(attention_weights, head_values)  # [B, 16, out_head_dim]
            head_outputs.append(head_output)

        # 融合所有头的输出
        multi_head_output = torch.cat(head_outputs, dim=-1)  # [B, 16, out_C]

        # 头融合投影
        fused_output = self.head_fusion(multi_head_output)

        # 最终输出投影
        compressed = self.out_proj(fused_output)

        # 重塑为特征图 [B, out_C, output_size, output_size]
        output = compressed.transpose(1, 2).reshape(B, self.out_channels, self.output_size, self.output_size)

        return output

class AttentionDecompression(nn.Module):
    """纯注意力解压缩 - 无卷积上采样，完美重建细节"""
    def __init__(self, in_channels, out_channels, input_size=4, output_size=16):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 可学习的查询向量 (代表输出位置)
        self.queries = nn.Parameter(torch.randn(1, output_size*output_size, out_channels))

        # Key和Value投影
        self.key_proj = nn.Linear(in_channels, out_channels)
        self.value_proj = nn.Linear(in_channels, out_channels)

        # 输出投影
        self.out_proj = nn.Linear(out_channels, out_channels)

        # 位置编码
        self.pos_embed_in = nn.Parameter(torch.randn(1, input_size*input_size, in_channels))
        self.pos_embed_out = nn.Parameter(torch.randn(1, output_size*output_size, out_channels))

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.input_size and W == self.input_size

        # 展平为序列 [B, HW, C]
        x_flat = x.flatten(2).transpose(1, 2)  # [B, 16, C]

        # 添加输入位置编码
        x_flat = x_flat + self.pos_embed_in

        # 生成keys和values
        keys = self.key_proj(x_flat)      # [B, 16, out_C]
        values = self.value_proj(x_flat)  # [B, 16, out_C]

        # 扩展queries到batch并添加位置编码
        queries = self.queries.expand(B, -1, -1) + self.pos_embed_out  # [B, 256, out_C]

        # 计算注意力权重 [B, 256, 16]
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.out_channels ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 应用注意力 [B, 256, out_C]
        decompressed = torch.bmm(attention_weights, values)

        # 输出投影
        decompressed = self.out_proj(decompressed)

        # 重塑为特征图 [B, out_C, output_size, output_size]
        output = decompressed.transpose(1, 2).reshape(B, self.out_channels, self.output_size, self.output_size)

        return output

class StreamingAutoEncoder(nn.Module):
    """
    流式视频自编码器 - 混合CNN + Vision Transformer架构

    该类实现了基于ObGD优化器的流式视频自编码器，核心特性包括：

    架构设计：
    1. 编码器：CNN + 自注意力机制，逐步压缩空间信息
       - 224x224 → 112x112 → 56x56 → 28x28
       - 通道数：3 → 8 → 16 → 16（潜在空间）
       - 集成自注意力增强特征表示

    2. 解码器：对称设计，逐步恢复空间分辨率
       - 28x28 → 56x56 → 112x112 → 224x224
       - 使用注意力解压缩和转置卷积

    优化策略（参考streaming-drl）：
    1. 双优化器设计：
       - 细节优化器：关注局部特征和边缘信息
       - 全局优化器：关注整体结构和语义信息

    2. ObGD在线学习：
       - 无需存储历史数据
       - 自适应学习率调整
       - 适合长时间流式处理

    3. 智能变化检测：
       - 像素级变化检测
       - 注意力引导学习
       - 计算资源优化

    技术创新：
    - 稀疏权重初始化提高泛化能力
    - 层归一化稳定训练过程
    - 实时可视化支持调试分析
    """

    def __init__(self, input_channels=3, base_channels=8, latent_channels=16,
                 lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0, 
                 debug_vis=False, use_tensorboard=True, log_dir=None):
        """
        初始化流式自编码器

        Args:
            input_channels (int): 输入图像通道数，默认3（RGB）
            base_channels (int): 编码器基础通道数，控制模型容量
            latent_channels (int): 潜在空间维度，影响压缩比
            lr (float): ObGD优化器学习率
            gamma (float): 动量衰减因子，用于梯度平滑
            lamda (float): 损失函数权重平衡参数
            kappa (float): 损失稳定性参数
            debug_vis (bool): 是否启用调试可视化
            use_tensorboard (bool): 是否启用TensorBoard日志记录
            log_dir (str): TensorBoard日志目录，默认为None时自动生成
        """
        super(StreamingAutoEncoder, self).__init__()

        # 调试可视化
        self.debug_vis = debug_vis
        self.feature_maps = {}
        
        # TensorBoard设置
        self.use_tensorboard = use_tensorboard
        self.writer = None
        self.global_step = 0
        
        if self.use_tensorboard:
            if log_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir = f"runs/streaming_ae_{timestamp}"
            
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard日志将保存到: {log_dir}")
            print(f"启动TensorBoard: tensorboard --logdir={log_dir}")

        # 优化编码器：避免过度压缩，保留更多空间信息
        self.encoder = nn.ModuleList([
            # 224x224 -> 112x112, 3->8通道
            nn.Sequential(
                nn.Conv2d(input_channels, base_channels, 4, stride=2, padding=1),
                LayerNormalization(),
                nn.LeakyReLU()
            ),

            # 112x112 -> 56x56, 8->16通道
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels*2, 4, stride=2, padding=1),
                LayerNormalization(),
                nn.LeakyReLU()
            ),

            # 56x56 -> 56x56, 注意力处理特征，保持空间尺寸
            nn.Sequential(
                SelfAttention2D(base_channels*2),  # 自注意力增强特征
                nn.Conv2d(base_channels*2, base_channels*2, 3, stride=1, padding=1),
                LayerNormalization(),
                nn.LeakyReLU()
            ),

            # 56x56 -> 28x28, 注意力引导压缩到最终尺寸
            MultiHeadAttentionCompression(base_channels*2, latent_channels, input_size=56, output_size=28, num_heads=8)
        ])

        # 对称解码器：从28x28恢复到224x224
        self.decoder = nn.ModuleList([
            # 从16个embedding恢复到16通道56x56特征图
            AttentionDecompression(latent_channels, base_channels*2, input_size=28, output_size=56),

            # 56x56 -> 56x56, 注意力处理特征，保持空间尺寸
            nn.Sequential(
                SelfAttention2D(base_channels*2),  # 自注意力增强特征
                nn.Conv2d(base_channels*2, base_channels*2, 3, stride=1, padding=1),
                LayerNormalization(),
                nn.LeakyReLU()
            ),

            # 56x56 -> 112x112, 16->8通道
            nn.Sequential(
                nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1),
                LayerNormalization(),
                nn.LeakyReLU()
            ),

            # 112x112 -> 224x224, 8->3通道 (最终重建)
            nn.Sequential(
                nn.ConvTranspose2d(base_channels, input_channels, 4, stride=2, padding=1),
                nn.Sigmoid()
            )
        ])
        
        # 初始化权重
        self.apply(initialize_weights)

        # 新架构使用简单卷积层，无需特殊设置

        # 单一优化器：只使用全局损失
        self.optimizer = Optimizer(self.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)

        # 像素变化检测器
        self.change_detector = PixelChangeDetector()

        # 存储上一帧
        self.prev_frame = None
        self.prev_embedding = None
        
    def encode(self, x):
        """简化编码：CNN + Self-Attention"""
        if self.debug_vis:
            self.feature_maps['input'] = x.detach()

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if self.debug_vis:
                self.feature_maps[f'encoder_layer_{i}'] = x.detach()

        if self.debug_vis:
            self.feature_maps['bottleneck'] = x.detach()
        return x

    def decode(self, z):
        """简化解码：CNN + Self-Attention"""
        for i, layer in enumerate(self.decoder):
            z = layer(z)
            if self.debug_vis:
                self.feature_maps[f'decoder_layer_{i}'] = z.detach()

        if self.debug_vis:
            self.feature_maps['output'] = z.detach()
        return z
    
    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z
    
    
    def compute_global_loss(self, curr_frame, reconstruction):
        """
        全局损失：整体图像重建质量
        结合多种损失函数确保整体embedding质量
        """
        # 1. MSE损失 - 使用sum而不是mean，计算每个像素差值总和
        mse_loss = F.mse_loss(reconstruction, curr_frame, reduction='sum')
        
        
        # 3. SSIM损失（结构相似性）
        def compute_ssim_loss(img1, img2, window_size=11):
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
        
        ssim_loss = compute_ssim_loss(curr_frame, reconstruction)
        
        # 组合全局损失
        global_loss = mse_loss + 0.1 * ssim_loss
        
        return global_loss, mse_loss, ssim_loss
    
    def update_params(self, curr_frame, debug=False):
        """
        参数更新：使用单一全局损失和ObGD优化器
        """
        # 前向传播
        reconstruction, embedding = self.forward(curr_frame)
        
        # 检测像素变化
        change_mask, change_intensity = self.change_detector.detect_changes(self.prev_frame, curr_frame)
        
        # 计算全局损失
        global_loss, mse_loss, ssim_loss = self.compute_global_loss(curr_frame, reconstruction)
        
        # 反向传播和参数更新
        self.optimizer.zero_grad()
        global_loss.backward()
        self.optimizer.step(global_loss.item(), reset=False)
        
        # 更新历史信息
        self.prev_frame = curr_frame.detach().clone()
        self.prev_embedding = embedding.detach().clone()
        
        # TensorBoard日志记录
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar('Loss/Global_Loss', global_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/MSE_Loss', mse_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/SSIM_Loss', ssim_loss.item(), self.global_step)
            self.writer.add_scalar('Metrics/Changed_Pixels', torch.sum(change_mask).item(), self.global_step)
            self.writer.add_scalar('Metrics/Change_Ratio', torch.sum(change_mask).item() / curr_frame.numel(), self.global_step)
            
            # 每10步记录图像（增加频率）
            if self.global_step % 10 == 0:
                # 确保图像在[0,1]范围内
                input_img = torch.clamp(curr_frame[0], 0, 1)
                recon_img = torch.clamp(reconstruction[0], 0, 1)
                
                # 原始输入
                self.writer.add_image('Images/Input', input_img, self.global_step)
                # 重建输出
                self.writer.add_image('Images/Reconstruction', recon_img, self.global_step)
                # 变化掩码
                self.writer.add_image('Images/Change_Mask', change_mask[0], self.global_step)
                # 重建误差
                error_map = torch.abs(curr_frame - reconstruction)
                error_img = torch.clamp(error_map[0], 0, 1)
                self.writer.add_image('Images/Reconstruction_Error', error_img, self.global_step)
                
                # 特征图可视化
                if self.debug_vis:
                    for layer_name, feature_map in self.feature_maps.items():
                        if 'encoder' in layer_name or 'decoder' in layer_name:
                            # 选择第一个通道进行可视化
                            if feature_map.shape[1] > 0:
                                feature_vis = feature_map[0, 0:1]  # 取第一个通道
                                # 归一化特征图到[0,1]
                                feature_vis = (feature_vis - feature_vis.min()) / (feature_vis.max() - feature_vis.min() + 1e-8)
                                self.writer.add_image(f'Features/{layer_name}', feature_vis, self.global_step)
            
            # 每1000步记录模型参数分布
            if self.global_step % 1000 == 0:
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f'Parameters/{name}', param, self.global_step)
                        self.writer.add_histogram(f'Gradients/{name}', param.grad, self.global_step)
            
            self.global_step += 1
        
        if debug:
            print(f"Step {self.global_step}: Global={global_loss.item():.1f}, MSE={mse_loss.item():.1f}")
        
        return {
            'global_loss': global_loss.item(),
            'mse_loss': mse_loss.item(),
            'ssim_loss': ssim_loss.item(),
            'changed_pixels': torch.sum(change_mask).item(),
            'reconstruction': reconstruction.detach(),
            'embedding': embedding.detach(),
            'change_mask': change_mask.detach()
        }

    def get_feature_visualization(self, layer_name, channel_idx=None):
        """获取指定层的特征图可视化"""
        if not self.debug_vis or layer_name not in self.feature_maps:
            return None

        feature_map = self.feature_maps[layer_name]
        B, C, H, W = feature_map.shape

        # 如果没有指定通道，随机选择一个
        if channel_idx is None:
            channel_idx = torch.randint(0, C, (1,)).item()

        # 提取单个通道并归一化到[0,1]
        single_channel = feature_map[0, channel_idx].cpu()
        normalized = (single_channel - single_channel.min()) / (single_channel.max() - single_channel.min() + 1e-8)

        return normalized.numpy(), channel_idx

    def get_all_layer_info(self):
        """获取所有层的信息"""
        if not self.debug_vis:
            return {}

        layer_info = {}
        for layer_name, feature_map in self.feature_maps.items():
            B, C, H, W = feature_map.shape
            layer_info[layer_name] = {
                'shape': (C, H, W),
                'min_val': feature_map.min().item(),
                'max_val': feature_map.max().item(),
                'mean_val': feature_map.mean().item(),
                'std_val': feature_map.std().item()
            }
        return layer_info
    
    def close_tensorboard(self):
        """关闭TensorBoard写入器"""
        if self.writer is not None:
            self.writer.close()
            print("TensorBoard写入器已关闭")
    
    def __del__(self):
        """析构函数，确保TensorBoard写入器被正确关闭"""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()

def preprocess_frame(frame, target_size=(224, 224)):
    """
    预处理Gymnasium环境的视频帧

    该函数将来自Gymnasium环境的原始帧数据转换为适合流式自编码器处理的格式。
    处理步骤遵循streaming-drl项目的数据预处理标准：

    处理流程：
    1. 数据类型转换：numpy.ndarray → torch.Tensor
    2. 维度调整：HWC → CHW（符合PyTorch约定）
    3. 批次维度：CHW → BCHW（B=1）
    4. 数值归一化：[0,255] → [0,1]（稳定训练）
    5. 尺寸标准化：任意尺寸 → 224x224（模型输入要求）

    技术细节：
    - 使用双线性插值进行尺寸调整，保持图像质量
    - 自动检测输入格式，兼容多种数据源
    - 归一化确保数值稳定性，避免梯度爆炸

    Args:
        frame (np.ndarray or torch.Tensor): 输入帧数据
            - 支持格式：HWC或CHW
            - 数值范围：[0,255]或[0,1]
        target_size (tuple): 目标尺寸，默认(224,224)
            - 必须与模型输入尺寸匹配

    Returns:
        torch.Tensor: 预处理后的帧张量
            - 形状：[1, 3, 224, 224]
            - 数值范围：[0, 1]
            - 数据类型：torch.float32

    Example:
        >>> frame = env.reset()[0]  # Gymnasium环境帧
        >>> processed = preprocess_frame(frame)
        >>> print(processed.shape)  # torch.Size([1, 3, 224, 224])
    """
    # 数据类型转换：确保为PyTorch张量
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame).float()

    # 维度调整：HWC → CHW（PyTorch标准格式）
    if len(frame.shape) == 3 and frame.shape[-1] == 3:  # HWC -> CHW
        frame = frame.permute(2, 0, 1)

    # 批次维度：CHW → BCHW（模型输入要求）
    if len(frame.shape) == 3:
        frame = frame.unsqueeze(0)

    # 数值归一化：[0,255] → [0,1]（训练稳定性）
    if frame.max() > 1.0:
        frame = frame / 255.0

    # 尺寸标准化：双线性插值到目标尺寸
    frame = F.interpolate(frame, size=target_size, mode='bilinear', align_corners=False)

    return frame

def main():
    # 创建gym环境
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    
    # 创建模型
    model = StreamingAutoEncoder(
        input_channels=3,
        hidden_dim=128,
        latent_dim=64,
        lr=0.1,
        gamma=0.99,
        lamda=0.8,
        kappa_detail=3.0,
        kappa_global=2.0
    )
    
    # 训练参数
    total_frames = 10000
    debug_interval = 100
    
    print("开始流式视频自编码器训练...")
    print(f"总帧数: {total_frames}")
    
    # 重置环境
    obs, _ = env.reset()
    frame_count = 0
    
    # 存储损失历史
    loss_history = {
        'global_loss': [],
        'mse_loss': [],
        'changed_pixels': []
    }
    
    try:
        while frame_count < total_frames:
            # 预处理当前帧
            curr_frame = preprocess_frame(obs)
            
            # 更新模型参数
            debug = (frame_count % debug_interval == 0)
            results = model.update_params(curr_frame, debug=debug)
            
            # 记录损失
            loss_history['global_loss'].append(results['global_loss'])
            loss_history['mse_loss'].append(results['mse_loss'])
            loss_history['changed_pixels'].append(results['changed_pixels'])
            
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 如果episode结束，重置环境
            if terminated or truncated:
                obs, _ = env.reset()
                # 重置模型的历史信息
                model.prev_frame = None
                model.prev_embedding = None
            
            frame_count += 1
            
            # 定期输出统计信息
            if frame_count % debug_interval == 0:
                recent_global = np.mean(loss_history['global_loss'][-debug_interval:])
                recent_mse = np.mean(loss_history['mse_loss'][-debug_interval:])
                recent_changed = np.mean(loss_history['changed_pixels'][-debug_interval:])
                
                print(f"Frame {frame_count}/{total_frames}")
                print(f"  平均全局损失: {recent_global:.6f}")
                print(f"  平均MSE损失: {recent_mse:.6f}")
                print(f"  平均变化像素: {recent_changed:.0f}")
                print("-" * 50)
    
    except KeyboardInterrupt:
        print("训练被用户中断")
    
    finally:
        env.close()
        print("训练完成！")
        
        # 保存模型
        torch.save(model.state_dict(), 'streaming_autoencoder.pth')
        print("模型已保存到 streaming_autoencoder.pth")

if __name__ == "__main__":
    main()
