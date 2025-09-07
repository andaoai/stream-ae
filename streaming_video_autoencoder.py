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
import cv2
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import time

# 导入自定义模块
from loss import compute_global_loss
from optim import initialize_weights
from model import (
    PixelChangeDetector, Encoder, Decoder
)


class StreamingAutoEncoder(nn.Module):
    """
    流式视频自编码器 - 优化的并行多尺度架构

    该类实现了基于ObGD优化器的流式视频自编码器，采用全新的并行多尺度架构：

    架构设计：
    1. 并行编码器：三个不同尺度的卷积分支并行处理
       - 小卷积分支 (3×3): 224×224×3 → 27×27×3，纹理特征
       - 中卷积分支 (5×5): 224×224×3 → 25×25×2，平衡特征  
       - 大卷积分支 (7×7): 224×224×3 → 23×23×2，结构特征
       - 压缩比：92:1, 161:1, 190:1

    2. 并行解码器：三个对应的解码分支并行重建
       - 小卷积分支解码器: 27×27×3 → 223×223×3
       - 中卷积分支解码器: 25×25×2 → 221×221×3
       - 大卷积分支解码器: 23×23×2 → 219×219×3
       - 自适应融合: 统一尺寸并融合为224×224×3

    核心优势：
    1. 无填充设计：所有卷积层不使用padding，保持边缘信息真实性
    2. 灵活尺寸：每个分支的embedding尺寸独立计算，在12~28范围内
    3. 并行处理：三个分支完全并行，提高计算效率
    4. 多尺度特征：不同卷积核捕获不同类型的特征信息
    5. 高压缩率：整体压缩比约92:1至190:1，参数量约15万

    优化策略（参考streaming-drl）：
    1. ObGD在线学习：无需存储历史数据，适合长时间流式处理
    2. 双损失函数：细节损失和全局损失的平衡设计
    3. 智能变化检测：像素级变化检测和注意力引导学习
    4. 稀疏初始化：提高模型泛化能力和训练稳定性

    技术创新：
    - 零填充策略避免边缘失真
    - 尺寸自适应融合模块
    - 实时可视化支持调试分析
    - TensorBoard集成监控训练过程
    """

    def __init__(self, input_channels=3, base_channels=8, latent_channels=3,
                 lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0, 
                 debug_vis=False, use_tensorboard=True, log_dir=None):
        """
        初始化流式自编码器

        Args:
            input_channels (int): 输入图像通道数，默认3（RGB）
            base_channels (int): 编码器基础通道数（保留兼容性，实际使用固定设计）
            latent_channels (int): 潜在空间维度，每个分支的输出通道数
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
        
        # 性能监控
        self.start_time = time.time()
        self.last_step_time = self.start_time
        self.step_times = deque(maxlen=100)  # 保存最近100步的时间
        self.fps_history = deque(maxlen=100)  # 保存最近100步的FPS
        
        if self.use_tensorboard:
            if log_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir = f"runs/streaming_ae_{timestamp}"
            
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard日志将保存到: {log_dir}")
            print(f"启动TensorBoard: tensorboard --logdir={log_dir}")

        # 优化的并行多尺度编码器
        self.encoder = Encoder(input_channels, latent_channels)
        
        # 优化的并行多尺度解码器
        self.decoder = Decoder(latent_channels, input_channels)
        
        # 初始化权重
        self.apply(initialize_weights)

        # 使用标准Adam优化器替代ObGD，提高训练稳定性
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))

        # 像素变化检测器
        self.change_detector = PixelChangeDetector()

        # 存储上一帧
        self.prev_frame = None
        self.prev_embedding = None
        
    def encode(self, x):
        """并行多尺度编码"""
        if self.debug_vis:
            self.feature_maps['input'] = x.detach()

        # 并行编码得到三个不同尺度的embedding
        small_emb, medium_emb, large_emb = self.encoder(x)
        
        if self.debug_vis:
            self.feature_maps['small_embedding'] = small_emb.detach()
            self.feature_maps['medium_embedding'] = medium_emb.detach()
            self.feature_maps['large_embedding'] = large_emb.detach()
            self.feature_maps['bottleneck'] = (small_emb.detach(), medium_emb.detach(), large_emb.detach())
        
        return small_emb, medium_emb, large_emb

    def decode(self, small_emb, medium_emb, large_emb):
        """并行多尺度解码"""
        # 并行解码三个embedding
        reconstruction = self.decoder(small_emb, medium_emb, large_emb)
        
        if self.debug_vis:
            self.feature_maps['output'] = reconstruction.detach()
        
        return reconstruction
    
    def forward(self, x):
        small_emb, medium_emb, large_emb = self.encode(x)
        reconstruction = self.decode(small_emb, medium_emb, large_emb)
        # 返回重建图像和三个embedding的元组
        return reconstruction, (small_emb, medium_emb, large_emb)
    
    
    def compute_global_loss(self, curr_frame, reconstruction):
        """
        全局损失：整体图像重建质量
        结合多种损失函数确保整体embedding质量
        """
        # 使用loss模块中的函数
        return compute_global_loss(curr_frame, reconstruction)
    
    def update_params(self, curr_frame, debug=False):
        """
        参数更新：使用单一全局损失和ObGD优化器
        """
        # 计算FPS
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.step_times.append(step_time)
        
        # 计算当前FPS
        if step_time > 0:
            current_fps = 1.0 / step_time
        else:
            current_fps = 0.0
        self.fps_history.append(current_fps)
        
        self.last_step_time = current_time
        
        # 前向传播
        reconstruction, embeddings = self.forward(curr_frame)
        small_emb, medium_emb, large_emb = embeddings
        
        # 检测像素变化
        change_mask, change_intensity = self.change_detector.detect_changes(self.prev_frame, curr_frame)
        
        # 计算全局损失
        global_loss, mse_loss, l1_loss, ssim_loss = self.compute_global_loss(curr_frame, reconstruction)
        
        # 反向传播和参数更新
        self.optimizer.zero_grad()
        global_loss.backward()
        
        # 梯度裁剪 - 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # 使用标准Adam优化器的step方法
        self.optimizer.step()
        
        # 更新历史信息
        self.prev_frame = curr_frame.detach().clone()
        self.prev_embedding = (small_emb.detach().clone(), medium_emb.detach().clone(), large_emb.detach().clone())
        
        # TensorBoard日志记录 - 专注于卷积核输出可视化
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar('Loss/Global_Loss', global_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/MSE_Loss', mse_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/L1_Loss', l1_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/SSIM_Loss', ssim_loss.item(), self.global_step)
            self.writer.add_scalar('Metrics/Changed_Pixels', torch.sum(change_mask).item(), self.global_step)
            
            # 性能监控
            self.writer.add_scalar('Performance/Current_FPS', current_fps, self.global_step)
            if len(self.fps_history) > 0:
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                self.writer.add_scalar('Performance/Average_FPS', avg_fps, self.global_step)
            
            # 计算总运行时间
            total_time = current_time - self.start_time
            self.writer.add_scalar('Performance/Total_Time', total_time, self.global_step)
            
            # 每20步记录卷积核输出（降低频率，专注于重要信息）
            if self.global_step % 20 == 0:
                # 确保图像在[0,1]范围内
                input_img = torch.clamp(curr_frame[0], 0, 1)
                recon_img = torch.clamp(reconstruction[0], 0, 1)
                
                # 基础图像
                self.writer.add_image('00_Input', input_img, self.global_step)
                self.writer.add_image('01_Reconstruction', recon_img, self.global_step)
                
                # 重建误差
                error_map = torch.abs(curr_frame - reconstruction)
                error_img = torch.clamp(error_map[0], 0, 1)
                self.writer.add_image('02_Reconstruction_Error', error_img, self.global_step)
                
                # 重点：三个卷积分支的embedding输出
                if self.debug_vis:
                    # 小卷积分支 (3×3) - 纹理特征
                    if 'small_embedding' in self.feature_maps:
                        small_emb = self.feature_maps['small_embedding'][0]  # [4, 27, 27]
                        # 显示前4个通道
                        for i in range(min(4, small_emb.shape[0])):
                            channel_vis = (small_emb[i] - small_emb[i].min()) / (small_emb[i].max() - small_emb[i].min() + 1e-8)
                            # 添加通道维度，变成 [1, 27, 27] 满足CHW格式
                            channel_vis = channel_vis.unsqueeze(0)
                            self.writer.add_image(f'10_Small_Kernel_3x3/Channel_{i+1}', channel_vis, self.global_step)
                    
                    # 中卷积分支 (5×5) - 平衡特征
                    if 'medium_embedding' in self.feature_maps:
                        medium_emb = self.feature_maps['medium_embedding'][0]  # [4, 25, 25]
                        # 显示前4个通道
                        for i in range(min(4, medium_emb.shape[0])):
                            channel_vis = (medium_emb[i] - medium_emb[i].min()) / (medium_emb[i].max() - medium_emb[i].min() + 1e-8)
                            # 添加通道维度，变成 [1, 25, 25] 满足CHW格式
                            channel_vis = channel_vis.unsqueeze(0)
                            self.writer.add_image(f'11_Medium_Kernel_5x5/Channel_{i+1}', channel_vis, self.global_step)
                    
                    # 大卷积分支 (7×7) - 结构特征
                    if 'large_embedding' in self.feature_maps:
                        large_emb = self.feature_maps['large_embedding'][0]  # [4, 23, 23]
                        # 显示前4个通道
                        for i in range(min(4, large_emb.shape[0])):
                            channel_vis = (large_emb[i] - large_emb[i].min()) / (large_emb[i].max() - large_emb[i].min() + 1e-8)
                            # 添加通道维度，变成 [1, 23, 23] 满足CHW格式
                            channel_vis = channel_vis.unsqueeze(0)
                            self.writer.add_image(f'12_Large_Kernel_7x7/Channel_{i+1}', channel_vis, self.global_step)
                    
                    # 计算embedding的统计信息
                    if 'small_embedding' in self.feature_maps:
                        small_emb = self.feature_maps['small_embedding'][0]
                        self.writer.add_scalar('Embeddings/Small_Mean', small_emb.mean().item(), self.global_step)
                        self.writer.add_scalar('Embeddings/Small_Std', small_emb.std().item(), self.global_step)
                        self.writer.add_scalar('Embeddings/Small_Max', small_emb.max().item(), self.global_step)
                    
                    if 'medium_embedding' in self.feature_maps:
                        medium_emb = self.feature_maps['medium_embedding'][0]
                        self.writer.add_scalar('Embeddings/Medium_Mean', medium_emb.mean().item(), self.global_step)
                        self.writer.add_scalar('Embeddings/Medium_Std', medium_emb.std().item(), self.global_step)
                        self.writer.add_scalar('Embeddings/Medium_Max', medium_emb.max().item(), self.global_step)
                    
                    if 'large_embedding' in self.feature_maps:
                        large_emb = self.feature_maps['large_embedding'][0]
                        self.writer.add_scalar('Embeddings/Large_Mean', large_emb.mean().item(), self.global_step)
                        self.writer.add_scalar('Embeddings/Large_Std', large_emb.std().item(), self.global_step)
                        self.writer.add_scalar('Embeddings/Large_Max', large_emb.max().item(), self.global_step)
            
            self.global_step += 1
        
        if debug:
            if len(self.fps_history) > 0:
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                print(f"Step {self.global_step}: Global={global_loss.item():.1f}, MSE={mse_loss.item():.1f}, FPS={current_fps:.1f}, AvgFPS={avg_fps:.1f}")
            else:
                print(f"Step {self.global_step}: Global={global_loss.item():.1f}, MSE={mse_loss.item():.1f}, FPS={current_fps:.1f}")
            # 打印卷积核输出摘要
            self.print_kernel_outputs()
        
        return {
            'global_loss': global_loss.item(),
            'mse_loss': mse_loss.item(),
            'ssim_loss': ssim_loss.item(),
            'changed_pixels': torch.sum(change_mask).item(),
            'current_fps': current_fps,
            'average_fps': sum(self.fps_history) / len(self.fps_history) if len(self.fps_history) > 0 else 0.0,
            'reconstruction': reconstruction.detach(),
            'embedding': embeddings,  # 返回三个embedding的元组
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

    def get_kernel_outputs_summary(self):
        """
        获取三个卷积分支的输出摘要信息
        
        Returns:
            dict: 包含每个分支输出的统计信息
        """
        if not self.debug_vis:
            return {}
        
        summary = {}
        
        # 小卷积分支 (3×3) - 纹理特征
        if 'small_embedding' in self.feature_maps:
            small_emb = self.feature_maps['small_embedding'][0]  # [4, 27, 27]
            summary['small_kernel'] = {
                'shape': tuple(small_emb.shape),
                'mean': small_emb.mean().item(),
                'std': small_emb.std().item(),
                'min': small_emb.min().item(),
                'max': small_emb.max().item(),
                'compression_ratio': (224*224*3) / (27*27*4),
                'description': '3×3卷积核 - 纹理特征'
            }
        
        # 中卷积分支 (5×5) - 平衡特征
        if 'medium_embedding' in self.feature_maps:
            medium_emb = self.feature_maps['medium_embedding'][0]  # [4, 25, 25]
            summary['medium_kernel'] = {
                'shape': tuple(medium_emb.shape),
                'mean': medium_emb.mean().item(),
                'std': medium_emb.std().item(),
                'min': medium_emb.min().item(),
                'max': medium_emb.max().item(),
                'compression_ratio': (224*224*3) / (25*25*4),
                'description': '5×5卷积核 - 平衡特征'
            }
        
        # 大卷积分支 (7×7) - 结构特征
        if 'large_embedding' in self.feature_maps:
            large_emb = self.feature_maps['large_embedding'][0]  # [4, 23, 23]
            summary['large_kernel'] = {
                'shape': tuple(large_emb.shape),
                'mean': large_emb.mean().item(),
                'std': large_emb.std().item(),
                'min': large_emb.min().item(),
                'max': large_emb.max().item(),
                'compression_ratio': (224*224*3) / (23*23*4),
                'description': '7×7卷积核 - 结构特征'
            }
        
        return summary

    def print_kernel_outputs(self):
        """打印三个卷积分支的输出信息"""
        summary = self.get_kernel_outputs_summary()
        
        if not summary:
            print("没有可用的卷积核输出信息（请确保debug_vis=True）")
            return
        
        print("=" * 60)
        print("并行多尺度卷积核输出摘要")
        print("=" * 60)
        
        for kernel_name, info in summary.items():
            print(f"\n{kernel_name.upper()} - {info['description']}")
            print(f"  输出形状: {info['shape']}")
            print(f"  压缩比: {info['compression_ratio']:.1f}:1")
            print(f"  数值范围: [{info['min']:.4f}, {info['max']:.4f}]")
            print(f"  均值: {info['mean']:.4f}")
            print(f"  标准差: {info['std']:.4f}")
        
        print("\n" + "=" * 60)
    
    def print_performance_summary(self):
        """打印性能监控摘要信息"""
        current_time = time.time()
        total_time = current_time - self.start_time
        
        print("=" * 60)
        print("性能监控摘要")
        print("=" * 60)
        
        # 基本性能指标
        print(f"总运行时间: {total_time:.2f} 秒")
        print(f"总步数: {self.global_step}")
        
        if total_time > 0:
            overall_fps = self.global_step / total_time
            print(f"整体FPS: {overall_fps:.2f}")
        
        # 最近性能指标
        if len(self.fps_history) > 0:
            recent_fps = sum(self.fps_history) / len(self.fps_history)
            max_fps = max(self.fps_history)
            min_fps = min(self.fps_history)
            
            print(f"最近平均FPS: {recent_fps:.2f}")
            print(f"最高FPS: {max_fps:.2f}")
            print(f"最低FPS: {min_fps:.2f}")
        
        # 步骤时间统计
        if len(self.step_times) > 0:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            max_step_time = max(self.step_times)
            min_step_time = min(self.step_times)
            
            print(f"平均步骤时间: {avg_step_time:.4f} 秒")
            print(f"最长步骤时间: {max_step_time:.4f} 秒")
            print(f"最短步骤时间: {min_step_time:.4f} 秒")
        
        print("=" * 60)

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
    
    # 创建模型 - 使用新的并行多尺度架构
    model = StreamingAutoEncoder(
        input_channels=3,
        base_channels=8,        # 基础通道数（新架构中实际使用固定设计）
        latent_channels=4,     # 潜在空间维度（每个分支的输出通道数）
        lr=0.007,              # ObGD优化器学习率
        gamma=0.99,            # 动量衰减因子
        lamda=0.8,             # 损失函数权重平衡参数
        kappa=2.0,             # 损失稳定性参数
        debug_vis=True,        # 启用调试可视化
        use_tensorboard=True   # 启用TensorBoard日志记录
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
        'changed_pixels': [],
        'current_fps': [],
        'average_fps': []
    }
    
    try:
        while frame_count < total_frames:
            # 预处理当前帧
            curr_frame = preprocess_frame(obs)
            
            # 更新模型参数
            debug = (frame_count % debug_interval == 0)
            results = model.update_params(curr_frame, debug=debug)
            
            # 记录损失和FPS
            loss_history['global_loss'].append(results['global_loss'])
            loss_history['mse_loss'].append(results['mse_loss'])
            loss_history['changed_pixels'].append(results['changed_pixels'])
            loss_history['current_fps'].append(results['current_fps'])
            loss_history['average_fps'].append(results['average_fps'])
            
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
                recent_fps = np.mean(loss_history['current_fps'][-debug_interval:])
                recent_avg_fps = np.mean(loss_history['average_fps'][-debug_interval:])
                
                print(f"Frame {frame_count}/{total_frames}")
                print(f"  平均全局损失: {recent_global:.6f}")
                print(f"  平均MSE损失: {recent_mse:.6f}")
                print(f"  平均变化像素: {recent_changed:.0f}")
                print(f"  当前FPS: {recent_fps:.2f}")
                print(f"  平均FPS: {recent_avg_fps:.2f}")
                print("-" * 50)
    
    except KeyboardInterrupt:
        print("训练被用户中断")
    
    finally:
        env.close()
        print("训练完成！")
        
        # 打印性能摘要
        model.print_performance_summary()
        
        # 保存模型
        torch.save(model.state_dict(), 'streaming_autoencoder.pth')
        print("模型已保存到 streaming_autoencoder.pth")


if __name__ == "__main__":
    main()