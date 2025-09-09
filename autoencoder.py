"""
流式训练器模块
负责训练逻辑、参数更新、损失计算等核心功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import compute_global_loss
from optim import initialize_weights, ObGD
from model import PixelChangeDetector, Encoder, Decoder
from monitoring import TensorBoardLogger, PerformanceMonitor, FeatureVisualizer


class StreamingAutoEncoder(nn.Module):
    """
    流式视频自编码器
    
    核心职责：
    1. 定义模型架构
    2. 提供前向传播接口
    3. 管理模型组件
    """

    def __init__(self, input_channels=3, latent_channels=3, 
                 lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0,
                 debug_vis=False, use_tensorboard=True, log_dir=None):
        """
        初始化流式自编码器
        
        Args:
            input_channels: 输入通道数
            latent_channels: 潜在空间维度
            lr: 学习率
            gamma: 动量衰减因子
            lamda: 损失权重参数
            kappa: 损失稳定性参数
            debug_vis: 是否启用调试可视化
            use_tensorboard: 是否启用TensorBoard
            log_dir: 日志目录
        """
        super(StreamingAutoEncoder, self).__init__()
        
        # 模型架构
        self.encoder = Encoder(input_channels, latent_channels)
        self.decoder = Decoder(latent_channels, input_channels)
        self.change_detector = PixelChangeDetector()
        
        # 初始化权重
        self.apply(initialize_weights)
        
        # 优化器
        self.optimizer = ObGD(self.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        
        # 监控组件
        self.perf_monitor = PerformanceMonitor()
        self.tensorboard_logger = TensorBoardLogger(log_dir) if use_tensorboard else None
        self.feature_visualizer = FeatureVisualizer(debug_vis)
        
        # 状态变量
        self.prev_frame = None
        self.prev_embedding = None
        self.global_step = 0
    
    def encode(self, x):
        """编码"""
        small_emb, medium_emb, large_emb = self.encoder(x)
        
        # 存储特征图用于可视化
        self.feature_visualizer.store_feature_map('input', x)
        self.feature_visualizer.store_feature_map('small_embedding', small_emb)
        self.feature_visualizer.store_feature_map('medium_embedding', medium_emb)
        self.feature_visualizer.store_feature_map('large_embedding', large_emb)
        self.feature_visualizer.store_feature_map('bottleneck', (small_emb, medium_emb, large_emb))
        
        return small_emb, medium_emb, large_emb
    
    def decode(self, small_emb, medium_emb, large_emb):
        """解码"""
        reconstruction = self.decoder(small_emb, medium_emb, large_emb)
        
        # 存储输出特征图
        self.feature_visualizer.store_feature_map('output', reconstruction)
        
        return reconstruction
    
    def forward(self, x):
        """前向传播"""
        small_emb, medium_emb, large_emb = self.encode(x)
        reconstruction = self.decode(small_emb, medium_emb, large_emb)
        return reconstruction, (small_emb, medium_emb, large_emb)
    
    def update_params(self, curr_frame, debug=False):
        """
        执行一步训练更新
        
        Args:
            curr_frame: 当前帧
            debug: 是否输出调试信息
            
        Returns:
            dict: 包含损失、重建结果等信息的字典
        """
        # 更新性能监控
        step_time = self.perf_monitor.update_step_time()
        
        # 前向传播
        reconstruction, embeddings = self(curr_frame)
        small_emb, medium_emb, large_emb = embeddings
        
        # 检测像素变化
        change_mask, change_intensity = self.change_detector.detect_changes(self.prev_frame, curr_frame)
        
        # 计算损失
        global_loss, mse_loss, l1_loss, ssim_loss = compute_global_loss(curr_frame, reconstruction)
        
        # 反向传播和参数更新
        self.optimizer.zero_grad()
        global_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # 参数更新
        self.optimizer.step()
        
        # 更新历史信息
        self.prev_frame = curr_frame.detach().clone()
        self.prev_embedding = (small_emb.detach().clone(), medium_emb.detach().clone(), large_emb.detach().clone())
        
        # 记录日志
        if self.tensorboard_logger:
            self._log_training_step(global_loss, mse_loss, l1_loss, ssim_loss, 
                                   change_mask, reconstruction, curr_frame)
        
        self.global_step += 1
        
        # 返回训练结果
        return {
            'loss': global_loss.item(),
            'mse_loss': mse_loss.item(),
            'l1_loss': l1_loss.item(),
            'ssim_loss': ssim_loss.item(),
            'reconstruction': reconstruction,
            'embeddings': embeddings,
            'change_mask': change_mask,
            'change_intensity': change_intensity,
            'fps': self.perf_monitor.get_avg_fps(),
            'step_time': step_time
        }
    
    def _log_training_step(self, global_loss, mse_loss, l1_loss, ssim_loss, 
                          change_mask, reconstruction, curr_frame):
        """记录训练步骤的TensorBoard日志"""
        logger = self.tensorboard_logger
        
        # 记录损失
        logger.log_scalar('Loss/Global_Loss', global_loss.item())
        logger.log_scalar('Loss/MSE_Loss', mse_loss.item())
        logger.log_scalar('Loss/L1_Loss', l1_loss.item())
        logger.log_scalar('Loss/SSIM_Loss', ssim_loss.item())
        
        # 记录变化统计
        logger.log_scalar('Metrics/Changed_Pixels', torch.sum(change_mask).item())
        
        # 记录性能指标
        logger.log_scalar('Performance/Current_FPS', self.perf_monitor.get_avg_fps())
        logger.log_scalar('Performance/Average_FPS', self.perf_monitor.get_avg_fps())
        
        # 每20步记录图像
        if self.global_step % 20 == 0:
            self._log_images(reconstruction, curr_frame)
    
    def _log_images(self, reconstruction, curr_frame):
        """记录图像到TensorBoard"""
        logger = self.tensorboard_logger
        
        # 确保图像在[0,1]范围内
        input_img = torch.clamp(curr_frame[0], 0, 1)
        recon_img = torch.clamp(reconstruction[0], 0, 1)
        
        # 记录基础图像
        logger.log_images('00_Input', input_img.unsqueeze(0))
        logger.log_images('01_Reconstruction', recon_img.unsqueeze(0))
        
        # 记录重建误差
        error_map = torch.abs(curr_frame - reconstruction)
        error_img = torch.clamp(error_map[0], 0, 1)
        logger.log_images('02_Reconstruction_Error', error_img.unsqueeze(0))
        
        # 记录特征图（如果启用调试）
        if self.feature_visualizer.debug_vis:
            self._log_feature_maps()
    
    def _log_feature_maps(self):
        """记录特征图到TensorBoard"""
        logger = self.tensorboard_logger
        
        # 记录三个分支的embedding
        for branch_name in ['small_embedding', 'medium_embedding', 'large_embedding']:
            feature_map = self.feature_visualizer.get_feature_visualization(branch_name)
            if feature_map is not None:
                # 将numpy数组转换为torch张量并添加批次维度
                feature_tensor = torch.from_numpy(feature_map).unsqueeze(0).unsqueeze(0)
                logger.log_images(f'03_Features/{branch_name}', feature_tensor)
    
    def get_performance_summary(self):
        """获取性能摘要"""
        return {
            'avg_fps': self.perf_monitor.get_avg_fps(),
            'avg_step_time': self.perf_monitor.get_avg_step_time(),
            'total_steps': self.global_step
        }
    
    def close(self):
        """关闭模型"""
        if self.tensorboard_logger:
            self.tensorboard_logger.close()


# 为了保持向后兼容，保留一些有用的方法
def create_streaming_ae(input_channels=3, latent_channels=3, lr=1.0, 
                       gamma=0.99, lamda=0.8, kappa=2.0, 
                       debug_vis=False, use_tensorboard=True, log_dir=None):
    """创建流式自编码器的工厂函数"""
    return StreamingAutoEncoder(
        input_channels=input_channels,
        latent_channels=latent_channels,
        lr=lr,
        gamma=gamma,
        lamda=lamda,
        kappa=kappa,
        debug_vis=debug_vis,
        use_tensorboard=use_tensorboard,
        log_dir=log_dir
    )