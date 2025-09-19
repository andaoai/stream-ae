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
from config import DEVICE, MODEL_CONFIG, TRAINING_CONFIG
from loss import LossPriorityQueue, BatchLossTracker


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
                 debug_vis=False, use_tensorboard=True, log_dir=None,
                 batch_size=4, queue_size=24, min_loss_threshold=0.001):
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
            batch_size: 批量大小（包含当前帧）
            queue_size: 损失优先级队列大小
            min_loss_threshold: 最低损失阈值
        """
        super(StreamingAutoEncoder, self).__init__()

        # 模型架构
        self.encoder = Encoder(input_channels, latent_channels)
        self.decoder = Decoder()
        self.change_detector = PixelChangeDetector()

        # 初始化权重
        self.apply(initialize_weights)

        # 移动模型到GPU
        self.to(DEVICE)

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # 监控组件
        self.perf_monitor = PerformanceMonitor()
        self.tensorboard_logger = TensorBoardLogger(log_dir) if use_tensorboard else None
        self.feature_visualizer = FeatureVisualizer(debug_vis)

        # 状态变量
        self.prev_frame = None
        self.prev_embeddings = None
        self._current_embeddings = None
        self.global_step = 0

        # 批量训练相关
        self.batch_size = batch_size
        self.loss_queue = LossPriorityQueue(max_size=queue_size, min_loss_threshold=min_loss_threshold)
        self.batch_loss_tracker = BatchLossTracker()
        self.current_frame_loss = 0.0
        self.use_batch_training = batch_size > 1
    
    def encode(self, x):
        """编码"""
        fused_features = self.encoder(x)

        # 存储特征图用于可视化
        self.feature_visualizer.store_feature_map('input', x)

        # 存储融合特征图（取第一个batch）
        self.feature_visualizer.store_feature_map('fused_embedding', fused_features[:1])

        # 保存当前embedding用于统计
        self._current_embeddings = fused_features.detach()

        return fused_features
    
    def decode(self, fused_features):
        """解码"""
        reconstruction = self.decoder(fused_features)

        # 存储输出特征图
        self.feature_visualizer.store_feature_map('output', reconstruction)

        return reconstruction
    
    def forward(self, x):
        """前向传播"""
        fused_features = self.encode(x)
        reconstruction = self.decode(fused_features)
        return reconstruction, fused_features
    
    def update_params(self, curr_frame, debug=False):
        """
        执行一步训练更新

        Args:
            curr_frame: 当前帧
            debug: 是否输出调试信息

        Returns:
            dict: 包含损失、重建结果等信息的字典
        """
        if self.use_batch_training:
            return self.update_params_batch(curr_frame, debug)
        else:
            return self.update_params_single(curr_frame, debug)

    def update_params_single(self, curr_frame, debug=False):
        """
        执行单帧训练更新（保持原有的单帧训练逻辑）

        Args:
            curr_frame: 当前帧
            debug: 是否输出调试信息

        Returns:
            dict: 包含损失、重建结果等信息的字典
        """
        # 更新性能监控
        step_time = self.perf_monitor.update_step_time()

        # 移动输入数据到GPU
        curr_frame = curr_frame.to(DEVICE)
        if self.prev_frame is not None:
            self.prev_frame = self.prev_frame.to(DEVICE)

        # 前向传播
        reconstruction, fused_features = self(curr_frame)

        # 检测像素变化
        change_mask, change_intensity = None, None
        if self.prev_frame is not None:
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

        # 更新历史信息（保持在GPU上）
        self.prev_frame = curr_frame.detach().clone()
        self.prev_embeddings = fused_features.detach().clone()

        # 记录日志
        if self.tensorboard_logger:
            # 同步步数计数器
            self.tensorboard_logger.global_step = self.global_step
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
            'embeddings': fused_features,
            'change_mask': change_mask,
            'change_intensity': change_intensity,
            'fps': self.perf_monitor.get_avg_fps(),
            'step_time': step_time,
            'batch_size': 1,
            'batch_training': False
        }

    def update_params_batch(self, curr_frame, debug=False):
        """
        执行批量训练更新

        Args:
            curr_frame: 当前帧
            debug: 是否输出调试信息

        Returns:
            dict: 包含损失、重建结果等信息的字典
        """
        # 更新性能监控
        step_time = self.perf_monitor.update_step_time()

        # 移动输入数据到GPU
        curr_frame = curr_frame.to(DEVICE)

        # 确保当前帧是4D张量 [batch, channels, height, width]
        if curr_frame.dim() == 3:
            curr_frame = curr_frame.unsqueeze(0)  # [1, 3, 224, 224]

        # 首先计算当前帧的损失
        with torch.no_grad():
            curr_reconstruction, curr_fused_features = self(curr_frame)
            curr_global_loss, curr_mse_loss, curr_l1_loss, curr_ssim_loss = compute_global_loss(curr_frame, curr_reconstruction)
            self.current_frame_loss = curr_global_loss.item()

        # 尝试将当前帧加入损失队列
        frame_added = self.loss_queue.add_frame(curr_frame, self.current_frame_loss)

        # 获取批量数据
        batch_tensor, frame_ids = self.loss_queue.get_batch(self.batch_size, curr_frame)

        # 确保批量张量是正确的维度
        if batch_tensor.dim() == 5:  # 如果是5D张量 [2, batch, channels, height, width]
            batch_tensor = batch_tensor.reshape(-1, *batch_tensor.shape[2:])  # 重新塑形为 [total_batch, channels, height, width]

        # 执行批量训练
        batch_results = self._train_batch(batch_tensor, frame_ids)

        # 更新队列中帧的损失
        self._update_queue_losses(batch_results)

        # 记录日志
        if self.tensorboard_logger:
            self.tensorboard_logger.global_step = self.global_step
            self._log_batch_training_step(batch_results, curr_frame)

        self.global_step += 1

        # 返回训练结果
        return {
            'loss': batch_results['total_loss'],
            'mse_loss': batch_results['total_mse_loss'],
            'l1_loss': batch_results['total_l1_loss'],
            'ssim_loss': batch_results['total_ssim_loss'],
            'reconstruction': batch_results['reconstructions'][0],  # 返回当前帧的重建结果
            'embeddings': batch_results['embeddings'][0:1],  # 返回当前帧的embedding
            'change_mask': batch_results.get('change_mask'),
            'change_intensity': batch_results.get('change_intensity'),
            'fps': self.perf_monitor.get_avg_fps(),
            'step_time': step_time,
            'batch_size': len(batch_results['losses']),
            'batch_training': True,
            'queue_stats': self.loss_queue.get_stats(),
            'batch_losses': batch_results['losses']
        }

    def _train_batch(self, batch_tensor, frame_ids):
        """
        执行批量训练

        Args:
            batch_tensor: 批量数据张量
            frame_ids: 帧ID列表

        Returns:
            dict: 批量训练结果
        """
        batch_size = batch_tensor.shape[0]

        # 前向传播
        reconstructions, fused_features = self(batch_tensor)

        # 计算每帧的损失
        batch_losses = []
        batch_mse_losses = []
        batch_l1_losses = []
        batch_ssim_losses = []

        total_loss = 0.0
        total_mse_loss = 0.0
        total_l1_loss = 0.0
        total_ssim_loss = 0.0

        # 首先计算所有帧的损失
        frame_losses = []
        for i in range(batch_size):
            frame_recon = reconstructions[i:i+1]
            frame_input = batch_tensor[i:i+1]

            global_loss, mse_loss, l1_loss, ssim_loss = compute_global_loss(frame_input, frame_recon)

            frame_losses.append(global_loss)
            batch_losses.append(global_loss.item())
            batch_mse_losses.append(mse_loss.item())
            batch_l1_losses.append(l1_loss.item())
            batch_ssim_losses.append(ssim_loss.item())

            total_mse_loss += mse_loss
            total_l1_loss += l1_loss
            total_ssim_loss += ssim_loss

        # 计算总损失（确保精确匹配）
        total_loss = torch.stack(frame_losses).sum()

        # 反向传播和参数更新（使用总损失）
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # 参数更新
        self.optimizer.step()

        # 检测像素变化（使用第一帧）
        change_mask = None
        change_intensity = None
        if self.prev_frame is not None:
            change_mask, change_intensity = self.change_detector.detect_changes(self.prev_frame, batch_tensor[0:1])

        # 更新历史信息
        self.prev_frame = batch_tensor[0:1].detach().clone()
        self.prev_embeddings = fused_features[0:1].detach().clone()

        return {
            'total_loss': total_loss.item(),
            'total_mse_loss': total_mse_loss.item(),
            'total_l1_loss': total_l1_loss.item(),
            'total_ssim_loss': total_ssim_loss.item(),
            'losses': batch_losses,
            'mse_losses': batch_mse_losses,
            'l1_losses': batch_l1_losses,
            'ssim_losses': batch_ssim_losses,
            'reconstructions': reconstructions,
            'embeddings': fused_features,
            'frame_ids': frame_ids,
            'change_mask': change_mask,
            'change_intensity': change_intensity
        }

    def _update_queue_losses(self, batch_results):
        """
        更新队列中帧的损失

        Args:
            batch_results: 批量训练结果
        """
        frame_ids = batch_results['frame_ids']
        losses = batch_results['losses']

        # 跳过当前帧（第一个）
        for i in range(1, len(frame_ids)):
            frame_id = frame_ids[i]
            new_loss = losses[i]

            # 更新优先级队列中的损失
            self.loss_queue.update_frame_loss(frame_id, new_loss)

    def _log_batch_training_step(self, batch_results, curr_frame):
        """
        记录批量训练步骤的TensorBoard日志

        Args:
            batch_results: 批量训练结果
            curr_frame: 当前帧
        """
        logger = self.tensorboard_logger

        # 记录总损失
        logger.log_scalar('Loss/Batch_Total_Loss', batch_results['total_loss'])
        logger.log_scalar('Loss/Batch_Total_MSE_Loss', batch_results['total_mse_loss'])
        logger.log_scalar('Loss/Batch_Total_L1_Loss', batch_results['total_l1_loss'])
        logger.log_scalar('Loss/Batch_Total_SSIM_Loss', batch_results['total_ssim_loss'])

        # 记录队列统计信息
        queue_stats = self.loss_queue.get_stats()
        logger.log_scalar('Queue/Size', queue_stats['queue_size'])
        logger.log_scalar('Queue/Min_Loss', queue_stats['min_loss'])
        logger.log_scalar('Queue/Max_Loss', queue_stats['max_loss'])
        logger.log_scalar('Queue/Avg_Loss', queue_stats['avg_loss'])

        # 记录当前帧损失
        logger.log_scalar('Current_Frame/Loss', self.current_frame_loss)

        # 记录批量损失分布
        for i, loss in enumerate(batch_results['losses']):
            logger.log_scalar(f'Batch_Losses/Frame_{i}', loss)

        # 记录性能指标
        logger.log_scalar('Performance/Current_FPS', self.perf_monitor.get_avg_fps())

        # 记录图像（当前帧）
        self._log_images(batch_results['reconstructions'][0:1], curr_frame)

        # 刷新TensorBoard
        logger.flush()
    
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
        
        # 记录embedding统计信息
        if hasattr(self, '_current_embeddings'):
            # 记录融合embedding的统计信息
            logger.log_scalar('Embedding/Fused_Mean', self._current_embeddings.mean().item())
            logger.log_scalar('Embedding/Fused_Std', self._current_embeddings.std().item())
            logger.log_scalar('Embedding/Fused_Norm', torch.norm(self._current_embeddings).item())
        
        # 每步都记录图像，实现真正的实时更新
        self._log_images(reconstruction, curr_frame)
    
    def _log_images(self, reconstruction, curr_frame):
        """记录图像到TensorBoard"""
        logger = self.tensorboard_logger

        # 确保图像在[0,1]范围内并移动到CPU
        input_img = torch.clamp(curr_frame[0], 0, 1).cpu()
        recon_img = torch.clamp(reconstruction[0], 0, 1).cpu()

        # 记录基础图像
        logger.log_images('00_Input', input_img.unsqueeze(0))
        logger.log_images('01_Reconstruction', recon_img.unsqueeze(0))

        # 记录重建误差
        error_map = torch.abs(curr_frame - reconstruction)
        error_img = torch.clamp(error_map[0], 0, 1).cpu()
        logger.log_images('02_Reconstruction_Error', error_img.unsqueeze(0))

        # 记录特征图（如果启用调试）
        if self.feature_visualizer.debug_vis:
            self._log_feature_maps()

        # 刷新TensorBoard以确保实时更新
        logger.flush()
    
    def _log_feature_maps(self):
        """记录特征图到TensorBoard"""
        logger = self.tensorboard_logger

        # 记录融合embedding的可视化
        embedding_map = self.feature_visualizer.get_feature_visualization('fused_embedding')
        if embedding_map is not None:
            # 将numpy数组转换为torch张量并添加批次维度
            feature_tensor = torch.from_numpy(embedding_map).unsqueeze(0).unsqueeze(0)
            logger.log_images('03_Features/fused_embedding', feature_tensor)
    
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
                       debug_vis=False, use_tensorboard=True, log_dir=None,
                       batch_size=4, queue_size=24, min_loss_threshold=0.001):
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
        log_dir=log_dir,
        batch_size=batch_size,
        queue_size=queue_size,
        min_loss_threshold=min_loss_threshold
    )