"""
监控和可视化模块
负责TensorBoard日志、性能监控、调试可视化等功能
"""

import torch
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.start_time = time.time()
        self.last_step_time = self.start_time
        self.step_times = deque(maxlen=window_size)
        self.fps_history = deque(maxlen=window_size)
    
    def update_step_time(self):
        """更新步骤时间"""
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.step_times.append(step_time)
        
        # 计算FPS
        if step_time > 0:
            fps = 1.0 / step_time
        else:
            fps = 0.0
        self.fps_history.append(fps)
        
        self.last_step_time = current_time
        return step_time
    
    def get_avg_fps(self):
        """获取平均FPS"""
        if len(self.fps_history) == 0:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)
    
    def get_avg_step_time(self):
        """获取平均步骤时间"""
        if len(self.step_times) == 0:
            return 0.0
        return sum(self.step_times) / len(self.step_times)


class TensorBoardLogger:
    """TensorBoard日志记录器"""
    
    def __init__(self, log_dir=None):
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"runs/streaming_ae_{timestamp}"
        
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        print(f"TensorBoard日志将保存到: {log_dir}")
    
    def log_scalar(self, tag, value, step=None):
        """记录标量"""
        if step is None:
            step = self.global_step
        self.writer.add_scalar(tag, value, step)
    
    def log_images(self, tag, images, step=None):
        """记录图像"""
        if step is None:
            step = self.global_step
        self.writer.add_images(tag, images, step)
    
    def log_histogram(self, tag, values, step=None):
        """记录直方图"""
        if step is None:
            step = self.global_step
        self.writer.add_histogram(tag, values, step)
    
    def increment_step(self):
        """增加步数"""
        self.global_step += 1
    
    def flush(self):
        """刷新TensorBoard缓存"""
        if self.writer:
            self.writer.flush()
    
    def close(self):
        """关闭TensorBoard"""
        if self.writer:
            self.writer.close()


class FeatureVisualizer:
    """特征可视化器"""
    
    def __init__(self, debug_vis=False):
        self.debug_vis = debug_vis
        self.feature_maps = {}
    
    def store_feature_map(self, name, tensor):
        """存储特征图"""
        if self.debug_vis:
            if isinstance(tensor, tuple):
                # 处理元组类型的特征图
                self.feature_maps[name] = tuple(t.detach() for t in tensor)
            else:
                self.feature_maps[name] = tensor.detach()
    
    def get_feature_visualization(self, layer_name, channel_idx=None):
        """获取特征可视化"""
        if not self.debug_vis or layer_name not in self.feature_maps:
            return None
            
        feature_map = self.feature_maps[layer_name]
        
        if channel_idx is not None:
            # 特定通道
            if len(feature_map.shape) > 1 and channel_idx < feature_map.shape[1]:
                channel_data = feature_map[0, channel_idx].cpu().numpy()
                return channel_data
            else:
                return None
        else:
            # 处理不同形状的特征图
            if len(feature_map.shape) == 4:  # [B, C, H, W]
                if feature_map.shape[1] == 1:  # 单通道
                    return feature_map[0, 0].cpu().numpy()
                else:  # 多通道，取平均
                    mean_feature = torch.mean(feature_map, dim=1, keepdim=True)
                    return mean_feature[0, 0].cpu().numpy()
            elif len(feature_map.shape) == 3:  # [B, H, W]
                return feature_map[0].cpu().numpy()
            else:  # [H, W]
                return feature_map.cpu().numpy()
    
    def clear_feature_maps(self):
        """清空特征图"""
        self.feature_maps.clear()