"""
配置模块

集中管理项目的所有配置参数，包括设备配置、模型参数、训练参数等。
避免在多个文件中重复定义相同的配置。
"""

import torch
import os

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 模型配置
MODEL_CONFIG = {
    'input_channels': 3,
    'latent_channels': 3,
    'lr': 0.005,
    'gamma': 0.99,
    'lamda': 0.8,
    'kappa': 2.0,
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 1,  # 流式训练，每帧一个批次
    'gradient_clip_norm': 1.0,
    'debug_vis': True,
    'use_tensorboard': True,
}

# 数据配置
DATA_CONFIG = {
    'frame_size': (224, 224),
    'normalization_range': (0.0, 1.0),
}

# 监控配置
MONITORING_CONFIG = {
    'performance_window_size': 100,
    'tensorboard_log_dir': 'runs',
    'save_model_path': 'quick_demo_model.pth',
}

# 环境配置
ENV_CONFIG = {
    'available_games': [
        'ALE/Breakout-v5',
        'ALE/Assault-v5',
        'ALE/SpaceInvaders-v5',
        'ALE/Pacman-v5',
        'ALE/Asteroids-v5'
    ],
    'render_mode': 'rgb_array',
}

# 便捷函数
def get_device():
    """获取当前设备"""
    return DEVICE

def is_cuda_available():
    """检查CUDA是否可用"""
    return torch.cuda.is_available()

def get_gpu_info():
    """获取GPU信息"""
    if is_cuda_available():
        return {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'device_name': torch.cuda.get_device_name(0),
            'device_capability': torch.cuda.get_device_capability(0),
        }
    else:
        return {'available': False}

def print_config():
    """打印当前配置信息"""
    print("=" * 50)
    print("项目配置信息")
    print("=" * 50)
    print(f"设备: {DEVICE}")

    if is_cuda_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")

    print(f"模型配置: {MODEL_CONFIG}")
    print(f"训练配置: {TRAINING_CONFIG}")
    print(f"数据配置: {DATA_CONFIG}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()