# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

流式视频自编码器项目，实现视频帧压缩和重建的在线学习。使用自定义 ObGD（在线梯度下降）优化器，采用并行多尺度架构。**重要：必须使用 uv 执行。**

## 关键命令

### 环境设置和运行
```bash

# 使用 uv 安装依赖（推荐）
uv pip install -r requirements.txt

# 使用 uv 运行主程序（推荐）
uv run main.py

# 启动 TensorBoard 监控
tensorboard --logdir=runs
# 在浏览器中查看 http://localhost:6006
```

### 开发工作流
```bash
# 激活环境并运行 uv run main.py

# 在另一个终端启动 TensorBoard
tensorboard --logdir=runs
```

### ALE 环境设置
```bash
# ALE (Atari Learning Environment) 已正确配置
# 包含所需的游戏 ROMs：
# - Breakout
# - Assault  
# - SpaceInvaders
# - Pacman
# - Asteroids

# 环境注册已在 main.py 中完成：
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)
```

## Architecture Overview

### Core Components

1. **StreamingAutoEncoder** (`autoencoder.py:15`):
   - Main model class integrating all components
   - Parallel multi-scale architecture with three convolutional branches
   - Integrated monitoring and visualization system
   - No-padding design preserving edge information
   - Compression ratios: 92:1, 161:1, 190:1

2. **Encoder** (`model.py:228`):
   - Three parallel branches with different kernel sizes
   - Small (3×3): 224×224×3 → 27×27×3, texture features
   - Medium (5×5): 224×224×3 → 25×25×2, balanced features
   - Large (7×7): 224×224×3 → 23×23×2, structural features
   - No-padding design preserving edge information
   - Default latent_channels=3 (updated from 4)

3. **Decoder** (`model.py:314`):
   - Symmetric three-branch decoder architecture
   - Adaptive fusion module for size unification
   - Final sigmoid activation for output normalization
   - Bilinear interpolation for size alignment

4. **ObGD Optimizer** (`optim.py:13`):
   - Online Gradient Descent optimizer for streaming learning
   - Momentum updates with adaptive learning rates
   - Memory-efficient design without full gradient history storage
   - Key parameters: `lr=1.0`, `gamma=0.99`, `lamda=0.8`, `kappa=2.0`
   - Default learning rate is 1.0 (not 0.01 as previously used)

5. **Loss Functions** (`loss.py:12`):
   - Multi-component loss: MSE + 0.5*L1 + 0.1*SSIM
   - Global loss for overall reconstruction quality
   - Designed for streaming video frame reconstruction

6. **Monitoring System** (`monitoring.py`):
   - TensorBoardLogger: Real-time TensorBoard integration
   - PerformanceMonitor: FPS and step time tracking
   - FeatureVisualizer: Feature map visualization

7. **Model Components** (`model.py`):
   - `LayerNormalization`: Dynamic layer normalization for streaming data
   - `PixelChangeDetector`: Frame change detection with adaptive threshold
   - `ResidualBlock`: ResNet-style residual connections
   - `TransformerBlock`: Deep feature processing with attention

8. **Utilities** (`utils.py`):
   - `preprocess_frame`: Frame preprocessing and normalization
   - `postprocess_output`: Output postprocessing
   - `compute_frame_difference`: Frame difference calculation

### Data Flow

1. **Input**: 224×224×3 RGB frames from Gymnasium environments
2. **Preprocessing**: Frame resize, normalization, tensor conversion (`utils.py`)
3. **Encoding**: Three parallel branches extract multi-scale features
4. **Change Detection**: Pixel-level change detection for attention mechanism
5. **Compression**: Each branch produces different embedding sizes
6. **Loss Calculation**: Multi-component loss computation
7. **Optimization**: ObGD optimizer with gradient clipping
8. **Decoding**: Parallel reconstruction with adaptive fusion
9. **Monitoring**: Real-time TensorBoard logging and visualization
10. **Output**: Reconstructed 224×224×3 frames

## Development Workflow

### Environment Setup
```bash

# Install dependencies with uv
uv pip install -r requirements.txt

# Run main program
uv run main.py

# Start TensorBoard in another terminal
tensorboard --logdir=runs
```

### Development Notes

### File Structure
- `autoencoder.py`: Main model implementation (StreamingAutoEncoder class)
- `main.py`: Entry point with live viewer functionality
- `model.py`: Model components (Encoder, Decoder, LayerNormalization, etc.)
- `optim.py`: ObGD optimizer and weight initialization
- `loss.py`: Loss function implementations
- `monitoring.py`: TensorBoard monitoring and performance analysis
- `utils.py`: Data preprocessing and utility functions
- `requirements.txt`: Dependencies including PyTorch 2.3.0, TensorBoard 2.16.2

### Key Dependencies
- PyTorch 2.3.0 with TensorBoard 2.16.2
- Gymnasium 0.29.1 for environment interaction
- MuJoCo 3.3.5 for physics simulation (upgraded from 2.3.7)
- ALE (Atari Learning Environment) with ROMs for game environments
- dm-control 1.0.31 for control environments
- stable-baselines3 2.7.0 for reinforcement learning algorithms
- OpenCV 4.9.0.80 for frame processing
- NumPy 1.26.4 and scientific computing libraries
- Additional ML libraries: matplotlib, scipy, pandas

### Training Characteristics
- Online learning with frame-by-frame updates
- No batching required (streaming architecture)
- Typical training: 500,000 frames over 2-3 hours on GPU
- Target metrics: global loss ~0.0156, MSE loss ~0.0089
- Real-time TensorBoard monitoring with per-frame updates
- Gradient clipping (max_norm=1.0) for stability

### Code Style
- Chinese comments throughout the codebase
- Modular design with clear separation of concerns
- Extensive TensorBoard integration for monitoring
- Sparse initialization (50% sparsity) to avoid gradient vanishing
- Factory pattern for model creation (`create_streaming_ae`)

## Testing and Validation

### Performance Monitoring
- TensorBoard tracks: loss curves, reconstruction quality, feature maps
- Real-time visualization of input/output frames
- Change detection and attention mechanisms
- Parameter distribution monitoring

### Environment Integration
- Tested with Gymnasium environments (ALE, dm-control)
- ALE environments properly registered with `gym.register_envs(ale_py)`
- Atari ROMs installed and configured for game environments
- Frame preprocessing: resize, normalize, convert to tensor
- Supports various game environments for testing

### Runtime Mode
Single runtime mode: live viewer
- Randomly selects from ALE game environments (Breakout, Assault, SpaceInvaders, Pacman, Asteroids)
- ALE environments properly registered with `gym.register_envs(ale_py)`
- Real-time TensorBoard monitoring with per-frame updates and flush
- Supports loading pretrained model if `quick_demo_model.pth` exists
- Automatic TensorBoard writer flushing for real-time visualization
- Performance monitoring: FPS tracking, step time measurement
- Stop with Ctrl+C

### Key Implementation Details

#### Model Creation
```python
# Factory function for model creation
model = create_streaming_ae(
    input_channels=3,
    latent_channels=3,  # Updated from 4 to 3
    lr=1.0,             # Learning rate (default: 1.0)
    debug_vis=True,
    use_tensorboard=True,
    log_dir=f"runs/live_viewer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
```

#### Training Loop
```python
# Frame-by-frame training update
results = model.update_params(curr_frame, debug=True)
# Returns: loss, reconstruction, embeddings, change_mask, fps, etc.
```

#### Loss Function Weights
- MSE loss: Primary reconstruction error (weight=1.0)
- L1 loss: Stable gradients (weight=0.5)
- SSIM loss: Structural similarity (weight=0.1)

#### Monitoring Features
- Real-time loss tracking
- Feature map visualization
- Performance metrics (FPS, step time)
- Change detection statistics
- Model parameter distribution