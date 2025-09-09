# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

流式视频自编码器项目，实现视频帧压缩和重建的在线学习。使用自定义 ObGD（在线梯度下降）优化器，采用并行多尺度架构。**重要：必须使用 drl conda 环境和 uv 执行。**

## 关键命令

### 环境设置和运行
```bash
# 激活 DRL conda 环境
conda activate drl

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
# 激活环境并运行
conda activate drl && uv run main.py

# 在另一个终端启动 TensorBoard
conda activate drl && tensorboard --logdir=runs
```

## Architecture Overview

### Core Components

1. **StreamingAutoEncoder** (`streaming_video_autoencoder.py:25`):
   - Parallel multi-scale architecture with three convolutional branches
   - Small (3×3), medium (5×5), and large (7×7) kernel branches
   - No-padding design preserving edge information
   - Compression ratios: 92:1, 161:1, 190:1

2. **ObGD Optimizer** (`optim.py:13`):
   - Online Gradient Descent optimizer for streaming learning
   - Momentum updates with adaptive learning rates
   - Memory-efficient design without full gradient history storage
   - Key parameters: `lr=1.0`, `gamma=0.99`, `lamda=0.8`, `kappa=2.0`

3. **Loss Functions** (`loss.py:12`):
   - Multi-component loss: MSE + L1 + SSIM
   - Global loss for overall reconstruction quality
   - Designed for streaming video frame reconstruction

4. **Model Components** (`model.py`):
   - `LayerNormalization`: Dynamic layer normalization for streaming data
   - `PixelChangeDetector`: Frame change detection
   - `Encoder`/`Decoder`: Multi-scale encoding/decoding branches

### Data Flow

1. **Input**: 224×224×3 RGB frames from Gymnasium environments
2. **Encoding**: Three parallel branches extract multi-scale features
3. **Compression**: Each branch produces different embedding sizes
4. **Decoding**: Parallel reconstruction with adaptive fusion
5. **Output**: Reconstructed 224×224×3 frames

## Development Notes

### File Structure
- `streaming_video_autoencoder.py`: Main model implementation (27k lines)
- `main.py`: Entry point with live viewer functionality
- `model.py`: Model components (Encoder, Decoder, normalization)
- `optim.py`: ObGD optimizer and weight initialization
- `loss.py`: Loss function implementations
- `sparse_init.py`: Sparse weight initialization utilities

### Key Dependencies
- PyTorch 2.3.0 with TensorBoard 2.16.2
- Gymnasium 0.29.1 for environment interaction
- OpenCV for frame processing
- NumPy and scientific computing libraries

### Training Characteristics
- Online learning with frame-by-frame updates
- No batching required (streaming architecture)
- Typical training: 500,000 frames over 2-3 hours on GPU
- Target metrics: detail loss ~0.0234, global loss ~0.0156

### Code Style
- Chinese comments throughout the codebase
- Modular design with clear separation of concerns
- Extensive TensorBoard integration for monitoring
- Sparse initialization (90% sparsity) for better generalization

## Testing and Validation

### Performance Monitoring
- TensorBoard tracks: loss curves, reconstruction quality, feature maps
- Real-time visualization of input/output frames
- Change detection and attention mechanisms
- Parameter distribution monitoring

### Environment Integration
- Tested with Gymnasium environments (ALE, dm-control)
- Frame preprocessing: resize, normalize, convert to tensor
- Supports various game environments for testing

### Runtime Mode
Single runtime mode: live viewer
- Randomly selects from game environments (Breakout, Assault, SpaceInvaders, Pacman, Asteroids)
- Real-time TensorBoard monitoring with per-frame updates
- Supports loading pretrained model if `quick_demo_model.pth` exists
- Stop with Ctrl+C