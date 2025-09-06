# 流式视频自编码器 (Streaming Video Autoencoder)

基于ObGD优化器的流式视频自编码器实现，参考了 [streaming-drl](https://github.com/mohmdelsayed/streaming-drl) 项目的在线学习思想。

## 项目概述

本项目实现了一个能够处理连续视频流的自编码器，具有以下核心特性：

### 🚀 核心特性

1. **在线学习** - 基于ObGD（Online Gradient Descent）优化器，无需存储历史数据
2. **双损失函数** - 细节损失和全局损失的平衡设计
3. **混合架构** - CNN + Vision Transformer的混合设计
4. **实时处理** - 支持流式数据的实时处理和可视化
5. **智能检测** - 像素级变化检测和注意力机制

### 🏗️ 架构设计

#### 编码器架构
```
输入: 224×224×3 (RGB图像)
  ↓
Conv2d + LayerNorm + LeakyReLU: 224×224×3 → 112×112×8
  ↓
Conv2d + LayerNorm + LeakyReLU: 112×112×8 → 56×56×16
  ↓
SelfAttention2D + Conv2d: 56×56×16 → 56×56×16 (特征增强)
  ↓
MultiHeadAttentionCompression: 56×56×16 → 28×28×16 (潜在空间)
```

#### 解码器架构
```
潜在空间: 28×28×16
  ↓
AttentionDecompression: 28×28×16 → 56×56×16
  ↓
SelfAttention2D + Conv2d: 56×56×16 → 56×56×16 (特征增强)
  ↓
ConvTranspose2d + LayerNorm: 56×56×16 → 112×112×8
  ↓
ConvTranspose2d + Sigmoid: 112×112×8 → 224×224×3 (重建输出)
```

### 🧠 优化策略

#### ObGD优化器 (参考streaming-drl)
- **在线学习**: 逐帧更新参数，无需批量数据
- **自适应学习率**: 根据梯度变化动态调整
- **内存高效**: 避免存储大量历史数据
- **稳定性保证**: 通过kappa参数控制更新幅度

#### 损失函数设计
```python
# 全局损失 - 关注整体结构  
global_loss = mse_loss + 0.1 * gradient_loss + 0.1 * ssim_loss

# 包含多种损失函数确保重建质量
# - MSE损失: 像素级重建误差
# - 梯度损失: 保持边缘和纹理信息
# - SSIM损失: 结构相似性
```

### 📁 项目结构

```
stream-ae/
├── quick_start.py              # 快速启动脚本
├── streaming_video_autoencoder.py  # 核心模型实现
├── requirements.txt            # 依赖包列表
├── optim.py                   # ObGD优化器实现
├── sparse_init.py             # 稀疏权重初始化
└── README.md                  # 项目文档
```

### 🚀 快速开始

#### 1. 安装依赖
```bash
# 使用uv安装依赖（推荐）
uv pip install -r requirements.txt

# 或使用pip
pip install -r requirements.txt
```

#### 2. 运行演示
```bash
# 使用uv运行
uv run quick_start.py

# 或直接运行
python quick_start.py
```

#### 3. 选择运行模式
- **模式1**: Quick Demo - 完整训练和分析流程 (TensorBoard)
- **模式2**: Live Viewer - 实时可视化监控 (TensorBoard)

#### 4. TensorBoard可视化
```bash
# 启动TensorBoard（在另一个终端）
tensorboard --logdir=runs

# 在浏览器中打开
http://localhost:6006
```

#### 5. 测试TensorBoard集成
```bash
# 运行测试脚本
python test_tensorboard.py

# 查看测试结果
tensorboard --logdir=runs/test_tensorboard
```

### 📊 性能监控

#### TensorBoard可视化功能
- **损失曲线**: 实时监控细节损失、全局损失、MSE损失等
- **图像重建**: 输入图像、重建图像、变化掩码、重建误差
- **特征图**: 编码器和解码器各层的特征图可视化
- **模型参数**: 参数分布和梯度分布监控
- **指标统计**: 变化像素数量、变化比率等

#### 训练指标
- **细节损失**: 局部特征重建质量
- **全局损失**: 整体结构保持程度
- **MSE损失**: 像素级重建误差
- **变化像素**: 帧间变化检测统计

#### 可视化功能
- **TensorBoard**: 专业的深度学习可视化工具
- **实时监控**: 训练过程的实时可视化
- **交互式界面**: 支持缩放、筛选、对比等操作
- **多维度分析**: 从不同角度分析模型性能

### 🔧 技术细节

#### 稀疏初始化
```python
# 90%稀疏度的权重初始化
sparse_init(weight, sparsity=0.9)
```
- 提高模型泛化能力
- 减少过拟合风险
- 加速训练收敛

#### 像素变化检测
```python
# 智能变化检测
change_mask, change_intensity = detector.detect_changes(prev_frame, curr_frame)
```
- 运动区域识别
- 注意力引导学习
- 计算资源优化

### 📈 实验结果

典型训练结果（500,000帧）：
- **平均细节损失**: ~0.0234
- **平均全局损失**: ~0.0156  
- **平均MSE损失**: ~0.0089
- **收敛时间**: ~2-3小时（GPU）

### 🔗 参考项目

本项目的优化器设计参考了：
- [streaming-drl](https://github.com/mohmdelsayed/streaming-drl) - 流式深度强化学习
- ObGD优化器的在线学习思想
- 非平稳数据流的处理策略

### 📝 许可证

本项目采用MIT许可证，详见LICENSE文件。

### 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

### 📧 联系方式

如有问题或建议，请通过GitHub Issues联系我们。
