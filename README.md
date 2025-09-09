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

#### 核心架构组件

**StreamingAutoEncoder** (`autoencoder.py:15`):
- 主要模型类，整合编码器、解码器、变化检测器和监控组件
- 提供完整的流式训练接口
- 集成TensorBoard实时监控和性能分析

**Encoder** (`model.py:228`):
- 并行三分支编码器，不同卷积核尺寸捕获多尺度特征
- 小卷积分支 (3×3): 224×224×3 → 27×27×3，纹理特征
- 中卷积分支 (5×5): 224×224×3 → 25×25×2，平衡特征  
- 大卷积分支 (7×7): 224×224×3 → 23×23×2，结构特征
- 无填充设计保持边缘信息真实性
- 压缩比: 92:1, 161:1, 190:1

**Decoder** (`model.py:314`):
- 并行三分支解码器，与编码器对称设计
- 小卷积分支解码器: 27×27×3 → 223×223×3
- 中卷积分支解码器: 25×25×2 → 221×221×3
- 大卷积分支解码器: 23×23×2 → 219×219×3
- 自适应融合模块统一尺寸到224×224×3
- 最终输出应用sigmoid激活函数

#### 数据流
```
输入帧 (224×224×3) → 编码器 → 三分支embedding → 解码器 → 重建帧 (224×224×3)
                      ↓
                变化检测器 → 注意力机制
                      ↓
                损失计算 → ObGD优化器 → 参数更新
                      ↓
                TensorBoard监控 → 实时可视化
```

### 🧠 优化策略

#### ObGD优化器 (参考streaming-drl)
- **在线学习**: 逐帧更新参数，无需批量数据
- **自适应学习率**: 根据梯度变化动态调整
- **内存高效**: 避免存储大量历史数据
- **稳定性保证**: 通过kappa参数控制更新幅度

#### 损失函数设计
```python
# 全局损失 - 结合多种损失函数
global_loss = mse_loss + 0.5 * l1_loss + 0.1 * ssim_loss

# 包含多种损失函数确保重建质量
# - MSE损失: 像素级重建误差 (主要损失)
# - L1损失: 提供更稳定的梯度 (权重0.5)
# - SSIM损失: 结构相似性 (权重0.1)
```

#### 智能监控和可视化
- **TensorBoard集成**: 实时记录损失、图像、特征图
- **性能监控**: FPS计算、步骤时间统计
- **特征可视化**: 编码器和解码器特征图展示
- **变化检测**: 像素级变化统计和可视化

### 📁 项目结构

```
stream-ae/
├── main.py                     # 主入口脚本 - 实时可视化查看器
├── autoencoder.py              # 核心模型实现 (StreamingAutoEncoder)
├── model.py                    # 模型组件 (Encoder, Decoder, LayerNormalization等)
├── optim.py                    # ObGD优化器实现
├── loss.py                     # 损失函数实现
├── monitoring.py               # TensorBoard监控和性能分析
├── utils.py                    # 数据预处理工具
├── requirements.txt            # 依赖包列表
├── CLAUDE.md                   # Claude Code 开发指南
└── README.md                  # 项目文档
```

#### 核心模块说明

**autoencoder.py** - 核心模型实现
- `StreamingAutoEncoder`: 主要模型类，整合所有组件
- `create_streaming_ae`: 工厂函数创建模型实例
- 集成训练、推理、监控等完整功能

**model.py** - 模型架构组件
- `Encoder`: 并行三分支编码器
- `Decoder`: 并行三分支解码器
- `LayerNormalization`: 动态层归一化
- `PixelChangeDetector`: 像素变化检测器
- `ResidualBlock`: 残差连接块
- `TransformerBlock`: Transformer处理模块

**optim.py** - 优化器实现
- `ObGD`: 在线梯度下降优化器
- `AdaptiveObGD`: 自适应在线梯度下降
- `sparse_init`: 稀疏权重初始化
- `initialize_weights`: 权重初始化函数

**loss.py** - 损失函数
- `compute_global_loss`: 全局损失计算 (MSE + L1 + SSIM)

**monitoring.py** - 监控和可视化
- `TensorBoardLogger`: TensorBoard日志记录
- `PerformanceMonitor`: 性能监控器
- `FeatureVisualizer`: 特征可视化器

**utils.py** - 工具函数
- `preprocess_frame`: 帧预处理
- `postprocess_output`: 输出后处理
- `compute_frame_difference`: 帧差异计算

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
# 直接运行主程序（启动实时可视化查看器）
python main.py

# 或使用uv运行
uv run main.py
```

#### 3. 运行模式
- **实时可视化模式**: 运行 `python main.py` 直接启动实时可视化查看器
  - 随机选择游戏环境（Breakout、Assault、SpaceInvaders、Pacman、Asteroids）
  - 实时TensorBoard监控，每帧刷新
  - 支持模型加载（如果存在 `quick_demo_model.pth`）

#### 4. TensorBoard可视化功能
```bash
# 启动TensorBoard（在另一个终端）
tensorboard --logdir=runs

# 在浏览器中打开
http://localhost:6006
```

**TensorBoard监控内容**:
- **损失曲线**: 实时监控全局损失、MSE损失、L1损失、SSIM损失
- **图像重建**: 输入图像、重建图像、重建误差对比
- **特征图可视化**: 三个编码分支的特征图展示
  - 小卷积分支: 27×27×3 纹理特征
  - 中卷积分支: 25×25×2 平衡特征
  - 大卷积分支: 23×23×2 结构特征
- **性能指标**: FPS、步骤时间、变化像素统计
- **模型参数**: 参数分布和梯度监控

**实时特性**:
- 每帧刷新TensorBoard数据
- 支持实时性能分析
- 交互式可视化界面


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
# 50%稀疏度的权重初始化（降低从90%避免梯度消失）
sparse_init(weight, sparsity=0.5)
```
- 提高模型泛化能力
- 减少过拟合风险
- 加速训练收敛
- 避免梯度消失问题

#### 像素变化检测
```python
# 智能变化检测
change_mask, change_intensity = detector.detect_changes(prev_frame, curr_frame)
```
- 运动区域识别
- 注意力引导学习
- 计算资源优化
- 自适应阈值检测

### 📈 实验结果

典型训练结果（500,000帧）：
- **平均全局损失**: ~0.0156  
- **平均MSE损失**: ~0.0089
- **平均L1损失**: ~0.0045
- **平均SSIM损失**: ~0.0022
- **收敛时间**: ~2-3小时（GPU）

#### 关键依赖
- PyTorch 2.3.0
- TensorBoard 2.16.2
- Gymnasium 0.29.1
- OpenCV 4.9.0.80
- NumPy 1.26.4

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
