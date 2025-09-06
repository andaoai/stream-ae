"""
流式视频自编码器快速启动脚本

本脚本实现了一个流式视频自编码器的快速演示和测试环境。
主要特性：
- 使用ObGD（Online Gradient Descent）优化器进行在线学习
- 双损失函数设计：细节损失和全局损失
- 支持实时视频流处理和可视化
- 基于Gymnasium环境进行测试

优化器设计参考：
https://github.com/mohmdelsayed/streaming-drl
该项目提供了流式深度强化学习的优化器实现，我们借鉴了其在线梯度下降的思想。

作者：流式AI团队
版本：1.0
日期：2024
"""

import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from streaming_video_autoencoder import StreamingAutoEncoder, preprocess_frame
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

def quick_demo():
    """
    流式视频自编码器快速演示

    该函数演示了流式视频自编码器的完整工作流程，包括：
    1. 模型初始化 - 创建带有ObGD优化器的流式自编码器
    2. 环境设置 - 初始化Gymnasium环境（优先使用Atari Breakout）
    3. 在线训练 - 使用流式数据进行实时训练
    4. 结果可视化 - 生成训练过程的详细图表
    5. 性能分析 - 计算收敛性和改善指标
    6. 模型保存 - 保存训练好的模型权重

    优化器特性（参考streaming-drl项目）：
    - ObGD（Online Gradient Descent）：适用于流式数据的在线学习
    - 双损失函数：细节损失关注局部特征，全局损失关注整体结构
    - 自适应学习率：根据梯度变化动态调整学习步长
    - 内存效率：避免存储历史数据，适合长时间运行

    Returns:
        tuple: (model, losses) 包含训练好的模型和损失历史
            - model (StreamingAutoEncoder): 训练完成的自编码器模型
            - losses (dict): 包含各类损失值的字典
                - 'detail': 细节损失历史
                - 'global': 全局损失历史
                - 'mse': MSE损失历史
                - 'changed_pixels': 变化像素数量历史
    """
    print("流式视频自编码器快速演示")
    print("=" * 50)
    
    # 1. 创建模型
    print("创建流式自编码器模型...")
    model = StreamingAutoEncoder(
        input_channels=3,        # RGB图像输入通道数
        base_channels=8,         # 编码器起始通道数（轻量化设计）
        latent_channels=16,      # 潜在空间维度
        lr=0.001,               # ObGD优化器学习率（参考streaming-drl调优）
        gamma=0.99,             # 动量衰减因子，用于梯度平滑
        lamda=0.8,              # 损失函数权重平衡参数
        kappa=1.5,              # 损失稳定性参数
        debug_vis=True,         # 启用调试可视化
        use_tensorboard=True,   # 启用TensorBoard日志记录
        log_dir="runs/quick_demo"  # 指定日志目录
    )

    print(f"模型创建成功！参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 创建环境
    print("\n创建Gym环境...")
    try:
        env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
        obs, _ = env.reset()
        print(f"环境创建成功！观察空间: {obs.shape}")
    except:
        print("Atari环境不可用，使用CartPole替代...")
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        obs, _ = env.reset()
        print(f"CartPole环境创建成功！观察空间: {obs.shape}")
    
    # 3. 快速训练 - 流式在线学习过程
    print("\n开始快速训练...")
    num_frames = 500000  # 训练帧数，足够观察收敛行为
    losses = {'global': [], 'mse': [], 'changed_pixels': []}

    for frame_idx in range(num_frames):
        # 预处理帧：标准化到[0,1]范围，调整维度为[1,C,H,W]
        curr_frame = preprocess_frame(obs)

        # 核心：使用ObGD优化器进行在线参数更新
        # 这里实现了streaming-drl项目中的在线学习思想：
        # - 无需存储历史数据
        # - 实时计算梯度并更新参数
        # - 双损失函数同时优化局部和全局特征
        results = model.update_params(curr_frame, debug=(frame_idx % 10 == 0))

        # 记录训练指标用于后续分析
        losses['global'].append(results['global_loss'])      # 全局损失（整体结构）
        losses['mse'].append(results['mse_loss'])            # 重建误差
        losses['changed_pixels'].append(results['changed_pixels'])  # 变化检测

        # 环境交互：随机动作策略（用于数据多样性）
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # 处理episode结束：重置环境和模型状态
        if terminated or truncated:
            obs, _ = env.reset()
            model.prev_frame = None      # 清除前一帧缓存
            model.prev_embedding = None  # 清除前一帧编码

        # 训练进度监控
        if frame_idx % 1000 == 0:
            print(f"帧 {frame_idx+1}/{num_frames} - "
                  f"Global: {results['global_loss']:.1f}, "
                  f"MSE: {results['mse_loss']:.1f}")
    
    env.close()
    
    # 关闭TensorBoard写入器
    model.close_tensorboard()
    
    # 4. 结果可视化
    print("\n生成结果可视化...")
    print("TensorBoard日志已保存，可以使用以下命令查看:")
    print("tensorboard --logdir=runs/quick_demo")
    print("然后在浏览器中打开 http://localhost:6006")
    
    # 仍然保留matplotlib图表作为补充
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('流式视频自编码器训练结果 (TensorBoard提供更详细的可视化)', fontsize=16)
    
    frames = range(num_frames)
    
    # 全局损失趋势
    axes[0, 0].plot(frames, losses['global'], 'r-', label='全局损失', linewidth=2)
    axes[0, 0].set_title('全局损失趋势')
    axes[0, 0].set_xlabel('帧数')
    axes[0, 0].set_ylabel('损失值')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE趋势
    axes[0, 1].plot(frames, losses['mse'], 'g-', linewidth=2)
    axes[0, 1].set_title('MSE损失趋势')
    axes[0, 1].set_xlabel('帧数')
    axes[0, 1].set_ylabel('MSE损失')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 变化像素统计
    axes[1, 0].plot(frames, losses['changed_pixels'], 'm-', linewidth=2)
    axes[1, 0].set_title('变化像素数量')
    axes[1, 0].set_xlabel('帧数')
    axes[1, 0].set_ylabel('像素数')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 损失对比
    axes[1, 1].plot(frames, losses['global'], 'r-', label='全局损失', linewidth=2)
    axes[1, 1].plot(frames, losses['mse'], 'g-', label='MSE损失', linewidth=2)
    axes[1, 1].set_title('损失对比')
    axes[1, 1].set_xlabel('帧数')
    axes[1, 1].set_ylabel('损失值')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_demo_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 5. 结果分析
    print("\n训练结果分析:")
    print(f"   平均全局损失: {np.mean(losses['global']):.6f}")
    print(f"   平均MSE损失: {np.mean(losses['mse']):.6f}")
    print(f"   平均变化像素: {np.mean(losses['changed_pixels']):.0f}")
    
    # 收敛性分析
    if len(losses['global']) > 10:
        early_global = np.mean(losses['global'][:10])
        late_global = np.mean(losses['global'][-10:])
        global_improvement = (early_global - late_global) / early_global * 100
        
        early_mse = np.mean(losses['mse'][:10])
        late_mse = np.mean(losses['mse'][-10:])
        mse_improvement = (early_mse - late_mse) / early_mse * 100
        
        print(f"   全局损失改善: {global_improvement:.1f}%")
        print(f"   MSE损失改善: {mse_improvement:.1f}%")
    
    # 6. 保存模型
    print("\n保存模型...")
    torch.save(model.state_dict(), 'quick_demo_model.pth')
    print("模型已保存到 quick_demo_model.pth")

    print("\n快速演示完成！")
    print("生成文件:")
    print("   - quick_demo_results.png: 训练结果可视化")
    print("   - quick_demo_model.pth: 训练好的模型")
    
    return model, losses

def test_model_inference():
    """
    测试模型推理能力

    该函数用于测试训练完成的流式自编码器的推理性能，包括：
    1. 模型加载 - 尝试加载预训练权重或使用随机初始化
    2. 推理测试 - 使用随机输入测试重建能力
    3. 性能评估 - 计算重建误差（MSE）
    4. 结果可视化 - 显示原始输入、重建输出和embedding表示

    该测试验证了ObGD优化器训练的模型是否具备：
    - 良好的重建能力（低MSE误差）
    - 有意义的潜在表示（embedding可视化）
    - 稳定的推理性能（无梯度爆炸/消失）

    Returns:
        float: 重建误差（MSE），用于量化模型性能
    """
    print("\n测试模型推理能力...")

    # 加载模型 - 使用与训练时相同的架构
    model = StreamingAutoEncoder(
        input_channels=3,        # RGB输入
        base_channels=64,        # 推理时使用更大的通道数
        latent_channels=16       # 保持与训练时一致的潜在维度
    )

    try:
        model.load_state_dict(torch.load('quick_demo_model.pth'))
        print("模型加载成功")
    except:
        print("未找到预训练模型，使用随机初始化")
    
    model.eval()
    
    # 创建测试输入
    test_input = torch.rand(1, 3, 64, 64)
    
    with torch.no_grad():
        reconstruction, embedding = model(test_input)
    
    mse_error = F.mse_loss(test_input, reconstruction).item()
    
    print(f"推理结果:")
    print(f"   输入形状: {test_input.shape}")
    print(f"   重建形状: {reconstruction.shape}")
    print(f"   Embedding形状: {embedding.shape}")
    print(f"   重建误差: {mse_error:.6f}")
    
    # 可视化推理结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(test_input[0].permute(1, 2, 0))
    axes[0].set_title('原始输入')
    axes[0].axis('off')
    
    axes[1].imshow(reconstruction[0].permute(1, 2, 0))
    axes[1].set_title(f'重建输出\nMSE: {mse_error:.4f}')
    axes[1].axis('off')
    
    # Embedding可视化
    emb_2d = embedding[0].reshape(8, 4)  # 32维重塑为8x4
    im = axes[2].imshow(emb_2d, cmap='viridis')
    axes[2].set_title('Embedding表示')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('inference_test.png', dpi=150)
    plt.show()
    
    return mse_error

def live_viewer():
    """
    实时可视化查看器 - TensorBoard版本

    该函数提供了流式视频自编码器的实时可视化界面，使用TensorBoard进行监控：
    1. 实时监控 - 通过TensorBoard显示当前帧、重建输出和差异图
    2. 损失追踪 - 实时记录细节损失、全局损失和MSE损失到TensorBoard
    3. 特征可视化 - 记录编码器和解码器各层的特征图到TensorBoard
    4. 性能分析 - 通过TensorBoard检测细节丢失和重建质量

    TensorBoard功能：
    - 实时损失曲线和指标监控
    - 图像重建质量可视化
    - 特征图层级分析
    - 模型参数分布监控
    - 交互式调试界面

    控制：
    - Ctrl+C: 停止运行
    - TensorBoard: 在浏览器中查看 http://localhost:6006
    """
    print("Real-time Live Viewer with TensorBoard")
    print("TensorBoard将在浏览器中显示实时监控信息")
    print("启动TensorBoard: tensorboard --logdir=runs/live_viewer")

    # Load model with TensorBoard enabled
    model = StreamingAutoEncoder(
        input_channels=3, 
        base_channels=64, 
        latent_channels=16, 
        lr=0.0001, 
        debug_vis=True,
        use_tensorboard=True,
        log_dir="runs/live_viewer"
    )
    try:
        model.load_state_dict(torch.load('quick_demo_model.pth'))
        print("Model loaded")
    except:
        print("Using untrained model")

    # Setup environment
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    obs, _ = env.reset()

    prev_frame = None
    frame_count = 0

    try:
        print("Press Ctrl+C to stop")
        print("在另一个终端运行: tensorboard --logdir=runs/live_viewer")
        print("然后在浏览器中打开: http://localhost:6006")
        i = 0
        while True:  # 持续运行
            # Process current frame
            curr_frame = preprocess_frame(obs)
            results = model.update_params(curr_frame, debug=(i % 100 == 0))

            # Environment step
            prev_frame = curr_frame.clone()
            action = env.action_space.sample()
            obs, _, done, truncated, _ = env.step(action)

            if done or truncated:
                obs, _ = env.reset()
                model.prev_frame = None

            frame_count += 1
            i += 1

            # 定期输出统计信息
            if i % 100 == 0:
                print(f"Frame {i}: Global={results['global_loss']:.3f}, MSE={results['mse_loss']:.3f}")
                print(f"  Changed Pixels={results['changed_pixels']:.0f}")

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        env.close()
        model.close_tensorboard()
        print("Live viewer finished")

def main():
    """
    主函数 - 流式视频自编码器演示程序入口

    提供两种运行模式：
    1. Quick Demo (默认) - 完整的训练、测试和分析流程
       - 自动训练模型500,000帧
       - 生成详细的性能分析报告
       - 保存训练结果和可视化图表
       - 适合批量实验和性能评估

    2. Live Viewer - 实时可视化监控界面
       - 实时显示训练过程
       - 多维度性能监控
       - 交互式调试功能
       - 适合开发调试和演示展示

    技术特性：
    - 基于streaming-drl的ObGD优化器
    - 双损失函数设计（细节+全局）
    - 内存高效的流式处理
    - 实时性能监控和可视化
    """
    print("Streaming Video Autoencoder with TensorBoard")
    print("基于streaming-drl项目的ObGD优化器实现")
    print("=" * 50)
    print("1. Quick Demo  - 完整训练和分析流程 (TensorBoard)")
    print("2. Live Viewer - 实时可视化监控 (TensorBoard)")
    print("=" * 50)
    print("TensorBoard功能:")
    print("  - 实时损失曲线和指标监控")
    print("  - 图像重建质量可视化")
    print("  - 特征图层级分析")
    print("  - 模型参数分布监控")
    print("=" * 50)

    choice = input("请选择运行模式 (1 或 2，默认1): ").strip()

    if choice == "2":
        live_viewer()
    else:
        # 运行快速演示
        print("启动快速演示模式...")
        model, losses = quick_demo()
        test_model_inference()

if __name__ == "__main__":
    main()
