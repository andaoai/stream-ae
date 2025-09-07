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
import gymnasium as gym
from streaming_video_autoencoder import StreamingAutoEncoder, preprocess_frame
import os
from datetime import datetime



def live_viewer():
    """
    实时可视化查看器 - TensorBoard版本

    该函数提供了流式视频自编码器的实时可视化界面，专注于TensorBoard监控：
    1. 实时监控 - 通过TensorBoard显示当前帧、重建输出和差异图
    2. 损失追踪 - 实时记录全局损失和MSE损失到TensorBoard
    3. 特征可视化 - 记录三个卷积分支的特征图到TensorBoard
    4. 性能分析 - 通过TensorBoard检测重建质量和收敛情况

    TensorBoard功能：
    - 实时损失曲线和指标监控
    - 图像重建质量可视化
    - 三个卷积分支输出特征图:
      * 小卷积(3×3): 纹理特征 - 27×27×4
      * 中卷积(5×5): 平衡特征 - 25×25×4
      * 大卷积(7×7): 结构特征 - 23×23×4
    - Embedding统计信息监控
    - 模型参数分布监控

    控制：
    - Ctrl+C: 停止运行
    - TensorBoard: 在浏览器中查看 http://localhost:6006
    """
    print("Real-time Live Viewer with TensorBoard")
    print("TensorBoard将在浏览器中显示实时监控信息")
    print("启动TensorBoard: tensorboard --logdir=runs")

    # Load model with TensorBoard enabled
    model = StreamingAutoEncoder(
        input_channels=3, 
        base_channels=8, 
        latent_channels=4, 
        lr=0.0001, 
        debug_vis=True,
        use_tensorboard=True,
        log_dir=f"runs/live_viewer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    try:
        model.load_state_dict(torch.load('quick_demo_model.pth'))
        print("Model loaded")
    except:
        print("Using untrained model")

    # Setup environment
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    obs, _ = env.reset()

    try:
        print("Press Ctrl+C to stop")
        print("在另一个终端运行: tensorboard --logdir=runs")
        print("然后在浏览器中打开: http://localhost:6006")
        i = 0
        while True:  # 持续运行
            # Process current frame
            curr_frame = preprocess_frame(obs)
            results = model.update_params(curr_frame, debug=(i % 100 == 0))

            # Environment step
            action = env.action_space.sample()
            obs, _, done, truncated, _ = env.step(action)

            if done or truncated:
                obs, _ = env.reset()
                model.prev_frame = None

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
    主函数 - 流式视频自编码器实时可视化监控

    直接启动实时可视化查看器，提供：
    - 实时监控 - 通过TensorBoard显示当前帧、重建输出和差异图
    - 损失追踪 - 实时记录损失到TensorBoard
    - 特征可视化 - 记录三个卷积分支的特征图
    - 性能分析 - 实时检测重建质量

    TensorBoard功能：
    - 实时损失曲线和指标监控
    - 图像重建质量可视化
    - 三个卷积分支输出特征图:
      * 小卷积(3×3): 纹理特征 - 27×27×4
      * 中卷积(5×5): 平衡特征 - 25×25×4
      * 大卷积(7×7): 结构特征 - 23×23×4
    - Embedding统计信息监控
    - 模型参数分布监控
    """
    print("Streaming Video Autoencoder - Live Viewer")
    print("基于streaming-drl项目的ObGD优化器实现")
    print("=" * 50)
    print("启动实时可视化监控...")
    print("在另一个终端运行: tensorboard --logdir=runs")
    print("然后在浏览器中打开: http://localhost:6006")
    print("=" * 50)
    
    live_viewer()

if __name__ == "__main__":
    main()
