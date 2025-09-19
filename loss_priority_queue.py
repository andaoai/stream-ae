"""
损失优先级队列模块
实现基于损失的帧优先级队列，用于选择高损失帧进行批量训练
"""

import torch
import torch.nn.functional as F
from collections import deque
from typing import List, Tuple, Optional
import heapq


class LossPriorityQueue:
    """
    基于损失的优先级队列

    功能：
    1. 维护一个固定大小的队列，存储损失最高的帧
    2. 提供基于损失的优先级排序
    3. 支持批量获取高损失帧
    4. 动态更新帧损失并维护队列排序
    """

    def __init__(self, max_size=24, min_loss_threshold=0.001):
        """
        初始化损失优先级队列

        Args:
            max_size: 队列最大容量
            min_loss_threshold: 最低损失阈值，低于此值的帧不会被加入队列
        """
        self.max_size = max_size
        self.min_loss_threshold = min_loss_threshold
        self.queue = []  # 使用堆实现的优先级队列
        self.frame_data = {}  # 存储帧数据的字典 {frame_id: (frame_tensor, loss)}
        self.frame_counter = 0  # 帧计数器，用于生成唯一ID
        self.loss_history = []  # 损失历史记录

    def add_frame(self, frame: torch.Tensor, loss: float) -> bool:
        """
        添加帧到队列

        Args:
            frame: 帧数据张量
            loss: 帧的损失值

        Returns:
            bool: 是否成功添加到队列
        """
        # 如果损失低于阈值，不加入队列
        if loss < self.min_loss_threshold:
            return False

        # 如果队列已满，检查是否可以替换最低损失的帧
        if len(self.queue) >= self.max_size:
            # 获取队列中最低的损失
            lowest_neg_loss, lowest_frame_id = self.queue[0]
            lowest_loss = -lowest_neg_loss

            # 如果当前帧损失不足以替换队列中的帧，则不加入
            if loss <= lowest_loss:
                return False

        # 创建帧ID并存储帧数据
        frame_id = f"frame_{self.frame_counter}"
        self.frame_counter += 1

        # 存储帧数据
        self.frame_data[frame_id] = (frame.detach().cpu(), loss)

        # 如果队列未满，直接加入
        if len(self.queue) < self.max_size:
            heapq.heappush(self.queue, (-loss, frame_id))  # 使用负损失实现最大堆
            self.loss_history.append(loss)
            return True

        # 如果队列已满，替换最低损失的帧
        else:
            # 移除最低损失的帧
            removed_frame_id = heapq.heappop(self.queue)[1]
            if removed_frame_id in self.frame_data:
                del self.frame_data[removed_frame_id]

            # 添加新帧
            heapq.heappush(self.queue, (-loss, frame_id))
            self.loss_history.append(loss)
            return True

    def update_frame_loss(self, frame_id: str, new_loss: float) -> bool:
        """
        更新帧的损失值并重新排序队列

        Args:
            frame_id: 帧ID
            new_loss: 新的损失值

        Returns:
            bool: 是否成功更新
        """
        if frame_id not in self.frame_data:
            return False

        # 获取原始帧数据
        frame_tensor, _ = self.frame_data[frame_id]

        # 如果新损失低于阈值，从队列中移除
        if new_loss < self.min_loss_threshold:
            # 从队列中移除该帧
            self._remove_frame_from_queue(frame_id)
            del self.frame_data[frame_id]
            return False

        # 更新帧数据
        self.frame_data[frame_id] = (frame_tensor, new_loss)

        # 重建队列以重新排序
        self._rebuild_queue()

        return True

    def get_top_frames(self, num_frames: int) -> List[Tuple[torch.Tensor, float, str]]:
        """
        获取损失最高的若干帧

        Args:
            num_frames: 要获取的帧数量

        Returns:
            List[Tuple[torch.Tensor, float, str]]: (帧数据, 损失, 帧ID) 的列表
        """
        # 获取所有帧并按损失排序
        all_frames = []
        for neg_loss, frame_id in self.queue:
            loss = -neg_loss
            frame_tensor = self.frame_data[frame_id][0]
            all_frames.append((frame_tensor, loss, frame_id))

        # 按损失降序排序
        all_frames.sort(key=lambda x: x[1], reverse=True)

        # 返回前num_frames个帧
        return all_frames[:min(num_frames, len(all_frames))]

    def get_batch(self, batch_size: int, current_frame: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        获取批量训练数据

        Args:
            batch_size: 批量大小
            current_frame: 当前最新帧

        Returns:
            Tuple[torch.Tensor, List[str]]: (批量数据张量, 帧ID列表)
        """
        # 获取高损失帧
        top_frames = self.get_top_frames(batch_size - 1)

        # 构建批量数据
        batch_frames = []
        frame_ids = []

        # 添加当前帧作为第一个元素
        batch_frames.append(current_frame)
        frame_ids.append("current_frame")

        # 添加高损失帧，确保设备一致
        current_device = current_frame.device
        for frame_tensor, loss, frame_id in top_frames:
            # 将帧数据移动到当前帧的设备
            frame_tensor = frame_tensor.to(current_device)
            batch_frames.append(frame_tensor)
            frame_ids.append(frame_id)

        # 堆叠为批量张量
        batch_tensor = torch.stack(batch_frames, dim=0)

        return batch_tensor, frame_ids

    def get_min_loss(self) -> float:
        """
        获取队列中的最低损失值

        Returns:
            float: 最低损失值，如果队列为空返回0
        """
        if not self.queue:
            return 0.0

        # 最小损失在堆顶
        return -self.queue[0][0]

    def get_max_loss(self) -> float:
        """
        获取队列中的最高损失值

        Returns:
            float: 最高损失值，如果队列为空返回0
        """
        if not self.queue:
            return 0.0

        # 需要遍历找到最大损失
        max_loss = 0.0
        for neg_loss, _ in self.queue:
            max_loss = max(max_loss, -neg_loss)

        return max_loss

    def get_average_loss(self) -> float:
        """
        获取队列中的平均损失值

        Returns:
            float: 平均损失值，如果队列为空返回0
        """
        if not self.queue:
            return 0.0

        total_loss = sum(-neg_loss for neg_loss, _ in self.queue)
        return total_loss / len(self.queue)

    def get_queue_size(self) -> int:
        """
        获取当前队列大小

        Returns:
            int: 队列大小
        """
        return len(self.queue)

    def is_full(self) -> bool:
        """
        检查队列是否已满

        Returns:
            bool: 是否已满
        """
        return len(self.queue) >= self.max_size

    def clear(self):
        """
        清空队列
        """
        self.queue.clear()
        self.frame_data.clear()
        self.loss_history.clear()

    def get_stats(self) -> dict:
        """
        获取队列统计信息

        Returns:
            dict: 包含队列统计信息的字典
        """
        return {
            'queue_size': len(self.queue),
            'max_size': self.max_size,
            'min_loss': self.get_min_loss(),
            'max_loss': self.get_max_loss(),
            'avg_loss': self.get_average_loss(),
            'total_frames_processed': self.frame_counter
        }

    def _remove_frame_from_queue(self, frame_id: str):
        """
        从队列中移除指定帧

        Args:
            frame_id: 要移除的帧ID
        """
        # 找到并移除指定帧
        new_queue = []
        for neg_loss, fid in self.queue:
            if fid != frame_id:
                new_queue.append((neg_loss, fid))

        # 重建堆
        self.queue = new_queue
        heapq.heapify(self.queue)

    def _rebuild_queue(self):
        """
        重建队列以重新排序
        """
        # 重新构建队列
        new_queue = []
        for frame_id, (frame_tensor, loss) in self.frame_data.items():
            new_queue.append((-loss, frame_id))

        # 重建堆
        self.queue = new_queue
        heapq.heapify(self.queue)


class BatchLossTracker:
    """
    批量损失追踪器

    功能：
    1. 追踪批量中每个帧的损失
    2. 提供批量损失统计
    3. 更新优先级队列中的帧损失
    """

    def __init__(self):
        """
        初始化批量损失追踪器
        """
        self.batch_losses = {}  # {frame_id: loss}
        self.batch_stats = {
            'min_loss': float('inf'),
            'max_loss': 0.0,
            'avg_loss': 0.0,
            'total_loss': 0.0
        }

    def update_batch_loss(self, frame_id: str, loss: float):
        """
        更新批量中帧的损失

        Args:
            frame_id: 帧ID
            loss: 损失值
        """
        self.batch_losses[frame_id] = loss

        # 更新统计信息
        self._update_stats()

    def get_batch_loss(self, frame_id: str) -> Optional[float]:
        """
        获取批量中指定帧的损失

        Args:
            frame_id: 帧ID

        Returns:
            Optional[float]: 损失值，如果不存在返回None
        """
        return self.batch_losses.get(frame_id)

    def get_all_losses(self) -> dict:
        """
        获取所有帧的损失

        Returns:
            dict: {frame_id: loss} 字典
        """
        return self.batch_losses.copy()

    def get_stats(self) -> dict:
        """
        获取批量损失统计信息

        Returns:
            dict: 包含统计信息的字典
        """
        return self.batch_stats.copy()

    def clear(self):
        """
        清空批量损失数据
        """
        self.batch_losses.clear()
        self.batch_stats = {
            'min_loss': float('inf'),
            'max_loss': 0.0,
            'avg_loss': 0.0,
            'total_loss': 0.0
        }

    def _update_stats(self):
        """
        更新统计信息
        """
        if not self.batch_losses:
            self.batch_stats = {
                'min_loss': float('inf'),
                'max_loss': 0.0,
                'avg_loss': 0.0,
                'total_loss': 0.0
            }
            return

        losses = list(self.batch_losses.values())
        self.batch_stats = {
            'min_loss': min(losses),
            'max_loss': max(losses),
            'avg_loss': sum(losses) / len(losses),
            'total_loss': sum(losses)
        }