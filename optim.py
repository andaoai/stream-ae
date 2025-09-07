"""
优化器模块

该模块包含了流式视频自编码器中使用的优化器相关组件。
包括ObGD优化器和稀疏初始化函数。
"""

import torch
import torch.nn as nn
import math


class ObGD(torch.optim.Optimizer):
    """
    在线梯度下降优化器 (Online Gradient Descent)
    
    该优化器专为流式学习设计，参考了streaming-drl项目中的在线优化思想。
    主要特性：
    1. 动量更新 - 使用衰减因子平滑梯度
    2. 自适应学习率 - 根据梯度历史调整学习率
    3. 内存高效 - 不需要存储完整的梯度历史
    4. 适合非平稳数据流 - 能够适应数据分布的变化
    
    参数：
        params: 模型参数
        lr: 学习率
        gamma: 动量衰减因子
        lamda: 梯度平衡参数
        kappa: 稳定性参数
    """
    
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super(ObGD, self).__init__(params, defaults)
        
        # 为每个参数初始化动量缓冲区
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['momentum'] = torch.zeros_like(p.data)
    
    def step(self, closure=None):
        """
        执行一步参数更新
        
        Args:
            closure: 闭包函数，用于重新计算损失
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # 获取参数
                lr = group['lr']
                gamma = group['gamma']
                lamda = group['lamda']
                kappa = group['kappa']
                
                # 更新动量
                state['momentum'].mul_(gamma).add_(grad, alpha=1 - gamma)
                
                # 计算自适应学习率
                grad_norm = torch.norm(grad)
                if grad_norm > 0:
                    adaptive_lr = lr / (1 + kappa * grad_norm.item())
                else:
                    adaptive_lr = lr
                
                # 参数更新
                p.data.add_(state['momentum'], alpha=-adaptive_lr * lamda)
        
        return loss


class AdaptiveObGD(torch.optim.Optimizer):
    """
    自适应在线梯度下降优化器
    
    基于ObGD的改进版本，添加了自适应学习率调整机制。
    结合了Adam风格的二阶矩估计和ObGD的在线学习特性。
    """
    
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, beta2=beta2, eps=eps)
        self.counter = 0
        super(AdaptiveObGD, self).__init__(params, defaults)
    
    def step(self, delta, reset=False):
        """
        执行一步参数更新
        
        Args:
            delta: 损失变化量
            reset: 是否重置资格迹
        """
        z_sum = 0.0
        self.counter += 1
        
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                
                # 初始化状态变量
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                e, v = state["eligibility_trace"], state["v"]
                
                # 更新资格迹
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)

                # 更新二阶矩估计
                v.mul_(group["beta2"]).addcmul_(delta*e, delta*e, value=1.0 - group["beta2"])
                v_hat = v / (1.0 - group["beta2"] ** self.counter)
                
                # 计算自适应梯度
                z_sum += (e / (v_hat + group["eps"]).sqrt()).abs().sum().item()

        # 计算自适应步长
        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        if dot_product > 1:
            step_size = group["lr"] / dot_product
        else:
            step_size = group["lr"]

        # 更新参数
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                v, e = state["v"], state["eligibility_trace"]
                v_hat = v / (1.0 - group["beta2"] ** self.counter)
                
                # 使用自适应步长更新参数
                p.data.addcdiv_(delta * e, (v_hat + group["eps"]).sqrt(), value=-step_size)
                
                if reset:
                    e.zero_()


def sparse_init(tensor, sparsity=0.9):
    """
    稀疏权重初始化函数
    
    该函数实现了稀疏权重初始化，是streaming-drl项目中
    提高模型泛化能力的重要技术。
    
    工作原理：
    1. 随机选择要保留的权重位置
    2. 使用正态分布初始化这些权重
    3. 将其他权重置零
    
    优势：
    1. 减少过拟合 - 通过稀疏连接降低模型复杂度
    2. 提高泛化 - 强制模型学习更鲁棒的特征表示
    3. 加速训练 - 减少需要更新的参数数量
    4. 内存效率 - 降低模型的内存占用
    
    Args:
        tensor (torch.Tensor): 待初始化的张量
        sparsity (float): 稀疏度，范围[0,1]，表示被置零的权重比例
    
    Note:
        - 稀疏度0.9表示90%的权重被置零
        - 使用标准正态分布初始化非零权重
        - 适用于Linear和Conv2d层的权重初始化
    """
    with torch.no_grad():
        # 计算要保留的权重数量
        num_elements = tensor.numel()
        num_nonzero = int(num_elements * (1 - sparsity))
        
        # 随机选择要保留的位置
        indices = torch.randperm(num_elements, device=tensor.device)[:num_nonzero]
        
        # 将所有权重置零
        tensor.zero_()
        
        # 使用标准正态分布初始化选中的权重
        tensor.view(-1)[indices] = torch.randn(num_nonzero, device=tensor.device)


def initialize_weights(m):
    """
    稀疏权重初始化函数
    
    对线性层和卷积层应用稀疏初始化，这是streaming-drl项目中
    提高模型泛化能力的重要技术。
    
    Args:
        m (nn.Module): 待初始化的网络层
    
    Note:
        - 稀疏度设置为0.5，即50%的权重被置零
        - 偏置项统一初始化为0
        - 仅对Linear和Conv2d层进行稀疏初始化
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        sparse_init(m.weight, sparsity=0.5)  # 降低稀疏度从90%到50%，避免梯度消失
        if m.bias is not None:
            m.bias.data.fill_(0.0)  # 偏置置零