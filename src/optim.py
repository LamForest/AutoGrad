import numpy as np

class SGD:
    """随机梯度下降优化器"""
    def __init__(self, parameters, lr=0.0001):
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        """更新参数"""
        for param in self.parameters:
            if param.grad is not None:
                # print(f"SGD {param.grad.shape=}, ", param.grad.data)
                param.data -= self.lr * param.grad.data
    
    def zero_grad(self):
        """清除梯度"""
        for param in self.parameters:
            param.grad = None
