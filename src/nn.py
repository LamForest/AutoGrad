from tensor import Tensor
import numpy as np
from ops import matmul, mean, pow, sub

# 线性层实现（无偏置）
class Linear:
    def __init__(self, in_features, out_features):
        # He初始化
        std = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features).astype(np.float32) * std.astype(np.float32),
            requires_grad=True
        )
        # print(f"Linear {self.weight.data.dtype=}")
    
    def __call__(self, x):
        return matmul(x, self.weight)
    
def mse_loss(pred, target):
    """
    均方误差损失函数:
    loss = mean((pred - target) ** 2)
    """
    return mean(
        pow(
            sub(pred, target),
            2
        )
    )