from nn import Linear, mse_loss
from optim import SGD

from autograd import backward, viz
from tensor import Tensor

import numpy as np
from dataclasses import dataclass
from ops import add, mul, pow, relu, matmul, mean, sub


@dataclass
class MLPConfig:
    input_size = 784
    hidden_size = 1024
    output_size = 10

class MLP:
    """三层全连接网络（无偏置）"""
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size)
        self.fc3 = Linear(hidden_size, hidden_size)
        self.fc4 = Linear(hidden_size, output_size)
    
    def __call__(self, x):
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        x = relu(x)
        x = self.fc3(x)
        x = relu(x)
        x = self.fc4(x)
        return x