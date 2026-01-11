import sys

import numpy as np
from typing import List, Any, Optional, Tuple

import time
import numpy as np

from ops import add, mul, pow, relu, matmul, mean, sub
from nn import Linear, mse_loss
from optim import SGD

from autograd import backward, viz
from tensor import Tensor
from utils import load_mnist, plot_training_history, visualize_digit_predictions
from model import MLP, MLPConfig




def accuracy(outputs, labels):
    """计算分类准确率"""
    preds = np.argmax(outputs.data, axis=1)
    return np.mean(preds == labels)

def one_hot_encode(labels, num_classes=10):
    """手动实现 one-hot 编码"""
    batch_size = len(labels)
    encoded = np.zeros((batch_size, num_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        encoded[i, label] = 1.0
    return encoded


# ====================== 修改后的训练函数 ======================
def train_mnist(mlp_config: MLPConfig, num_epochs: int):
    """训练MNIST分类模型"""
    # 加载数据集
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # 创建模型
    print(f"init MLP")
    np.random.seed(42)
    model = MLP(mlp_config.input_size, mlp_config.hidden_size, mlp_config.output_size)
    
    # 获取所有可训练参数
    parameters = []
    for layer in [model.fc1, model.fc2, model.fc3, model.fc4]:
        parameters.append(layer.weight)
    
    # 创建优化器
    optimizer = SGD(parameters, lr=0.005)
    
    # 训练参数
    batch_size = 64
    
    num_batches = train_images.shape[0] // batch_size
    if num_batches * batch_size < train_images.shape[0]:
        num_batches += 1
    
    # 用于记录训练历史的变量
    history = {
        'epoch': [],
        'step': [],
        'batch_loss': [],
        'batch_acc': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 训练循环
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        start_time = time.time()
        
        # 随机打乱数据
        indices = np.arange(train_images.shape[0])
        np.random.shuffle(indices)
        shuffled_images = train_images[indices]
        shuffled_labels = train_labels[indices]
        
        for i in range(num_batches):
            # 准备批次数据
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, train_images.shape[0])

            batch_images = shuffled_images[start_idx:end_idx]
            batch_labels = shuffled_labels[start_idx:end_idx]
            batch_targets = one_hot_encode(batch_labels)
            
            inputs = Tensor(batch_images)
            targets = Tensor(batch_targets)

            # 前向传播
            outputs = model.__call__(inputs)
            
            # 计算损失
            diff = sub(outputs, targets)
            loss = mean(pow(diff, 2))

            # 计算平均损失用于记录
            avg_loss = np.mean(loss.data)
            epoch_loss += avg_loss
            
            # 计算准确率
            batch_acc = accuracy(outputs, batch_labels)
            epoch_acc += batch_acc
            
            # 记录每个batch的指标
            current_step = epoch * num_batches + i
            history['step'].append(current_step)
            history['batch_loss'].append(avg_loss)
            history['batch_acc'].append(batch_acc)
            
            # 反向传播
            optimizer.zero_grad()
            backward(loss)
            
            # 更新参数
            optimizer.step()
            
            # 每100批次打印一次
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{num_batches}], Loss: {avg_loss:.4f}, Acc: {batch_acc:.4f}')
        
        # 计算epoch平均指标
        epoch_loss /= num_batches
        epoch_acc /= num_batches
        epoch_time = time.time() - start_time
        
        # 记录epoch指标
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # ====================== 新增: 每个epoch后验证 ======================
        val_loss, val_acc = evaluate(model, test_images, test_labels, batch_size)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印epoch统计
        print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # 测试模型
    test_loss, test_acc = evaluate(model, test_images, test_labels, batch_size)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    # ====================== 新增: 绘制训练历史 ======================
    plot_training_history(history)
    
    # ====================== 新增: 可视化数字预测 ======================
    visualize_digit_predictions(model, test_images, test_labels)

def evaluate(model, images, labels, batch_size=64):
    """评估模型性能"""
    total_loss = 0.0
    total_acc = 0.0
    num_batches = images.shape[0] // batch_size
    
    for i in range(num_batches):
        # 准备批次数据
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_images = images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        batch_targets = one_hot_encode(batch_labels)
        
        # 转换为Tensor
        inputs = Tensor(batch_images)
        targets = Tensor(batch_targets)

        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        diff = sub(outputs, targets)
        loss = mean(pow(diff, 2))
        
        # 计算指标
        batch_loss = np.mean(loss.data)
        batch_acc = accuracy(outputs, batch_labels)
        
        total_loss += batch_loss
        total_acc += batch_acc
    
    return total_loss / num_batches, total_acc / num_batches



if __name__ == "__main__":
    num_epochs = int(sys.argv[1])
    print(f"=======Config============")
    print(f"Epoch = {num_epochs}")
    print(f"===================")

    train_mnist(MLPConfig(), num_epochs)