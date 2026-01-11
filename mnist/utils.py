import matplotlib.pyplot as plt
import numpy as np
from typing import List, Any, Optional, Tuple
import struct
import gzip
import os
import zipfile
import numpy as np
import shutil
import subprocess
from tensor import Tensor

def download_mnist():
    """下载并解压MNIST数据集（从gitee仓库）"""
    # 创建数据目录
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 设置路径
    zip_url = "https://gitee.com/aczz/mnist/repository/archive/master.zip"
    zip_path = os.path.join(data_dir, "mnist_master.zip")
    extract_dir = os.path.join(data_dir, "mnist-master")
    
    # 检查是否需要下载
    required_files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    # 检查文件是否已存在
    files_exist = all(os.path.exists(os.path.join(extract_dir, f)) for f in required_files)
    
    if not files_exist:
        print("Downloading MNIST dataset from gitee using wget...")
        
        # 使用wget下载ZIP文件
        try:
            subprocess.run(['wget', '-O', zip_path, zip_url], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # 如果wget不可用，则使用urllib作为备选方案
            print("wget not found, falling back to urllib")
            import urllib.request
            urllib.request.urlretrieve(zip_url, zip_path)
        
        # 解压ZIP文件
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # 清理ZIP文件
        os.remove(zip_path)
        print("Download and extraction completed.")
    else:
        print("MNIST files already exist. Skipping download.")

def load_mnist():
    """加载MNIST数据集"""
    download_mnist()  # 确保数据已下载
    
    # 设置解压后的文件路径
    base_path = 'data/mnist-master'
    
    # 加载训练图像
    with gzip.open(os.path.join(base_path, 'train-images-idx3-ubyte.gz'), 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        train_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    
    # 加载训练标签
    with gzip.open(os.path.join(base_path, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        train_labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    # 加载测试图像
    with gzip.open(os.path.join(base_path, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        test_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    
    # 加载测试标签
    with gzip.open(os.path.join(base_path, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        test_labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    # 归一化图像数据 [0, 255] -> [0, 1]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    return train_images, train_labels, test_images, test_labels

# ====================== 新增功能1: 训练过程可视化 ======================
def plot_training_history(history):
    """绘制训练历史图表"""
    plt.figure(figsize=(16, 12))
    
    # 图1: Batch指标 (step为单位)
    plt.subplot(2, 1, 1)
    
    # 创建双Y轴
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # 绘制batch损失和准确率
    line1, = ax1.plot(history['step'], history['batch_loss'], 'g-', alpha=0.7, label='Batch Loss')
    line2, = ax2.plot(history['step'], history['batch_acc'], 'b-', alpha=0.7, label='Batch Accuracy')
    
    # 设置标签和标题
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Loss', color='g', fontsize=12)
    ax2.set_ylabel('Accuracy', color='b', fontsize=12)
    plt.title('Batch Metrics During Training', fontsize=16)
    
    # 添加图例
    plt.legend([line1, line2], ['Batch Loss', 'Batch Accuracy'], loc='best')
    
    # 图2: Epoch指标
    plt.subplot(2, 1, 2)
    
    # 创建双Y轴
    ax3 = plt.gca()
    ax4 = ax3.twinx()
    
    # 绘制epoch损失和准确率
    line3, = ax3.plot(history['epoch'], history['train_loss'], 'g-', linewidth=2, label='Epoch Loss')
    line4, = ax4.plot(history['epoch'], history['val_acc'], 'b-', linewidth=2, label='Validation Accuracy')
    
    # 添加验证损失
    line5, = ax3.plot(history['epoch'], history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    
    # 设置标签和标题
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    plt.title('Epoch Metrics', fontsize=16)
    
    # 设置x轴刻度为整数
    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # 添加图例
    plt.legend([line3, line5, line4], ['Training Loss', 'Validation Loss', 'Validation Accuracy'], loc='best')
    
    plt.tight_layout()
    plt.savefig('loss_accuracy.png', dpi=150)
    plt.close()

# ====================== 新增功能2: 数字预测可视化 ======================
def visualize_digit_predictions(model, test_images, test_labels):
    """可视化每个数字的预测结果，合并图像和logits"""
    # 为每个数字收集5个样本（原来是10个）
    digit_samples = {i: [] for i in range(10)}
    digit_indices = {i: [] for i in range(10)}
    
    # 收集样本及其索引
    for i, label in enumerate(test_labels):
        if len(digit_samples[label]) < 5:  # 只收集5个样本
            digit_samples[label].append(test_images[i])
            digit_indices[label].append(i)
    
    # 创建更大的图形 - 增加宽度以容纳更宽的热力图
    plt.figure(figsize=(30, 20))  # 宽度30英寸，高度20英寸
    plt.suptitle('Digit Predictions with Logits Visualization', fontsize=28, y=0.98)
    
    # 为每个数字的每个样本创建子图
    for digit in range(10):
        for sample_idx, (image, idx) in enumerate(zip(digit_samples[digit], digit_indices[digit])):
            # 获取预测结果
            inputs = Tensor(np.array([image]))
            outputs = model(inputs)
            logits = outputs.data[0]
            
            # 原始图像位置 (左)
            ax_img = plt.subplot(10, 10, digit * 10 + sample_idx + 1)
            plt.imshow(image.reshape(28, 28), cmap='gray')
            plt.axis('off')
            
            # 添加真实标签和预测标签
            pred_label = np.argmax(logits)
            true_label = test_labels[idx]
            color = 'green' if pred_label == true_label else 'red'
            ax_img.set_title(f'T:{true_label} P:{pred_label}', color=color, fontsize=14)
            
            # Logits热力图位置 (右) - 宽度为图像部分的3倍
            # 计算热力图的位置（每个数字对应5个样本，所以热力图在每行的6-10列）
            ax_logits = plt.subplot(10, 10, digit * 10 + sample_idx + 6)
            
            # 创建更宽的热力图 (宽度增加3倍)
            heatmap = ax_logits.imshow([logits], cmap='viridis', aspect=0.5)  # aspect=0.5使热力图更宽
            
            # 设置坐标轴
            ax_logits.set_xticks(np.arange(10))
            ax_logits.set_xticklabels([str(i) for i in range(10)], fontsize=12)
            ax_logits.set_yticks([])
            
            # 添加logits值文本 - 更小的字体（原来的一半）
            for j, logit_val in enumerate(logits):
                color = 'white' if logit_val < np.max(logits)/2 else 'black'
                ax_logits.text(j, 0, f'{logit_val:.2f}', color=color, 
                              ha='center', va='center', fontsize=6)  # 字号从12减小到6
    
    # 添加颜色条
    cbar_ax = plt.axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(heatmap, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    cbar_ax.set_title('Logit Value', fontsize=18, pad=20)
    
    # 添加列标题
    plt.figtext(0.25, 0.95, 'Digit Images', ha='center', fontsize=24)
    plt.figtext(0.75, 0.95, 'Logits Visualization (3x Wider)', ha='center', fontsize=24)
    
    # 添加行标题（数字标签）
    for digit in range(10):
        plt.figtext(0.02, 0.9 - digit * 0.09, f'Digit {digit}:', 
                   ha='right', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0.03, 0, 0.95, 0.96])  # 左侧留出空间给行标题
    plt.savefig('digit_predictions.png', dpi=200)  # 更高DPI确保清晰度
    plt.close()

