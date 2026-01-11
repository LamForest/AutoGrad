这里参考PyTorch的autograd，实现了一个完全不依赖Pytorch的简化版的自动求导引擎，并支持add, sub, pow, mul, relu, matmul等算子。

同时提供了一个mnist示例，利用上述实现的自动求导引擎进行端到端的模型训练，验证集准确率可达到99.25%。

# 运行mnist

1. 依赖


```sh
cd mnist
PYTHONPATH=../src python mnist.py
```
首次运行会下载mnist数据集。

默认训练5个epoch，需要约25分钟，可以调整 num_epochs=500 缩短训练时间。

最终validate accuracy为99.25%:

```sh
...
Epoch [497/500], Time: 3.06s, Train Loss: 0.0062, Train Acc: 0.9920, Val Loss: 0.0061, Val Acc: 0.9920
Epoch [498/500], Time: 3.05s, Train Loss: 0.0062, Train Acc: 0.9920, Val Loss: 0.0061, Val Acc: 0.9921
Epoch [499/500], Time: 2.86s, Train Loss: 0.0062, Train Acc: 0.9920, Val Loss: 0.0061, Val Acc: 0.9921
Epoch [500/500], Time: 3.09s, Train Loss: 0.0062, Train Acc: 0.9921, Val Loss: 0.0061, Val Acc: 0.9925
```