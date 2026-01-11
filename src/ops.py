from autograd import set_history, collect_next_edges
from autograd import AddBackward0, PowBackward0, MatMulBackward0, MulBackward0, ReLUBackward0, MeanBackward0, SubBackward0
from tensor import Tensor
import numpy as np


def add(_self, _other):
    """
    没有实现完整的add语义，仅能够做tensor add
    """
    grad_fn = None
    any_requires_grad = _self.requires_grad or _other.requires_grad

    if any_requires_grad:
        grad_fn = AddBackward0()
        grad_fn.next_edges = collect_next_edges(_self, _other)

    _result = Tensor(np.add(_self.data, _other.data),requires_grad=any_requires_grad) #result require_grads我还不确定在哪里设的

    if grad_fn:
        set_history(_result, grad_fn)
    # _result.grad_fn = grad_fn
    return _result

def mul(_self, _other):
    grad_fn = None
    any_requires_grad = _self.requires_grad or _other.requires_grad
    if any_requires_grad:
        grad_fn = MulBackward0()
        grad_fn.next_edges = collect_next_edges(_self, _other)
        if _self.requires_grad: #torch是通过should_compute_output判断的，简化为requires_grad了
            grad_fn._saved_other = _other
        if _other.requires_grad:
            grad_fn._saved_self = _self

    _result = Tensor(np.multiply(_self.data, _other.data), requires_grad=any_requires_grad)

    if grad_fn:
        set_history(_result, grad_fn)
    return _result


def matmul(_self, _other):
    # 简单的矩阵乘法实现
    grad_fn = None
    any_requires_grad = _self.requires_grad or _other.requires_grad
    if any_requires_grad:
        grad_fn = MatMulBackward0()
        grad_fn.next_edges = collect_next_edges(_self, _other)
        if _self.requires_grad:
            grad_fn._saved_other = _other
        if _other.requires_grad:
            grad_fn._saved_self = _self
    
    result = Tensor(np.matmul(_self.data, _other.data), requires_grad=any_requires_grad)
    
    if grad_fn:
        set_history(result, grad_fn)

    return result

def relu(_self):
    grad_fn = None
    any_requires_grad = _self.requires_grad
    if any_requires_grad:
        grad_fn = ReLUBackward0()
        grad_fn.next_edges = collect_next_edges(_self)
        # grad_fn._saved_input = _self
    
    result = Tensor(np.maximum(0, _self.data), requires_grad=any_requires_grad)
    
    grad_fn._saved_result = result #TODO 循环引用？ grad_fn.saved_result->result, result.grad_fn -> grad_fn 循环了。有什么害处吗？
    if grad_fn:
        set_history(result, grad_fn)

    return result

def pow(_self, exponent):
    """
    实现张量的幂运算
    """
    grad_fn = None
    if _self.requires_grad:
        grad_fn = PowBackward0(exponent)
        grad_fn.next_edges = collect_next_edges(_self)
        grad_fn._saved_self = _self
    result_data = np.power(_self.data, exponent)
    _result = Tensor(result_data, requires_grad=_self.requires_grad)
    
    if grad_fn:
        set_history(_result, grad_fn)
    
    return _result

def mean(_self):
    """实现平均值操作的前向传播（全局平均）"""
    grad_fn = None
    if _self.requires_grad:
        numel = np.prod(_self.shape)  # 计算元素总数
        grad_fn = MeanBackward0(numel, _self.shape)
        grad_fn.next_edges = collect_next_edges(_self)
    
    result_data = np.mean(_self.data)
    _result = Tensor(result_data, requires_grad=_self.requires_grad)
    
    if grad_fn:
        set_history(_result, grad_fn)
    
    return _result

def sub(_self, _other):
    """实现减法操作的前向传播"""
    # 确保输入维度相等
    assert _self.data.shape == _other.data.shape, "减法操作要求输入维度相等"
    
    grad_fn = None
    any_requires_grad = _self.requires_grad or _other.requires_grad
    
    if any_requires_grad:
        grad_fn = SubBackward0()
        grad_fn.next_edges = collect_next_edges(_self, _other)
    
    result_data = np.subtract(_self.data, _other.data)
    _result = Tensor(result_data, requires_grad=any_requires_grad)
    
    if grad_fn:
        set_history(_result, grad_fn)
    
    return _result