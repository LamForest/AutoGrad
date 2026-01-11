from dataclasses import dataclass
from typing import List, Any, Optional, Tuple
import numpy as np
from typing import NamedTuple

"""
Edge在c++侧是一个struct(torch/csrc/autograd/edge.h), 在python侧是一个tuple
于是用一个namedtuple来兼容。
"""

class Edge(NamedTuple):
    function: 'Node'
    input_nr: int
@dataclass
class InputMeta:
    shape: List[int]
    dtype: np.dtype


class Node:
    def __init__(self, name: str = None):
        # TODO: input_meta的作用是什么？
        self.input_meta: List[InputMeta] = []
        self.next_edges: List[Edge] = []
        self._name = name
        self.retains_grad_hooks: List[callable] = []

    def apply(self, grad):
        raise NotImplementedError
    
    def name(self):
        return self._name
    
    def task_should_compute_output(self, i):
        """
        Source: torch/include/torch/csrc/autograd/function.h
            bool task_should_compute_output(size_t output_edge_index)
        """
        assert i < len(self.next_edges)
        is_valid = self.next_edges[i].function is not None
        return is_valid
    
    @property
    def next_functions(self):
        """
        next_edge的别名
        因为c++侧，使用的是next_functions，而python侧使用的是next_edges
        """
        return self.next_edges
    
    def num_inputs(self):
        """
        结点在反向计算式
        uint32_t num_inputs() const noexcept {
            return input_metadata_.size();
        }
        """
        return len(self.input_meta)

    def release_variables(self):
        """
        见Node:
            virtual void release_variables() {}
        """
        pass

    def add_retains_grad_hook(self, hook):
        self.retains_grad_hooks.append(hook)

@dataclass
class AutogradMeta:
    grad_fn: Node = None
    output_nr_: int = 0
    grad_accumulator_: 'AccumulateGrad' = None #循环引用了
    retains_grad: bool = False


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad

        # self.next_edges = []
        self.autograd_meta = AutogradMeta()
        self.grad = None
        # self.grad_fn = None

    @property
    def grad_fn(self):
        return self.autograd_meta.grad_fn
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def is_leaf(self):
        return self.grad_fn is None
    
    def retain_grad(self):
        """
        torch/csrc/autograd/variable.cpp; void VariableHooks::retain_grad(const at::TensorBase& self) const;
        """
        if self.is_leaf():
            return

        if self.autograd_meta.retains_grad:
            return

        def retain_grad_hook(grads: List[Tensor]):
            """
            retain_grad_hook参数应该是Tensor，这里合并了CppFunctionSingleTensorPreHook，简化实现。
            """
            #这里统一做了clone，而没有考虑复用梯度，因为retain_grad作为一个hook，参数grad肯定会在反向计算时被其他结点用到
            #既然一定会被其他人用到，那就直接拷贝吧
            grad = grads[self.autograd_meta.output_nr_]
            if self.grad is not None:
                self.grad.data = self.grad.data + self.grad
            else:
                self.grad = Tensor(grad.data.copy())
        self.grad_fn.add_retains_grad_hook(retain_grad_hook)
        self.autograd_meta.retains_grad = True
        return self.data.device

    def __str__(self):
        return f"Tensor({self.data})"
    
