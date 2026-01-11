import numpy as np
from dataclasses import dataclass
from typing import List, Any, Optional, Tuple, Dict
from collections import namedtuple
import queue
import numpy as np
from tensor import Tensor, Node, Edge, InputMeta
from collections import defaultdict



class AccumulateGrad(Node):
    def __init__(self, tensor):
        super().__init__('AccumulateGrad')
        self.variable = tensor
        #AccumulateGrad不会通过set_history增加input_meta；所以只能在初始化的时候增加input_meta
        #也挺合理的，因为grad 和 tensor 的 meta是完全一样的
        self.input_meta.append(InputMeta(tensor.shape, tensor.dtype))
    
    def apply(self, grads):
        grad = grads[0]
        """
        torch/csrc/autograd/functions/accumulate_grad.h 的 accumulateGrad做了优化，
        假设没有其他人使用grad，则self.variable.grad = grad，即直接使用grad

        否则self.variable.grad = grad.clone；考虑一种情况：
            c = a + b；则dy/dc = dy/da = dy/db，而AddBackward不会clone，只会传递相同的grad给到a、b的AccumulateGrad
            如果直接clone，则a、b的grad是同一个tensor，对a.grad.zero_()会影响b.grad，不符合用户期望。

        这里我们简化为统一clone
        """
        if self.variable.grad is None:
            self.variable.grad = Tensor(grad.data.copy()) #numpy clone 为 copy
        else:
            self.variable.grad.data += grad.data
        return []


class AddBackward0(Node):
    def __init__(self):
        super().__init__('AddBackward0')

    def apply(self, grads):
        """
        Source: torch/csrc/autograd/generated/Functions.cpp
            variable_list AddBackward0::apply(variable_list&& grads)
        """
        grad = grads[0]
        grad_inputs = [None] * 2
        if self.task_should_compute_output(0):
            # print("AddBackward0::apply 0 grad")
            grad_inputs[0] = grad
        if self.task_should_compute_output(1):
            # print("AddBackward0::apply 1 grad")
            grad_inputs[1] = grad
        return grad_inputs

class MulBackward0(Node):
    def __init__(self):
        super().__init__('MulBackward0')
        self._saved_self = None
        self._saved_other = None

    def apply(self, grads):
        """
        Source: torch/csrc/autograd/generated/Functions.cpp
            variable_list MulBackward0::apply(variable_list&& grads)
        """
        grad_inputs = [None] * 2
        grad = grads[0]
        if self.task_should_compute_output(0):
            assert self._saved_other is not None
            grad_inputs[0] = Tensor(grad.data * self._saved_other.data)
        if self.task_should_compute_output(1):
            assert self._saved_self is not None
            grad_inputs[1] = Tensor(grad.data * self._saved_self.data)

        return grad_inputs

    def release_variables(self):
        self._saved_self = None
        self._saved_other = None


# 实现矩阵乘法的反向传播
class MatMulBackward0(Node):
    def __init__(self):
        super().__init__('MatMulBackward0')
        self._saved_self = None # (b, m)
        self._saved_other = None # (m, n)
    
    def apply(self, grads):
        grad = grads[0] # (b, n)
        grad_inputs = [None] * 2
        # print(f"{grad.data.shape=}, {self._saved_other.data.shape=}, {self._saved_self.data.shape=}")
        if self.task_should_compute_output(0):
            assert self._saved_other is not None
            grad_inputs[0] = Tensor(np.matmul(
                grad.data, #(b, n)
                self._saved_other.data.T # (n, m)
            ))
        if self.task_should_compute_output(1):
            assert self._saved_self is not None
            grad_inputs[1] = Tensor(np.matmul(
                self._saved_self.data.T, # (m, b)
                grad.data # (b, n)
            ))
        # print(f"{type(grad_inputs[0])=}, {type(grad_inputs[1])=}")
        return grad_inputs
    
    def release_variables(self):
        self._saved_self = None
        self._saved_other = None


# 实现ReLU的反向传播
class ReLUBackward0(Node):
    def __init__(self):
        super().__init__('ReLUBackward0')
        self.next_edges = []
        self.input_meta = []
        self._saved_result = None
    
    def apply(self, grads):
        grad = grads[0]
        grad_inputs = [None] * 1
        if self.task_should_compute_output(0):
            grad_inputs[0] = Tensor(grad.data * (self._saved_result.data > 0))
        return grad_inputs
    
    def release_variables(self):
        self._saved_result = None


def grad_accumulator(tensor: Tensor):
    """
    Source:
        torch/csrc/autograd/variable.cpp
        std::shared_ptr<Node> grad_accumulator(const Variable& self)
    """
    if not tensor.requires_grad:
        return None
    if tensor.autograd_meta.grad_accumulator_ is None:
        """
        a = Tensor(requires_grad=True)
        b = a + a
        则a的indegree=2，需要复用AccumulateGrad结点。
        PyTorch中，在tensor的autograd_meta使用了std::weak_ptr<Node>记录该tensor被哪个AccumulateGrad所指向（为了避免引用循环）
        这里为了简洁，没有处理 tensor -> AccumulateGrad -> tensor
        """
        tensor.autograd_meta.grad_accumulator_ = AccumulateGrad(tensor)
    return tensor.autograd_meta.grad_accumulator_

def gradient_edge(tensor):
    """
    Source:
        torch/csrc/autograd/variable.cpp
        Edge gradient_edge(const Variable& self)
    """
    if tensor.autograd_meta.grad_fn is None:
        return Edge(grad_accumulator(tensor), 0)
    else:
        return Edge(tensor.autograd_meta.grad_fn, tensor.autograd_meta.output_nr_)

# def set_gradient_edge(tensor, edge):
#     tensor.autograd_meta.grad_fn = edge.function

#     tensor.autograd_meta.output_nr_ = edge.input_nr

def set_history(tensors: List[Tensor], grad_fn: Node):
    if isinstance(tensors, Tensor):
        tensors = [tensors]
    for tensor in tensors:
        tensor.autograd_meta.grad_fn = grad_fn
        output_nr = len(grad_fn.input_meta)
        grad_fn.input_meta.append(InputMeta(tensor.data.shape, tensor.data.dtype))
        tensor.autograd_meta.output_nr_ = output_nr

def collect_next_edges(*tensors):
    """
    Source:
    torch/csrc/autograd/function.h : 
        template <typename... Variables>
        edge_list collect_next_edges(Variables&&... variables) {
            detail::MakeNextFunctionList make;
            make.apply(std::forward<Variables>(variables)...);
            return std::move(make.next_edges);
        }
    """
    edges = []
    for tensor in tensors:
        edges.append(gradient_edge(tensor))
    return edges




@dataclass
class InputBuffer:
    buffer: List[Tensor]
    def add(self, pos: int, tensor: Tensor):
        assert pos < len(self.buffer), f"pos={pos}, len(self.buffer)={len(self.buffer)}"
        if self.buffer[pos] is None:
            self.buffer[pos] = tensor
        else:
            #####
            # buffer[pos]的grad可能被在其他地方有使用，torch/csrc/autograd/input_buffer.cpp的
            # can_accumulate_inplace通过引用计数判断是否可以inplace add。
            # 这里简化为任何情况下都不允许inplace add，而是新建tensor
            #
            # 可以试着改为 self.buffer[pos].data += tensor.data，再运行test_autograd.py 的 test_complex_graph()单测，会报错。
            #####
            self.buffer[pos] = Tensor(self.buffer[pos].data + tensor.data)


@dataclass
class NodeTask:
    """
    代表一次结点的计算任务
    fn: Node 比如 MulBackward0
    inputs: Node 的输入
    """
    fn: Node
    inputs: InputBuffer

@dataclass
class GraphTask:
    """
    Source:torch/csrc/autograd/graph_task.h
        代表一次反向任务，持有queue, input_buffer, dependencies等资源
    """
    queue: queue.Queue
    not_ready: Dict[Node, InputBuffer]
    dependencies: Dict[Node, int]



def compute_dependencies(graph_task: GraphTask, root: Node):
    """
    Source:torch/csrc/autograd/function.h
        void Function::compute_dependencies(
            std::shared_ptr<GraphTask>& graph_task,
            const std::shared_ptr<ReadyQueue>& cpu_ready_queue
        )
    """
    q = queue.Queue() #用于广度优先遍历的
    visited = set()
    q.put(root)
    while not q.empty():
        node: Node = q.get()
        visited.add(node)
        for edge in node.next_edges:
            if edge.function is not None:
                graph_task.dependencies[edge.function] += 1
                if edge.function not in visited:
                    q.put(edge.function)

    # for node, in_degree in graph_task.dependencies.items():
    #     print(f"{node=}, {in_degree=}")


def evaluate_function(fn: Node, inputs: InputBuffer, graph_task: GraphTask):
    """
    Source: torch/csrc/autograd/engine.cpp
        void Engine::evaluate_function(
            std::shared_ptr<GraphTask>& graph_task,
            Node* func,
            InputBuffer& inputs,
            const std::shared_ptr<ReadyQueue>& cpu_ready_queue
        )
    """


    ####
    # 执行 MulBackward0, AddBackward0, ...
    # pytorch中使用了一个单独的函数call_function来执行，顺便调用pre_hook 和 post_hook
    # 这里简化为直接调用了
    ####
    _inputs = inputs.buffer  #InputBuffer -> List[Tensor]
    inputs.buffer = None    # 释放
    # call retain grads hook
    for hook in fn.retains_grad_hooks:
        hook(_inputs)
    grad_inputs = fn.apply(_inputs)

    Q = graph_task.queue

    # 释放反向结点持有的变量
    fn.release_variables()

    if len(grad_inputs) == 0:
        #leaf node
        return
    for next_edge, grad_input in zip(fn.next_edges, grad_inputs):
        next_function, next_input_nr = next_edge.function, next_edge.input_nr
        if grad_input is None:
            continue
        graph_task.dependencies[next_function] -= 1

        is_ready = graph_task.dependencies[next_function] == 0

        if next_function in graph_task.not_ready:
            input_buffer = graph_task.not_ready[next_function]  
            input_buffer.add(
                pos=next_input_nr,
                tensor=grad_input
            )

            if is_ready:
                Q.put(NodeTask(fn=next_function, inputs=input_buffer,))

        else:
            input_buffer = InputBuffer(
                [None] * next_function.num_inputs()
            )
            # print(f"{next_function.name=}, {next_function.num_inputs()=}")
            input_buffer.add(
                pos=next_input_nr,
                tensor=grad_input
            )

            if is_ready:
                Q.put(NodeTask(fn=next_function, inputs=input_buffer))
            else:
                graph_task.not_ready[next_function] = input_buffer


def backward(output: Tensor, grad_output: Tensor=None):
    if grad_output is None:
        # 自动插入ones_like的操作见 torch/autograd/__init__.py backward()
        # 和这里相同，只对Scalar Tensor生效
        if output.data.shape == ():
            grad_output = Tensor(np.ones_like(output.data))
        else:
            #torch中仅对scalar tensor默认进行ones_like
            raise ValueError(f"grad_output must be specified for non-scalar outputs, but {output.shape=}")
    
    graph_task = GraphTask(queue=queue.Queue(), not_ready={}, dependencies=defaultdict(int))
    compute_dependencies(graph_task, output.grad_fn)

    Q = graph_task.queue
    Q.put(
        NodeTask(
            fn=output.grad_fn,
            inputs=InputBuffer(buffer=[grad_output])
        )
    ) # root task


    """
    Source:torch/csrc/autograd/engine.cpp
        auto Engine::thread_main(const std::shared_ptr<GraphTask>& graph_task) -> void {
    pytorch使用了一个子线程，这里简化为在主线程中操作。

    这是图的广度优先遍历的一种变体。
    """
    while not Q.empty():
        task = Q.get()
        # print(f"processing task.fn={task.fn}, {task.inputs.buffer[0].dtype=}")
        
        evaluate_function(task.fn, task.inputs, graph_task)


def viz(tensor, name='autograd'):
    import sys
    sys.path.insert(0, '/Users/bytedance/Documents/code/cpp-cuda-py-torch-mlir/pytorch/torchviz')
    from viz import viz_graph
    viz_graph(tensor).render(name, format="png")



class PowBackward0(Node):
    def __init__(self, exponent):
        super().__init__('PowBackward0')
        self.exponent = exponent
        self._saved_self = None
    
    def apply(self, grads):
        grad = grads[0]
        grad_inputs = [None]
        if self.task_should_compute_output(0):
            # 梯度计算: d(x^n)/dx = n * x^(n-1)
            assert self._saved_self is not None
            grad_inputs[0] = Tensor(
                grad.data * self.exponent * np.power(self._saved_self.data, self.exponent - 1)
            )
        return grad_inputs
    
    def release_variables(self):
        self._saved_self = None

class MeanBackward0(Node):
    def __init__(self, n, shape):
        super().__init__('MeanBackward0')
        self._self_numel = n  # 参与平均的元素数量
        self._self_shape = shape
    
    def apply(self, grads):
        grad_output = grads[0]
        grad_inputs = [None]
        assert grad_output.shape == (), f"only support mean result is Scalar"
        # 梯度计算: ∂L/∂x = (∂L/∂y) / n
        if self.task_should_compute_output(0):
            grad_input = Tensor(
                #TODO 这里不加astype(np.float32)，会导致grad_input.dtype为fp64，还未看原因
                np.full(self._self_shape, grad_output.data / self._self_numel).astype(np.float32) 
            )
            # print(f"MeanBackward0: {grad_input.shape=}, {grad_input.dtype=}, {grad_output.dtype=}")
            grad_inputs[0] = grad_input
            
        return grad_inputs
    
    def release_variables(self):
        pass  # 不需要保存任何变量


class SubBackward0(Node):
    def __init__(self):
        super().__init__('SubBackward0')
    
    def apply(self, grads):
        grad_output = grads[0]
        grad_inputs = [None] * 2
        
        # 梯度计算: ∂L/∂a = ∂L/∂c, ∂L/∂b = -∂L/∂c
        if self.task_should_compute_output(0):
            grad_inputs[0] = grad_output
            
        if self.task_should_compute_output(1):
            grad_inputs[1] = Tensor(-grad_output.data)
            
        return grad_inputs
    
    def release_variables(self):
        pass  # 不需要保存任何变量