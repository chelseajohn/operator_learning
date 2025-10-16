import torch
import torch.nn.functional as F

# Dictionary to store FLOPs
flop_counter = {}

# Helper to wrap any function or tensor method
def wrap_with_flops(module_or_func, name, flop_fn):
    original = module_or_func
    if isinstance(module_or_func, type(torch.Tensor.numpy)):
        # Tensor method like numpy
        def wrapper(self, *args, **kwargs):
            flops = int(flop_fn(self, *args, **kwargs))
            flop_counter[name] = flop_counter.get(name, 0) + flops
            return original(self, *args, **kwargs)
        return wrapper
    else:
        # Normal function
        def wrapper(*args, **kwargs):
            flops = int(flop_fn(*args, **kwargs))
            flop_counter[name] = flop_counter.get(name, 0) + flops
            return original(*args, **kwargs)
        return wrapper

# ---- FLOP functions ----
def zero_flop(*args, **kwargs) -> int:
    return 0

def gelu_flop(x, approximate='none') -> int:
    return x.numel() * 8

def fft_flop(x, *args, **kwargs) -> int:
    n = x.numel()
    # Use math.log2 instead of torch.log2 to avoid tensors
    import math
    return int(n * math.log2(max(n, 2)))  # avoid log2(0)

# Einsum → approximate as product of input dims
def einsum_multi_flops(node, inputs, outputs) -> int:
    from collections import Counter
    from functools import reduce
    import operator

    equation = inputs[0]
    tensors = [t for t in inputs if isinstance(t, torch.Tensor)]

    if '->' in equation:
        input_subscripts, output_subscript = equation.split('->')
        output_subscript = output_subscript.strip()
    else:
        input_subscripts = equation
        output_subscript = None
    input_subscripts = [s.strip() for s in input_subscripts.split(',')]

    dim_map = {}
    for subs, tensor in zip(input_subscripts, tensors):
        if len(subs) != tensor.dim():
            raise ValueError(f"Mismatched dims for tensor {tensor.shape} and subscripts {subs}")
        for c, d in zip(subs, tensor.shape):
            if c in dim_map and dim_map[c] != d:
                raise ValueError(f"Conflicting sizes for index '{c}'")
            dim_map[c] = d

    all_input_indices = ''.join(input_subscripts)
    if output_subscript is None:
        output_subscript = ''.join([c for c, count in Counter(all_input_indices).items() if count == 1])
    
    sum_indices = set(all_input_indices) - set(output_subscript)

    num_output_elements = reduce(operator.mul, [dim_map[i] for i in output_subscript], 1) if output_subscript else 1
    sum_dim_product = reduce(operator.mul, [dim_map[i] for i in sum_indices], 1) if sum_indices else 1

    flops = num_output_elements * sum_dim_product  # multiplications
    if sum_indices:
        flops += num_output_elements * (sum_dim_product - 1)  # additions

    return int(flops)

def einsum_flop(equation, *tensors) -> int:
    return einsum_multi_flops(None, [equation] + list(tensors), None)

# ---- Wrap functions ----
torch.reshape = wrap_with_flops(torch.reshape, 'reshape', zero_flop)
torch.eye = wrap_with_flops(torch.eye, 'eye', zero_flop)
torch.view_as_complex = wrap_with_flops(torch.view_as_complex, 'view_as_complex', zero_flop)
torch.Tensor.numpy = wrap_with_flops(torch.Tensor.numpy, 'numpy_t', zero_flop)

F.gelu = wrap_with_flops(F.gelu, 'gelu', gelu_flop)
torch.fft.rfftn = wrap_with_flops(torch.fft.rfftn, 'fft_rfftn', fft_flop)
torch.fft.irfftn = wrap_with_flops(torch.fft.irfftn, 'fft_irfftn', fft_flop)
torch.einsum = wrap_with_flops(torch.einsum, 'einsum', einsum_flop)
