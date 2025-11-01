import yaml
import torch
import torch.distributed as dist
from configmypy import Bunch
import opt_einsum

def readConfig(config):
    """
    Safe read config based on yaml
    """
    with open(config, "r") as f:
        conf = yaml.safe_load(f)
    return Bunch(conf)

def format_complexTensor(weight):
    """
    Convert complex to real for torch DDP with 
    NCCL communication
    """
    if weight.is_complex():
        R = torch.view_as_real(weight)
    else:
        R  = weight
    return R

def deformat_complexTensor(weight):  
    """
    Convert real to complex 
    """
    if weight.is_complex():
        R = weight
    else:
        R  = torch.view_as_complex(weight)
    return R

@torch._dynamo.disable
def print_rank0(message):
    """
    If distributed training is initiliazed, print only on rank 0
    """
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)
        
@torch._dynamo.disable
def einsum_complexhalf(eq, *args):
    """
    Compute einsum for complex half tensors
    since torch.einsum is not supported for
    torch.complex32 (torch.float16, torch.float16)
    """
    
    input_output = eq.split('->')
    input_label = input_output[0].split(',')
    tensors = dict(zip(input_label, args))

    # view_as_real: [..., 2] in torch.float16
    for label, input in tensors.items():
        if input.is_conj():
            input = input.resolve_conj()
        input = torch.view_as_real(input)
        if input.dtype != torch.float16:
            input = input.half()
        tensors[label] = input

    if len(input_label) == 2:
        new_eqn = input_label[0] + "l," + input_label[1]+ "m->lm" + input_output[1]
        inp_tensors = [*tensors.values()]
        m = torch.einsum(new_eqn, inp_tensors[0], inp_tensors[1])
        # m[0,0] = Re(a) * Re(b), m[0,1] = Re(a) * Im(b)
        # m[1,0] = Im(a) * Re(b), m[1,1] = Im(a) * Im(b)
        # (a_r + i a_i)(b_r + i b_i) = (a_r*b_r - a_i*b_i) + i(a_i*b_r + a_r*b_i)
        output = torch.stack(
                [m[0, 0, ...] - m[1, 1, ...],
                 m[1, 0, ...] + m[0, 1, ...]],dim = -1
                )
        return torch.view_as_complex(output)

    else:
        # find the optimal path using opt_einsum
        _, path_info = opt_einsum.contract_path(eq, *args)
        partial_eqns = [contraction_info[2] for contraction_info in path_info.contraction_list]
        for peq in partial_eqns:
            # get new input labels from optimized equation
            inp_label, out_label = peq.split('->')
            inp_label = inp_label.split(',')
            in_tensors = [tensors[label] for label in inp_label]

            # add new dimensions for view_as_real
            new_eqn = inp_label[0] + "l," + inp_label[1] + "m->lm" + out_label
            m = torch.einsum(new_eqn, *in_tensors)
            output = torch.stack(
                [m[0, 0, ...] - m[1, 1, ...],
                 m[1, 0, ...] + m[0, 1, ...]],dim = -1
                )
            tensors[out_label] = output

        return torch.view_as_complex(tensors[input_output[1]])

class NoScale:
    """
    Dummy function when not using
    torch.amp.GradScaler for mixed 
    precision
    """
    def scale(self, loss):
        return loss
    def step(self, optimizer):
        optimizer.step()
    def update(self):
        pass

def compile_timing(func):
    """
    Function to return timing in seconds
    and result of running func.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = func()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

@torch._dynamo.disable
def optimizer_step(scaler, optimizer):
    scaler.step(optimizer)
    scaler.update()

@torch._dynamo.disable
def scheduler_step(scheduler):
    scheduler.step()

def dtype_debug_hook(module, input, output):
    # Get input dtypes
    input_dtypes = [i.dtype if isinstance(i, torch.Tensor) else type(i) for i in input]
    output_dtype = output.dtype if isinstance(output, torch.Tensor) else type(output)
    
    print(f"[Hook] {module.__class__.__name__}")
    print(f"  ├─ input dtypes: {input_dtypes}")
    print(f"  ├─ output dtype: {output_dtype}")
    print(f"  └─ device: {output.device if isinstance(output, torch.Tensor) else 'N/A'}\n")

def register_dtype_hooks(model):
    hooks = []
    for _, module in model.named_modules():
        # Skip the top-level model container itself
        if len(list(module.children())) == 0:
            hook = module.register_forward_hook(dtype_debug_hook)
            hooks.append(hook)
    return hooks
