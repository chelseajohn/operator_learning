import yaml
import torch
import torch.distributed as dist
from configmypy import Bunch


def readConfig(config):
    """
    Safe read config based on yaml
    """
    with open(config, "r") as f:
        conf = yaml.safe_load(f)
    return Bunch(conf)

def format_complexTensor(weight):
    """
    Convert torch.cfloat to torch.float32
    for torch DDP with NCCL communication
  
    """
  
    if weight.dtype == torch.complex64:
        R = torch.view_as_real(weight)
    else:
        R  = weight
    return R

def deformat_complexTensor(weight):  
    """
    Convert torch.float32 to torch.cfloat
    for computation
  
    """

    if weight.dtype != torch.complex64:
        R = torch.view_as_complex(weight)
    else:
        R  = weight
    return R

def print_rank0(message):
    """
    If distributed training is initiliazed, print only on rank 0
    """
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)