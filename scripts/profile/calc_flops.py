#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To get FLOPS for custom operations set:
ENABLE_FLOP_WRAPPERS=1 python calc_flops.py --config=config.yaml
   
"""
import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))
from operator_learning.utils import flop_wrappers
import argparse
import torch
import torchprofile
from calflops import calculate_flops
from operator_learning.utils.misc import readConfig
from operator_learning.model import FNO

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='FLOP calculation for 1D/2D/3D FNO model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--config", default="config.yaml", help="configuration file")
args = parser.parse_args()
config = readConfig(args.config)

if "profile" in config:
    args.__dict__.update(**config.profile)

sections = ["model", "profile"]
assert "model" in config, "config file needs a 'model' section for profiling"
configs = {name: config.get(name) for name in (sections)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


# FNO model
model = FNO(**configs['model']).to(device)
# change according to model 
input = torch.rand(1, 5, 256, 256, 64).to(device)  
input_shape = (1, 5, 256, 256, 64)
print(f"For FNO model: {config['model']}\nusing input of {input_shape}")

# 1 MAC (Multiply-Accumulate Operations)= 2 FLOPs
# torchprofile
torch_macs = torchprofile.profile_macs(model, input)
custom_flops = sum(flop_wrappers.flop_counter.values()) 
custom_macs = custom_flops/2
total_macs = torch_macs + custom_macs

data = [
    ["Custom Operations", f"{flop_wrappers.flop_counter}"],
    ["Torchprofile MAC", f"{torch_macs:,}"],
    ["Custom MAC", f"{custom_macs:,}"],
    ["Torchprofile FLOP", f"{torch_macs * 2:,}"],
    ["Custom FLOP", f"{custom_flops:,}"],
    ["Total MAC", f"{total_macs:,}"],
    ["Total FLOP", f"{2 * total_macs / 10**12:.4f} TFLOP"],
]

# Print formatted table
print("=" * 80)
print(f"{'Metric':<30} | {'Value':>45}")
print("-" * 80)
for metric, value in data:
    print(f"{metric:<30} | {value:>45}")
print("=" * 80)


# calcflops
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
print("Calflops Model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))



# TorchFLOPsByFX
# from torch_flops import TorchFLOPsByFX
# # NOTE: First run the model once for accurate time measurement in the following process.
# with torch.no_grad():
#     model(input)
# with torch.no_grad():
#     # Build the graph of the model. You can specify the operations (listed in `MODULE_FLOPs_MAPPING`, `FUNCTION_FLOPs_MAPPING` and `METHOD_FLOPs_MAPPING` in 'flops_ops.py') to ignore.
#     flops_counter = TorchFLOPsByFX(model)
#     # Print the graph (not essential)
#     print('*' * 120)
#     flops_counter.graph_model.graph.print_tabular()
#     # Feed the input tensor
#     flops_counter.propagate(input)
# # Print the flops of each node in the graph. Note that if there are unsupported operations, the "flops" of these ops will be marked as 'not recognized'.
# print('*' * 120)
# result_table = flops_counter.print_result_table()
# # Print the total FLOPs
# total_flops = flops_counter.print_total_flops(show=True)
# total_time = flops_counter.print_total_time()
# max_memory = flops_counter.print_max_memory()