#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))

import argparse
import torch
import torch.multiprocessing as mp
from training.train_fno import FourierNeuralOperator
from operator_learning.utils.misc import readConfig, print_rank0, enable_tf32_only_on_a100, slugify
import numpy as np


# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Train a 1D/2D/3D FNO model on a given dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--trainDir", default="trainDir", help="directory to store training results")
parser.add_argument(
    "--dataFile", type=str, help="path to Hdf5 data file")
parser.add_argument(
    "--epochs", default=200, type=int, help="training epochs")
parser.add_argument(
    "--checkpoint", help="model checkpoint name")
parser.add_argument(
    "--saveInterval", default=100, type=int, help="save checkpoint interval")
parser.add_argument(
    "--disableTensorboard", action="store_true", help="disable Tensorboard logging")
parser.add_argument(
    "--lossesFile", default=FourierNeuralOperator.LOSSES_FILE, help='base text file to write the loss')
parser.add_argument(
    "--benchmark", action="store_true", help="benchmark run")
parser.add_argument(
    "--use_amp", type=int, default=0, help="mixed precision training  [0:False, 1:True]")
parser.add_argument(
    "--use_complex_amp", type=int, default=0, help="mixed precision training with explicit \
    complexHalf type  [0:False, 1:True]")
parser.add_argument(
    "--compile_train", type=int, default=0, help="use torch.compile for training [0:False, 1:True]")
parser.add_argument(
    "--compile_mode", type=str, default="default", 
    help="compile options ['eager', 'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']")
parser.add_argument(
    "--model_dtype", type=str, default="float32", 
    help="Model dtype for layers except FNO_DSE layer, options['float32', 'float64'] ")
parser.add_argument(
    "--fno_dtype", type=str, default="float32", 
    help="FNO_DSE Layer dtype, options['float32', 'float64'] ")
parser.add_argument(
    "--measure_power", type=int, default=0, help="use jpwr for power measurement during training [0:False, 1:True]")
parser.add_argument(
    "--batchSize", type=int, help="training global batch size")
parser.add_argument(
    "--val_batchSize",type=int, help="validation local batch size")
parser.add_argument(
    "--gas", type=int, help="gradient accumulation steps")
parser.add_argument(
    "--tp_size", type=int, help="input sharding")
parser.add_argument(
    "--config", default="config.yaml", help="configuration file")

args = parser.parse_args()

config = readConfig(args.config)
if "train" in config:
    print_rank0(f'Overwriting train args with config values..')
    args.__dict__.update(**config.train)

for key in ["batchSize", "val_batchSize", "gas"]:
    val = getattr(args, key)
    if val is not None:
        print_rank0(f'Overwriting {key} with arg value..')
        config.data[key] = val
if args.tp_size is not None:
    print_rank0(f'Overwriting tp_size with arg value..')
    config.parallel_strategy['tp_size'] = args.tp_size

sections = ["data", "model", "optim", "lr_scheduler", "parallel_strategy", "loss"]
for name in sections:
    assert name in config, f"config file needs a {name} section"
# trainer class configs, "loss" parameter uses default if not specified
configs = {name: config.get(name) for name in (sections)}
measure_power = True if args.measure_power == 1 else False

def main(args):
    # -----------------------------------------------------------------------------
    # Script execution
    # -----------------------------------------------------------------------------
    FourierNeuralOperator.TRAIN_DIR = args.trainDir
    FourierNeuralOperator.LOSSES_FILE = args.lossesFile
    FourierNeuralOperator.USE_TENSORBOARD = True if not args.disableTensorboard else False
    benchmark = True if args.benchmark else False
    use_amp = True if args.use_amp == 1 else False
    use_complex_amp = True if args.use_complex_amp == 1 else False
    compile = True if args.compile_train == 1 else False
    compile_mode = args.compile_mode
    if args.model_dtype == 'float32':
        model_dtype = torch.float32
    else:
        model_dtype = torch.float64
    if args.fno_dtype == 'float32':
        fno_dtype = torch.float32
    else:
        fno_dtype = torch.float64

    if benchmark:
        print_rank0('Running FNO training for benchmarking...')
        if not os.path.exists(configs['data']['dataFile']):
            configs['data']['dataFile'] = args.dataFile
    if use_amp:
        print_rank0(f'Using mixed precision for training with float32 and float16...')
        if use_complex_amp:
            print_rank0(f'Explicit casting to complexHalf and performing GEMM in spectral layer..')
    if compile:
        print_rank0(f'Using torch.compile in mode={compile_mode}')
    if measure_power:
        print_rank0('Measuring power usage during training using jpwr..')

    model = FourierNeuralOperator(**configs, checkpoint=args.checkpoint, debug=False,\
                                benchmark=benchmark, use_amp=use_amp, use_complex_amp=use_complex_amp, \
                                compile=compile, compile_mode=compile_mode, model_dtype=model_dtype,
                                fno_dtype=fno_dtype)
    model.learn(args.epochs, args.saveInterval)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")   
    seed = 152
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # enable_tf32_only_on_a100()
    torch.set_float32_matmul_precision("high")  # Enable TF32 matmul
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if args.compile_train == 1:
        mp.set_start_method("spawn", force=True)
        mp.freeze_support()
    
    if measure_power:
        from jpwr.ctxmgr import get_power
        import platform
        methods = set()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                if "AMD" in device_name:
                    methods.add("rocm")
                if "NVIDIA" in device_name:
                    methods.add("pynvml")
                if "GH200" in device_name:
                    methods.add("gh")
        
        power_methods = []
        for m in methods:
            if "rocm" == m:
                from jpwr.gpu.rocm import power
                power_methods.append(power())
            if "pynvml" == m:
                from jpwr.gpu.pynvml import power
                power_methods.append(power())
            if "gh" == m:
                from jpwr.sys.gh import power
                power_methods.append(power())
        
        with get_power(power_methods, 100) as measured_scope:  
            main(args)
        
        energy_df, additional_data = measured_scope.energy()
        nodename  = platform.node()
        rankid    = int(os.getenv("RANK"))
        
        power_file = f"{args.trainDir}/{nodename}_rank{rankid}.csv"
        measured_scope.df["nodename"] = nodename
        measured_scope.df["rank"] = rankid
        if not os.path.exists(power_file):
            measured_scope.df.to_csv(power_file)
        
        energy_df["nodename"] = nodename
        energy_df["rank"] = rankid
        energy_file = power_file.replace(".csv", f"_energy.csv")
        if not os.path.exists(energy_file):
            energy_df.to_csv(energy_file)
        
        print(f"Host: {nodename}")
        print(f"Energy-per-GPU-list integrated(Wh): \n{energy_df.to_string()}")
        for k,v in additional_data.items():
            additional_path = power_file.replace(".csv", f"{slugify(k)}.csv")
            print(f"Writing {k} df to {additional_path}")
            v.T.to_csv(additional_path)
            print(f"Energy-per-GPU-list from {k}(Wh): {v.to_string()}")
    else:
        main(args)
