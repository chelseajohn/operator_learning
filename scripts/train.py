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
from operator_learning.utils.misc import readConfig, print_rank0, enable_tf32_only_on_a100


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
    "--config", default="config.yaml", help="configuration file")
args = parser.parse_args()

config = readConfig(args.config)
if "train" in config:
    print_rank0(f'Overwriting args with config values..')
    args.__dict__.update(**config.train)

sections = ["data", "model", "optim", "lr_scheduler", "parallel_strategy", "loss"]
for name in sections:
    assert name in config, f"config file needs a {name} section"
# trainer class configs, "loss" parameter uses default if not specified
configs = {name: config.get(name) for name in (sections)}

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

    model = FourierNeuralOperator(**configs, checkpoint=args.checkpoint, debug=False,\
                                benchmark=benchmark, use_amp=use_amp, use_complex_amp=use_complex_amp, \
                                compile=compile, compile_mode=compile_mode)
    model.learn(args.epochs, args.saveInterval)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    enable_tf32_only_on_a100()
    if args.compile_train == 1:
        mp.set_start_method("spawn", force=True)
        mp.freeze_support()
    main(args)