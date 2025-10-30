'''
Script to benchmark FNO with torch.compile
{'default': {},
 'reduce-overhead': {'triton.cudagraphs': True},
 'max-autotune-no-cudagraphs': {'max_autotune': True},
 'max-autotune': {'max_autotune': True, 'triton.cudagraphs': True}
 }
Execution Modes:
- compile_eval: Running inference with different compile modes, required: args.input_shape
- compile_train: Run training in benchmark mode, required: --compile_mode, optional: args.use_amp
'''


import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))
import argparse
import numpy as np
import pandas as pd
import torch, gc
import torch.multiprocessing as mp
from operator_learning.utils.misc import readConfig, print_rank0, compile_timing
torch.set_float32_matmul_precision('high')
torch._dynamo.config.cache_size_limit = 64  

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='torch.compile for 1D/2D/3D FNO model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--config", default="config.yaml", help="configuration file")
parser.add_argument(
    "--use_amp", type=int, default=0, help="mixed precision training [0:False, 1:True]")
parser.add_argument(
    "--compile_eval", type=int, default=0, help="use torch.compile for inference [0:False, 1:True]")
parser.add_argument(
     "--input_shape", type=int, nargs='+', help="Input tensor shape: e.g., --input_shape 16 4 64 64")
parser.add_argument(
    "--compile_train", type=int, default=0, help="use torch.compile for training [0:False, 1:True]")
parser.add_argument(
    "--compile_mode", type=str, default="default", 
    help="compile options ['eager', 'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']"
)

args = parser.parse_args()

def main(args):
    config = readConfig(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    print_rank0(f'Using torch.compile on {device_name}')
    compile_modes = ['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']
    N_Iters = 5

    if args.compile_eval == 1:
        from operator_learning.model import FNO
        print_rank0('*' * 120)
        print_rank0('Inference Mode')
        print_rank0('*' * 120)
        assert 'model' in config, f"config file needs a model section"
        # FNO model
        model = FNO(**config['model'], device=device).to(device)
        input = torch.rand(*args.input_shape).to(device)
        
        # Inference Eager Execution
        eager_times = []
        for i in range(N_Iters):
            with torch.no_grad():
                _, eager_time = compile_timing(lambda: model(input))
            eager_times.append(eager_time)
            print_rank0(f"Eager eval time (s) {i}: {eager_time}")
        eager_mean = np.mean(eager_times[2:])
        print_rank0(f"Eager eval mean time (s): {eager_mean}")
        print_rank0('*' * 120)
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Inference Compiled Execution
        compile_means = []
        speedup = []
        for mode in compile_modes:
            # Using torch.compile
            print_rank0(f'Using compile {mode} mode:')
            model_opt = FNO(**config['model'], device=device).to(device)
            model_opt.compile(mode=mode)
            compile_times = []
            for i in range(N_Iters):
                with torch.no_grad():
                    _, compile_time = compile_timing(lambda: model_opt(input))
                compile_times.append(compile_time)
                print_rank0(f"Compile mode eval time (s) in {i}: {compile_time}")
            print_rank0('*' * 120)

            compile_mean = np.mean(compile_times[2:])
            compile_means.append(np.round(compile_mean,3))
            speedup.append(np.round(eager_mean / compile_mean, 2))
            del model_opt
            torch.cuda.empty_cache()
            gc.collect()

        df = pd.DataFrame({'Compile Mode': compile_modes, 
                        'Compile(Eval) Mean Time (s)': compile_means,
                            'Speedup': speedup}
                        )
        # print_rank0(df.to_markdown(index=False, tablefmt="grid"))
        print_rank0(df.to_csv(index=False, sep='\t'))

    if args.compile_train == 1 :
        from training.train_fno import FourierNeuralOperator
        FourierNeuralOperator.LOSSES_FILE = 'loss.txt'
        FourierNeuralOperator.USE_TENSORBOARD = False
        use_amp = True if args.use_amp == 1 else False
        sections = ["data", "model", "optim", "lr_scheduler", "parallel_strategy", "loss"]
        for name in sections:
            assert name in config, f"config file needs a {name} section"
        configs = {name: config.get(name) for name in (sections)}

        print_rank0('*' * 120)
        print_rank0('Training Mode')
        print_rank0('*' * 120)
        print_rank0('Running FNO training for benchmarking...')
        if use_amp:
            print_rank0(f'Using mixed precision for training with float32 and float16...')
        
        if args.compile_mode == 'eager':
            # Training Eager Execution
            print_rank0(f'Using eager execution')
            model = FourierNeuralOperator(**configs, debug=False, device=device, \
                                            benchmark=True, use_amp=use_amp,
                                            compile=False)
            model.learn(nEpoch=N_Iters)
   
        if args.compile_mode in compile_modes:
            # Training Compiled Execution
            print_rank0(f'Using torch.compile with mode={args.compile_mode}')
            model_opt = FourierNeuralOperator(**configs, debug=False, device=device, \
                                            benchmark=True, use_amp=use_amp,
                                            compile=True, compile_mode=args.compile_mode)
            model_opt.learn(nEpoch=N_Iters)

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.freeze_support()
    main(args)
    




