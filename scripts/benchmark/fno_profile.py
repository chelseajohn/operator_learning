'''
Script to profile FNO Code with torch and nsys profiler

Nsight Systems:
nsys profile -o profile_report  python fno_profile.py --config=config.yaml

or distributed training
nsys profile -o nsys_report \
    -t cuda,nvtx,osrt \
    python -m torch.distributed.launch --nproc_per_node=4 fno_profile.py \
        --config=config.yaml


PROFILER_ARGS_NSYS=(
-t cuda,nvtx,cudnn,cublas,mpi,ucx,osrt
-s cpu 
--gpu-metrics-device=all
--stats=true 
--capture-range=cudaProfilerApi
--capture-range-end=stop 
--cudabacktrace all 
-x true 
--cuda-memory-usage true 
--nic-metrics true
--export hdf,json
-o $PROFILE_PATH
)

Torch profiler
python fno_profile.py --config=config.yaml
'''


import sys,os
from pathlib import Path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))
import argparse
import torch
from torch.profiler import profile, ProfilerActivity, schedule
from operator_learning.utils.misc import readConfig, print_rank0
torch.set_float32_matmul_precision('high')

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Profiling for 1D/2D/3D FNO model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--config", default="config.yaml", help="configuration file")
parser.add_argument(
    "--enableProfiler", action="store_true", help="enable profiling")
parser.add_argument(
    "--profiler", default="torch", choices=["torch", "nsys"], help="Profiler to use")
parser.add_argument(
    "--profileDir", default="profileDir", help="Profiler dir")
parser.add_argument(
    "--evalOnly", action="store_true", help="profile inference only")
parser.add_argument(
     "--input_shape", type=int, nargs='+', help="Input tensor shape: e.g., --input_shape 16 1 64 64")
parser.add_argument(
    "--benchmark", action="store_true", help="benchmark run")
parser.add_argument(
    "--use_amp", type=int, help="mixed precision training  [0:False, 1:True]")
parser.add_argument(
    "--compile_eval", action="store_true", help="use torch.compile for inference")
parser.add_argument(
    "--compile_train", action="store_true", help="use torch.compile for training")
parser.add_argument(
    "--compile_mode", type=str, default="default", 
    help="compile options ['eager', 'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']"
)
args = parser.parse_args()
config = readConfig(args.config)

if "profile" in config:
    args.__dict__.update(**config.profile)
    print_rank0(f'Using profiler args: {args.__dict__}')

os.makedirs(args.profileDir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
print_rank0(f'Profiling on {device_name} with {args.profiler}')

if args.evalOnly:
    from operator_learning.model import FNO
    print_rank0(f'Profiling Only Model Inference with torch profiler')
    assert 'model' in config, f"config file needs a model section"
    # FNO model
    model = FNO(**config['model'], device=device).to(device)
    input = torch.rand(*args.input_shape).to(device)
    activities = [ProfilerActivity.CUDA] # ProfilerActivity.CPU
    sort_by_keyword = str(device)+ "_time_total"
    fno_schedule = schedule(skip_first=0, wait=0, warmup=1, active=3, repeat=1)

    with profile(
        activities=activities, 
        record_shapes=True, 
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        schedule=fno_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{args.profileDir}'),
    ) as prof:
            for i in range(10):
                with torch.no_grad(): 
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = torch.cuda.memory_allocated()  
                    model(input)
                    torch.cuda.synchronize()
                    after_fwd = torch.cuda.memory_allocated()
                    peak = torch.cuda.max_memory_allocated()

                    print(f"Forward alloc: {(after_fwd - start)/1e6:.2f} MB")
                    print(f"Peak alloc: {peak/1e6:.2f} MB")
                prof.step()             

    print_rank0('*' * 120)
    print_rank0(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
    print_rank0('*' * 120)

else:
    from training.train_fno import FourierNeuralOperator
    sections = ["data", "model", "optim", "lr_scheduler", "parallel_strategy", "loss", "profile"]
    for name in sections:
        assert name in config, f"config file needs a {name} section"
    configs = {name: config.get(name) for name in (sections)}
    FourierNeuralOperator.TRAIN_DIR = args.profileDir
    FourierNeuralOperator.LOSSES_FILE = 'loss.txt'
    FourierNeuralOperator.USE_TENSORBOARD = False
    benchmark = True if args.benchmark else False
    use_amp = True if args.use_amp == 1 else False

    if benchmark:
        print_rank0('Running FNO training for benchmarking...')
    if use_amp:
        print_rank0(f'Using mixed precision for training with float32 and float16...')

    model = FourierNeuralOperator(**configs,debug=False, device=device, \
                                  benchmark=benchmark, use_amp=use_amp)
    model.learn(nEpoch=3, save_interval=5)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()




