# Overview

The scripts facilitate training FNO models and comprehensive evaluation of trained models with visualization and analysis capabilities.

## Training
Main training script for 1D/2D/3D Fourier Neural Operator models.

**Features:**
- Supports distributed training with PyTorch
- Configurable via YAML configuration files
- Tensorboard logging integration
- Checkpoint saving and resuming
- Multiple optimizer and learning rate scheduler options

**Usage:**
```bash
python train.py --config config.yaml --epochs 50 --trainDir results/
```

**Key Parameters:**
- `--config`: YAML configuration file (default: `config.yaml`)
- `--trainDir`: Directory to store training results (default: `trainDir`)
- `--epochs`: Number of training epochs (default: 200)
- `--checkpoint`: Model checkpoint to resume from
- `--saveInterval`: Checkpoint saving interval (default: 100)
- `--disableTensorboard`: Disable Tensorboard logging
- `--lossesFile`: File to write loss values
- `--benchmark` : To get benchmark matrices

## Rayleigh Benard Convection (RBC) Evaluation
Comprehensive evaluation script for trained FNO models with detailed analysis and visualization for Rayleigh Benard Convection

**Features:**
- Multi-timestep autoregressive evaluation
- Domain decomposition analysis
- Relative L2 error computation over time
- Contour plots for solution visualization
- Spectral analysis and comparison
- Nusselt number analysis (for 3D)
- Inference time benchmarking
- Automated report generation in Markdown format

**Usage:**
```bash
python eval{2d,3d}.py --checkpoint model.pt --dataFile dataset.h5 --tSteps 10
```

**Key Parameters:**
- `--dataFile`: HDF5 dataset file (default: `dataset.h5`)
- `--checkpoint`: Trained model checkpoint file
- `--tSteps`: Number of autoregressive timesteps (default: 1)
- `--model_dt`: Model timestep (default: 1e-3)
- `--iSimu`: Simulation index to evaluate (default: 8)
- `--evalDir`: Output directory for evaluation results (default: `eval`)
- `--runId`: Run identifier for output files (default: 1)
- `--imgExt`: Image file extension (default: `png`)
- `--subtitle`: Subtitle for contour plots (default: `(256,64)`)
- `--config`: Configuration file for evaluation parameters

**Dataset Requirements**

The scripts work with HDF5 datasets containing:
- `inputs`: Initial conditions for simulations
- `outputs`: Target solutions or updates
- `infos`: Metadata including grid information, timesteps, and dataset parameters
- `xGrid`, `yGrid`: Spatial grid coordinates (if available)

## Particle In Cell (PIC) Evaluation
Comprehensive evaluation script for trained FNO models with visualization for PIC in 1D/2D

**Usage:**
```bash
python eval.py --checkpoint model.pt 
``` 
OR
 ```bash
 python eval.py --config  config.yaml
 ```

**Key Parameters:**
- `--kc`: Wave vector (default: 0.5)  
- `--NG`: Number of grid points (default: 32)  
- `--T`: Total simulation time (default: 20)  
- `--dt`: Time step size (default: 0.05)  
- `--Vt`: Thermal velocity (default: 1)  
- `--nParticle`: Number of simulation particles (default: 50)  
- `--Qm`: Charge per mass (default: -1)  
- `--checkpoint`: Model checkpoint file (default: None)  
- `--runId`: Run index for output files (default: 1)  
- `--imgExt`: Image file extension (default: `png`)  
- `--evalDir`: Directory to store evaluation results (default: `eval`)  
- `--dim`: Dimension of the problem (default: 1)
- `--alpha`: Pertubation (default: 0.5)
- `--config`: Configuration file for evaluation parameters (default: None)  


## Benchmark

### Profile

The script allows you to profile training using either Nsight Systems or Torch profile. Additionally, Torch profiler can be used to profile only inference.

To launch Nsight systems do: 

`nsys profile -o profile_report  python fno_profile.py --config=config.yaml`

For distributed training do:

```bash
nsys profile -o nsys_report \
    -t cuda,nvtx,osrt \
    python -m torch.distributed.launch --nproc_per_node=4 fno_profile.py \
        --config=config.yaml
```

Thesea are additional `nsys` args that can be used:
```bash
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
```

To launch Torch profiler do:

`python fno_profile.py --config=config.yaml`

### FLOP/MAC Caculation

The script can calculate FLOP/MAC used by the code using [torchprofile](https://github.com/zhijian-liu/torchprofile) or [calcflops](https://github.com/MrYxJ/calculate-flops.pytorch).

To get FLOPS for custom operations set:
`ENABLE_FLOP_WRAPPERS=1 python calc_flops.py --config=config.yaml`

### Torch Compile

The script benchmarks various compile mode strategies: ['eager', 'default', 'reduce-overhead', 'max-autotune','max-autotune-no-cudagraphs'] using `torch.compile` for training and inference.

To run inference benchmark with all the compile mode strategies do:

`python fno_compile.py --config=config.yaml --compile_eval --input_shape 5 4 256 64`

To run training benchmark with compile mode do:

`python fno_compile.py --config=config.yaml --compile_train --compile_mode=mode`

where `mode` can be any of the strategies mentioned above.

Additionally `--use_amp=1` can be used to train with mixed precision.