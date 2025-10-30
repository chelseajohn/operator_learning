# Operator Learning

---

This repository contains implementations, experimental scripts, and documentation related to **operator learning**—a machine learning paradigm focused on learning mappings between infinite-dimensional function spaces, such as solving parametric partial differential equations (PDEs) using neural operators like the [Fourier Neural Operator (FNO)](https://arxiv.org/pdf/2010.08895).

---

## Installation

### ✅ Base Environment

- Python 
- OpenMPI
- CUDA 
- PyTorch  
- Python dependencies listed in [requirements.txt](requirements.txt)

### 🔧 Setup Instructions

```bash
git clone https://github.com/chelseajohn/operator_learning.git
cd operator_learning
python3 -m venv venv
source venv/bin/activate
python -m pip install -e .
```

---

## 🚀 Distributed Data Parallel (DDP) Training

Model training can be accelerated using **Distributed Data Parallel (DDP)**, implemented with [PyTorch DDP](https://pytorch.org/docs/stable/notes/ddp.html#distributed-data-parallel). The model is replicated across GPUs, trained in parallel on different data samples, and synchronized at each step.

### 🔧 Enable DDP in Configuration

Edit your `config.yaml` to include:

```yaml
parallel_strategy:
  ddp: True              # Enable Distributed Data Parallel
  gpus_per_node: 4       # Number of GPUs per node
```

### 🖥️ Launching DDP Jobs via SLURM

Distributed jobs can be launched using `torchrun` within a SLURM environment:

```bash
##### Network parameters #####
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
GPUS_PER_NODE=4

srun python -u -m torch.distributed.run   \
    --nproc_per_node $GPUS_PER_NODE   \
    --nnodes $SLURM_JOB_NUM_NODES  \
    --node_rank $SLURM_PROCID   \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d    \
    --rdzv_conf=is_host=$(if ((SLURM_NODEID)); then echo False; else echo True; fi)  \
    --max_restarts 0     \
    --tee 3 \
    $python_file  
```

## 🔍 Profiling 

To enable profiling using Nsight systems or Torch profiler add a `profile` section into the config:

```bash
profile:
  enableProfiler: True
  profiler: nsys               # or torch
  profileDir: profileLog
  evalOnly: false             # profile only inference using torch profiler 
  input_shape: [1,3,10000]    # relevant only if eval_only=true
  use_amp: 1                  # enable mixed precision
  benchmark: true             # run profiling in benchmark mode
  compile_eval: 1             # use torch.compile on model inference
  compile_train: 0            # use torch.compile on training (set to 1, if compile_eval=0)
  comipile_mode: default      # compiler mode 
```

## Problems Solved

- **Rayleigh-Bénard convection** in 2D/3D using FNO, with datasets generated via the pseudo-spectral solver [Dedalus](https://dedalus-project.readthedocs.io/en/latest/) for 2D and [pySDC](https://zenodo.org/records/15196003) for 3D.
- **Particle-In-Cell** (1D/2D) algorithm using FNO for electric field prediction