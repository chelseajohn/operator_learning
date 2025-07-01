# Operator Learning

---

This repository contains implementations, experimental scripts, and documentation related to **operator learning**—a machine learning paradigm focused on learning mappings between infinite-dimensional function spaces, such as solving parametric partial differential equations (PDEs) using neural operators like the [Fourier Neural Operator (FNO)](https://arxiv.org/pdf/2010.08895).

---

## Installation

### ✅ Base Environment

- Python: 3.12.3  
- OpenMPI: 5.0.5  
- CUDA: 12.6  
- PyTorch: 2.5.1+cu124  
- FFTW: 3.3.10  
- Python dependencies listed in [requirements.txt](requirements.txt)

### 🔧 Setup Instructions

```bash
git clone https://github.com/chelseajohn/operator_learning.git
cd operator_learning
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
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

You can launch distributed jobs using `torchrun` within a SLURM environment:

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

## Problems Solved

- **Rayleigh-Bénard convection** in 2D using FNO, with datasets generated via the pseudo-spectral solver [Dedalus](https://dedalus-project.readthedocs.io/en/latest/).