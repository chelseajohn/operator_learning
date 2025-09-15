# Overview

The scripts facilitate training FNO models and comprehensive evaluation of trained models with visualization and analysis capabilities.

## Scripts Description

### `train.py`
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

### `rbc_eval/eval{2d,3d}.py`
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

###  `pic_eval/eval1d.py`
Comprehensive evaluation script for trained FNO models with visualization for PIC in 1D

**Usage:**
```bash
python eval1d.py --checkpoint model.pt 
``` 
OR
 ```bash
 python eval1d.py --config  config.yaml
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
- `--config`: Configuration file for evaluation parameters (default: None)  


