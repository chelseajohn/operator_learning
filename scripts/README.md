# Overview

The scripts facilitate training FNO models on Rayleigh-Bénard convection and comprehensive evaluation of trained models with visualization and analysis capabilities.

## Scripts Description

### `train.py`
Main training script for 2D/3D Fourier Neural Operator models.

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

### `eval.py`
Comprehensive evaluation script for trained FNO models with detailed analysis and visualization.

**Features:**
- Multi-timestep autoregressive evaluation
- Domain decomposition analysis
- Relative L2 error computation over time
- Contour plots for solution visualization
- Spectral analysis and comparison
- Inference time benchmarking
- Automated report generation in Markdown format

**Usage:**
```bash
python eval.py --checkpoint model.pth --dataFile dataset.h5 --tSteps 10
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

### `eval_template.md`
Markdown template for evaluation report generation. Used internally by `eval.py` to structure the evaluation results.

### Evaluation Outputs (`eval.py`)
```
evalDir/
├── eval_run{runId}.md                    # Evaluation report
├── run{runId}_D{iDec}_error_over_time.png
├── run{runId}_D{iDec}_contour_solution.png
├── run{runId}_D{iDec}_contour_update.png
├── run{runId}_D{iDec}_contour_err.png
├── run{runId}_D{iDec}_contour_ref_solution.png
├── run{runId}_D{iDec}_contour_ref_update.png
├── run{runId}_D{iDec}_spectrum.png
└── run{runId}_D{iDec}_spectrum_HF.png
```

## Dataset Requirements

The scripts work with HDF5 datasets containing:
- `inputs`: Initial conditions for simulations
- `outputs`: Target solutions or updates
- `infos`: Metadata including grid information, timesteps, and dataset parameters
- `xGrid`, `yGrid`: Spatial grid coordinates

