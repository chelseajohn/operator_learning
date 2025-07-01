# Dedalus Simulation & Dataset

This folder provides scripts to:

1. Generate Dedalus-based simulations of Rayleigh-Bénard convection (2D/3D).
2. Process simulation outputs into training datasets.
3. Load and sample datasets for training Fourier Neural Operator (FNO) models
4. Perform spectrum analysis of velocity fields.

---

## Simulation & Processing Scripts

- `run1_simu.py`: Run Dedalus simulations with varying seeds and stores outputs in structured folders.
- `run2_processing.py`: Converts raw simulation data into structured HDF5 datasets.
- `dedalus_simu.py`: Core numerical solver for 2D/3D Rayleigh-Bénard convection.
- `dedalus_prop.py`: Utilities for reading, slicing, and computing spectra from Dedalus outputs.

---

## 1. Simulation Generation

### Usage

```bash
python run1_simu.py --dataDir simuData --nDim 2 --Rayleigh 1e7 --nSimu 1 --dtData 0.1
```

### Key Parameters

| Argument     | Description                              |
|--------------|------------------------------------------|
| `--nDim`     | Spatial dimensionality: 2 or 3           |
| `--Rayleigh` | Rayleigh number for simulation           |
| `--resFactor`| Spatial resolution scaling factor        |
| `--tInit`    | Initialization time before sampling      |
| `--tEnd`     | Time at which simulation ends            |
| `--nSimu`    | Number of different simulations (seeds)  |
| `--dtData`   | Output data time-step                    |

---

## 2. Dataset Creation

### Usage

```bash
python run2_processing.py --dataDir simuData --dataFile dataset.h5
```

### Key Parameters

| Parameter      | Description                                   |
|----------------|-----------------------------------------------|
| `--inSize`     | Number of input steps (currently must be 1)   |
| `--outStep`    | Steps ahead to predict                        |
| `--inStep`     | Step spacing between input samples            |
| `--outType`    | Type of output: `solution` or `update`        |
| `--outScaling` | Scale factor if using update as output        |
| `--dataFile`   | Output HDF5 dataset filename                  |

---
