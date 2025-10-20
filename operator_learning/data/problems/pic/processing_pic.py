#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[4]
sys.path.append(str(base_path))
import argparse
from operator_learning.utils.misc import readConfig
from operator_learning.data.pic_dataset import createDatasetFromPIC
# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Create training dataset from PIC simulation data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--picFile", default="electrostatic_plasma1D.h5", help="File containing simulation data")
parser.add_argument(
    "--iEnd", default=None, type=int, help="stopping index for weakLandau data")
parser.add_argument(
    "--step", default=1, type=int, help="slice interval")
parser.add_argument(
    "--dataFile", default="dataset.h5", help="name of the dataset HDF5 file")
parser.add_argument(
    "--config", default=None, help="config file, overwriting all parameters specified in it")
args = parser.parse_args()

if args.config is not None:
    config = readConfig(args.config)
    assert "sample" in config, "config file needs a sample section"
    args.__dict__.update(**config.sample)
    if "simu" in config and "picFile" in config.simu:
        args.picFile = config.simu.picFile
        args.nDim = config.simu.nDim
    if "data" in config:
        if "dataFile" in config.data: args.__dict__["dataFile"] = config.data["dataFile"]
kwargs = {**args.__dict__}
kwargs.pop("config")

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
createDatasetFromPIC(**kwargs)