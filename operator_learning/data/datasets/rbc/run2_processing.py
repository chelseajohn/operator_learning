#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[4]
sys.path.append(str(base_path))
import argparse
from operator_learning.utils.misc import readConfig
from operator_learning.data.hdf5_dataset import createDataset


# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Create training dataset from Dedalus simulation data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataDir", default="simuData", help="directory containing simulation data")
parser.add_argument(
    "--nDim", default=2, type=int, help="2-D or 3-D")
parser.add_argument(
    "--iBeg", default=0, help="starting index of data sample", type=int)
parser.add_argument(
    "--iEnd", help="stopping index of data sample", type=int)
parser.add_argument(
    "--inSize", default=1, help="input size", type=int)
parser.add_argument(
    "--outStep", default=1, help="output step", type=int)
parser.add_argument(
    "--inStep", default=5, help="input step", type=int)
parser.add_argument(
    "--outType", default="solution", help="output type in the dataset",
    choices=["solution", "update"])
parser.add_argument(
    "--outScaling", default=1, type=float,
    help="scaling factor for the output (ignored with outType=solution !)")
parser.add_argument(
    "--dataFile", default="dataset.h5", help="name of the dataset HDF5 file")
parser.add_argument(
    "--config", default=None, help="config file, overwriting all parameters specified in it")
parser.add_argument(
    "--verbose", action='store_true', help="create dataset with verbose option on")
args = parser.parse_args()


if args.config is not None:
    config = readConfig(args.config)
    assert "sample" in config, "config file needs a sample section"
    args.__dict__.update(**config.sample)
    if "simu" in config and "dataDir" in config.simu:
        args.dataDir = config.simu.dataDir
        args.nSimu = config.simu.nSimu
        args.nDim = config.simu.nDim
    if "data" in config:
        for key in ["outType", "outScaling", "dataFile"]:
            if key in config.data: args.__dict__[key] = config.data[key]
kwargs = {**args.__dict__}
kwargs.pop("config")
if kwargs.get("iEnd", None) is None: kwargs.pop("iEnd", None)

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
createDataset(**kwargs)
