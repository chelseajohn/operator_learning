#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

import argparse
import torch
import numpy as np
import cupy as cp

from operator_learning.utils.misc import readConfig
from training.train_fno import FourierNeuralOperator
from pic_plotter import PICVisualizer

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Evaluate a PIC FNO model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--kc", default="0.5", type=float, help="wave vector")
parser.add_argument(
    "--NG", default="32", type=int, help="number of grid points")
parser.add_argument(
    "--T", default="20", type=float, help="Time")
parser.add_argument(
    "--dt", default="0.05", type=float, help="timestep")
parser.add_argument(
    "--alpha", default="0.05", type=float, help="pertubation")
parser.add_argument(
    "--Vt", default=1, type=float, help="thermal velocity")
parser.add_argument(
    "--nParticle", default="50", type=int, help="number of simulation particles")
parser.add_argument(
    "--Qm", default="-1", type=float, help="charge per mass")
parser.add_argument(
    "--checkpoint", default=None, help="model checkpoint")
parser.add_argument(
    "--runId", default="1",type=int,  help="run index")
parser.add_argument(
    "--imgExt", default="png", help="extension for figure files")
parser.add_argument(
    "--evalDir", default="eval", help="directory to store the evaluation results")
parser.add_argument(
    "--dim", default="1", type=int, help="dimension")
parser.add_argument(
    "--predOnly", action="store_true", help="Perform only ML predictions without reference results")
parser.add_argument(
    "--testCase", default="strongLandau", help="Choose the test case among weakLandau, strongLandau, tsi or bti")
parser.add_argument(
    "--config", default=None, help="configuration file")
args = parser.parse_args()


if args.config is not None:
    config = readConfig(args.config)
    if "eval" in config:
        args.__dict__.update(**config["eval"])
    if "train" in config and "checkpoint" in config["train"]:
        args.checkpoint = config.train.checkpoint
        if "trainDir" in config.train:
            FourierNeuralOperator.TRAIN_DIR = config.train.trainDir

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
checkpoint = args.checkpoint
dim = args.dim
predOnly = args.predOnly
testCase = args.testCase

seed = 152
torch.manual_seed(seed)
np.random.seed(seed)
cp.random.seed(seed)
torch.cuda.manual_seed_all(seed)
vis = PICVisualizer(args)

if checkpoint is not None:
    fno_model = FourierNeuralOperator(checkpoint=checkpoint, eval_only=True, device=device, data_class='pic')
    if dim == 1:
        posPred, velPred, wPred, EnergyPred, EkPred, EpPred, pPred, ExpPred, timePred = vis.pic1D(ml_acc=True, model=fno_model, data_file=config.data.dataFile)
        #phase_spacePred = vis.phase_space(xp=posPred, vp=velPred, wp=wPred, ml_acc=True)
        phase_spacePred = None
    else:
        posPred, velPred, wPred, EnergyPred, EkPred, EpPred, pPred, ExpPred, EypPred, timePred = vis.pic2D(ml_acc=True, model=fno_model, data_file=config.data.dataFile)
        phase_spacePred = None

else:
    EnergyPred = None
    EkPred = None
    EpPred = None
    EPred = None
    pPred = None
    timePred = None
    phase_spacePred = None
    growth_ratePred = None
    speedup = 1

if predOnly is False:
    if dim == 1:
        posRef, velRef, wRef, EnergyRef, EkRef, EpRef, pRef, ExpRef, timeRef = vis.pic1D(ml_acc=False)
        #phase_spaceRef = vis.phase_space(xp=posRef, vp=velRef, wp=wRef, ml_acc=False)
        phase_spaceRef = None
    else:
        posRef, velRef, wRef, EnergyRef, EkRef, EpRef, pRef, ExpRef, EypRef, timeRef = vis.pic2D(ml_acc=False)
        phase_spaceRef = None
else:
    EnergyRef = None
    EkRef = None
    EpRef = None
    ERef = None
    pRef = None
    timeRef = None
    phase_spaceRef = None
    growth_rateRef = None
    speedup = 1

#growth_rate = vis.twoStreamIppl(ExRef=ERef, ExPred=EPred)  
energy = vis.energy(ERef=EnergyRef, EPred=EnergyPred, EkRef=EkRef, EkPred=EkPred, EpRef=EpRef, EpPred=EpPred)
conserv_error = vis.conservation_errors(ERef=EnergyRef, EPred=EnergyPred, pRef=pRef, pPred=pPred)
if dim == 1:
    landau_decay = vis.landau_decay(Ex=ExpRef, ExPred=ExpPred, label='strong')
else:
    if ((testCase == "weakLandau") or (testCase == "strongLandau")):
        landau_decay = vis.landau_decay(Ex=ExpRef, ExPred=ExpPred, Ey=EypRef, EyPred=EypPred, label=testCase)
    elif ((testCase == "tsi") or (testCase == "bti")):
        growth_rate = vis.instability(Ex=ExpRef, ExPred=ExpPred, Ey=EypRef, EyPred=EypPred, label=testCase)

HEADER = """
# FNO evaluation for PIC in {dim}D on {device}
## Simulation Configuration

| Parameter    | Value   |
|--------------|---------|
{rows}
"""

# Convert dict into Markdown table rows
rows = "\n".join([f"| {k:<12} | {v} |" for k, v in args.__dict__.items()])


op = os.path
with open(op.dirname(op.abspath(op.realpath(__file__)))+"/eval_template.md") as f:
    TEMPLATE = f.read()

summary = open(f"{args.evalDir}/eval_run{args.runId}.md", "w")
summary.write(HEADER.format(dim=dim, device=device_name, rows=rows))

if phase_spaceRef is not None:
    TEMPLATE += f"- [Phase space Ref]({phase_spaceRef})\n"
if phase_spacePred is not None:
    TEMPLATE += f"- [Phase space Pred]({phase_spacePred})\n"

TEMPLATE  += f"\nAverage time for Accleration per timestep in PIC (millisec): {timeRef}\n"

if timePred is not None and timeRef is not None:
    speedup = round(timeRef/timePred,3)
    TEMPLATE += f"Average Inference time for Accleration using FNO (millisec): {timePred}\n"
    TEMPLATE += f"Speed up PIC/FNO: {speedup}\n"
                
summary.write(TEMPLATE.format(
        dim=dim,
        device=device,
        energy=energy,
        conserv_errors=conserv_error,
        landau_decay=None,
        phase_spaceRef=phase_spaceRef,
        phase_spacePred=phase_spacePred,
        growth_rate=None,
        timeRef=timeRef,
        timePred=timePred,
        speedup=speedup
        ))
summary.close()
