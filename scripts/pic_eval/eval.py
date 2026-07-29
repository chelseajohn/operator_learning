#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
from textwrap import dedent
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
    "--ref", default="pic", help="Choose the reference numerical scheme pic or pif")
parser.add_argument(
    "--model_dtype", type=str, default="float32", 
    help="Model dtype for layers except FNO_DSE layer, options['float32', 'float64'] ")
parser.add_argument(
    "--fno_dtype", type=str, default="float32", 
    help="FNO_DSE Layer dtype, options['float32', 'float64'] ")
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
if args.model_dtype == 'float32':
    model_dtype = torch.float32
else:
    model_dtype = torch.float64
if args.fno_dtype == 'float32':
    fno_dtype = torch.float32
else:
    fno_dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
checkpoint = args.checkpoint
dim = args.dim
predOnly = args.predOnly
seed = 152
torch.manual_seed(seed)
np.random.seed(seed)
cp.random.seed(seed)
torch.cuda.manual_seed_all(seed)

for tc in config["testCases"]:
    args.__dict__.update(**config["testCases"][tc])
    args.evalDir = f"{config.eval.evalDir}/{tc}"
    args.testCase = tc
    vis = PICVisualizer(args)

    if checkpoint is not None:
        fno_model = FourierNeuralOperator(checkpoint=checkpoint, eval_only=True, device=device, data_class='pic', model_dtype=model_dtype, fno_dtype=fno_dtype)
        posPred, velPred, wPred, EnergyPred, EkPred, EpPred, pPred, ExpPred, EypPred, EzpPred, timePred = vis.picND(ml_acc=True, model=fno_model, data_file=config.data.dataFile)
        phase_spacePred = None

    else:
        EnergyPred = None
        EkPred = None
        EpPred = None
        EPred = None
        pPred = None
        ExpPred = None
        EypPred = None
        EzpPred = None
        timePred = None
        phase_spacePred = None
        growth_ratePred = None
        speedup = 1

    if predOnly is False:
        posRef, velRef, wRef, EnergyRef, EkRef, EpRef, pRef, ExpRef, EypRef, EzpRef, timeRef = vis.picND(ml_acc=False)
        phase_spaceRef = None
    else:
        EnergyRef = None
        EkRef = None
        EpRef = None
        ERef = None
        pRef = None
        ExpRef = None
        EypRef = None
        EzpRef = None
        timeRef = None
        phase_spaceRef = None
        growth_rateRef = None
        speedup = 1

    energy = vis.energy(ERef=EnergyRef, EPred=EnergyPred, EkRef=EkRef, EpRef=EpRef, EkPred=EkPred, EpPred=EpPred)
    conserv_error = vis.conservation_errors(ERef=EnergyRef, EPred=EnergyPred, pRef=pRef, pPred=pPred)
    if ((args.testCase == "weakLandau") or (args.testCase == "strongLandau")):
        landau_decay = vis.landau_decay(Ex=ExpRef, ExPred=ExpPred, Ey=EypRef, EyPred=EypPred, Ez=EzpRef, EzPred=EzpPred, label=args.testCase)
    elif ((args.testCase == "tsi") or (args.testCase == "bti")):
        growth_rate = vis.instability(Ex=ExpRef, ExPred=ExpPred, Ey=EypRef, EyPred=EypPred, Ez=EzpRef, EzPred=EzpPred, label=args.testCase)
    
    HEADER = dedent("""
    # FNO evaluation for PIC in {dim}D on {device}

    ## Simulation Configuration

    | Parameter | Value |
    |-----------|-------|
    {rows}
    """)

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
