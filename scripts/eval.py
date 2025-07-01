#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))

import argparse
import numpy as np
import pandas as pd
import torch
from timeit import default_timer
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from operator_learning.data.datasets.rbc.dedalus_prop import computeMeanSpectrum, getModes
from operator_learning.data import HDF5Dataset
from operator_learning.utils.misc import readConfig
from training.train_fno import FourierNeuralOperator

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Evaluate a model on a given dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataFile", default="dataset.h5", help="name of the dataset HDF5 file")
parser.add_argument(
    "--tSteps", default="1",type=int, help="number of autoregressive steps")
parser.add_argument(
    "--model_dt", default="1e-3", type=float, help="model timestep")
parser.add_argument(
    "--checkpoint",help="model checkpoint")
parser.add_argument(
    "--iSimu", default=8, type=int, help="index of the simulation to eval with")
parser.add_argument(
    "--imgExt", default="png", help="extension for figure files")
parser.add_argument(
    "--evalDir", default="eval", help="directory to store the evaluation results")
parser.add_argument(
    "--runId", default="1",type=int,  help="run index")
parser.add_argument(
    "--subtitle", default="(256,64)",type=str,  help="subtitle for contour plot") 
parser.add_argument(
    "--config", default=None, help="configuration file")
args = parser.parse_args()

if args.config is not None:
    config = readConfig(args.config)
    if "eval" in config:
        args.__dict__.update(**config["eval"])
    if "data" in config and "dataFile" in config["data"]:
        args.dataFile = config.data.dataFile
    if "train" in config and "checkpoint" in config["train"]:
        args.checkpoint = config.train.checkpoint
        if "trainDir" in config.train:
            FourierNeuralOperator.TRAIN_DIR = config.train.trainDir

dataFile = args.dataFile
checkpoint = args.checkpoint
iSimu = args.iSimu
imgExt = args.imgExt
evalDir = args.evalDir
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
tSteps = args.tSteps
model_dt = args.model_dt
subtitle = args.subtitle
run_id = args.runId

def contourPlot(field, refField,
                 x, y, title=None, refTitle=None, 
                 saveFig=False, closeFig=True, plot_refField=False,
                 error=False, refScales=False, time=None):
    
    
    fig, axs = plt.subplots(2 if plot_refField else 1)
    ax = axs[0] if plot_refField else axs
    scales =  (np.min(refField), np.max(refField))

    def setup(ax):
        ax.axis('on')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    def setColorbar(field, im, ax, scales, refScales):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if refScales:
            im.cmap.set_under("white")
            im.cmap.set_over("white")
            im.set_clim(*scales)
            fig.colorbar(im, cax=cax, ax=ax, ticks=np.linspace(*scales, 3))
        else:
            fig.colorbar(im, cax=cax, ax=ax, ticks=np.linspace(np.min(field), np.max(field), 3))

    timeSuffix = f' at t = {np.round(time,3)}s' if time is not None else ''
  
    im0 = ax.pcolormesh(x, y, field)
    setColorbar(field, im0,ax, scales, refScales)
    ax.set_title(f'{title}{timeSuffix}', fontsize=14)
    setup(ax)
    if plot_refField:
        im2 = axs[1].pcolormesh(x, y, refField)
        setColorbar(refField, im2, axs[1], scales, refScales)
        axs[1].set_title(f'{refTitle}{timeSuffix}',fontsize=14)
        setup(axs[1])
    
    fig.tight_layout()
    if saveFig:
        plt.savefig(saveFig, bbox_inches='tight', pad_inches=0.05)
    if closeFig:
        plt.close(fig)

def norm(x):
        return np.linalg.norm(x, axis=(-2, -1))

def computeError(uPred, uRef):
    diff = norm(uPred-uRef)
    nPred = norm(uPred)
    return diff/nPred


HEADER = """
# FNO evaluation on validation dataset

- simulation index: {iSimu}
- model name: {checkpoint}
- dataset : {dataFile}
    - nSamples : {nSamples}
    - dtInput (between input and output of the model) : {dtInput}
    - dtSample (between two samples) : {dtSample}
    - outType : {outType}
    - outScaling : {outScaling}

"""
op = os.path
with open(op.dirname(op.abspath(op.realpath(__file__)))+"/eval_template.md") as f:
    TEMPLATE = f.read()

def sliceToStr(s:slice):
    out = ":"
    if s.start is not None:
        out = str(s.start)+out
    if s.stop is not None:
        out = out+str(s.stop)
    return out

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
dataset = HDF5Dataset(dataFile)
model = FourierNeuralOperator(checkpoint=checkpoint, eval_only=True)
os.makedirs(evalDir, exist_ok=True)

nSamples = dataset.infos["nSamples"][()]
print(f'nSamples: {nSamples}')
nSimu = dataset.infos["nSimu"][()]
assert iSimu < nSimu, f"cannot evaluate with iSimu={iSimu} with only {nSimu} simu"
indices = slice(iSimu*nSamples, (iSimu+1)*nSamples)

# Initial solution for all samples
u0 = dataset.inputs[indices]

# Reference solution for all samples
uRef = dataset.outputs[indices].copy()
if dataset.outType == "update":
    uRef /= dataset.outScaling
    uRef += u0

# Create summary file, and write header
def fmt(hdfFloat): return float(hdfFloat[()])

summary = open(f"{evalDir}/eval_run{run_id}.md", "w")
summary.write(HEADER.format(
    iSimu=iSimu, checkpoint=checkpoint, dataFile=dataFile, nSamples=nSamples,
    dtInput=fmt(dataset.infos["dtInput"]), dtSample=fmt(dataset.infos["dtSample"]),
    outType=dataset.outType, outScaling=dataset.outScaling))

decomps = [
    [(slice(None), slice(None))],   # full domain evaluation

    # [(slice(0, 64), slice(None)),   # 4 domains distributed in X direction
    #  (slice(64, 128), slice(None)),
    #  (slice(128, 192), slice(None)),
    #  (slice(192, 256), slice(None))],

    # [(slice(None), slice(0, 32)),   # 2 domains distributed in Z direction
    #  (slice(None), slice(32, 64))],

    # [(slice(0, 64), slice(0,32)),     # 4 domains distributed in X & y direction
    #  (slice(0, 64), slice(32,64)),
    #  (slice(64, 128), slice(0,32)),
    #  (slice(64, 128), slice(32,64)),
    #  (slice(128, 192), slice(0,32)),
    #  (slice(128, 192), slice(32,64)),
    #  (slice(192, 256), slice(0,32)),
    #  (slice(192, 256), slice(32,64))],
    ]

for iDec in range(len(decomps)):
    slices = decomps[iDec]
    time = []
    uPred = np.zeros_like(uRef)
    _ = slice(None)
    print(f"Computing {tSteps}-Step prediction for D{iDec} with dt={model_dt}")
    input = u0
    for t in range(1,tSteps+1):
        for j, s in enumerate(slices):
            print(f" -- slice {j+1}/{len(slices)}")
            start_inference = default_timer()
            uPred[(_, _, *s)] = model(input[(_, _, *s)])
            stop_inference = default_timer() - start_inference
            time.append(stop_inference)
        input = uPred
    inferenceTime = np.round(sum(time),3)
    avg_inferenceTime = np.round(sum(time)/len(time),3)
    print(" -- done !")
    print(f'-- slices: {slices}')
    print(f"-- Avg inference time on {device_name} : {avg_inferenceTime}")
    print(f"-- Total inference time on {device_name} for {tSteps} : {inferenceTime}")
    print(f"-- Inference after (tSteps x dt)(s): {tSteps} x {model_dt}")
    

    # -------------------------------------------------------------------------
    # -- Relative error over time
    # -------------------------------------------------------------------------
    def norm(x):
        return np.linalg.norm(x, axis=(-2, -1))

    def computeError(uPred, uRef):
        diff = norm(uPred-uRef)
        nPred = norm(uPred)
        return diff/nPred

    err = computeError(uPred, uRef)
    errId = computeError(u0, uRef)

    varNames = ["v_x", "v_z", "b", "p"]
    fig = plt.figure(f"D{iDec}_error over time")
    for name, e, eId in zip(varNames, err.T, errId.T):
        p = plt.semilogy(e, '-', label=name, markevery=0.2)
        plt.semilogy(eId, '--', c=p[0].get_color())
    plt.legend()
    plt.grid(True)
    plt.xlabel("samples ordered with time")
    plt.ylabel("relative $L_2$ error")
    fig.set_size_inches(10, 5)
    plt.tight_layout()
    errorPlot = f"run{run_id}_D{iDec}_error_over_time.{imgExt}"
    plt.savefig(f"{evalDir}/{errorPlot}")

    avgErr = err.mean(axis=0)
    avgErrId = errId.mean(axis=0)
    errors = pd.DataFrame(data={"model": avgErr, "id": avgErrId}, index=varNames)
    errors.loc["avg"] = errors.mean(axis=0)


    # -------------------------------------------------------------------------
    # -- Contour plots
    # -------------------------------------------------------------------------
    xGrid = dataset.infos["xGrid"][:]
    yGrid = dataset.infos["yGrid"][:]

    uI = u0[0, 2].T
    uM = uPred[0, 2].T
    uR = uRef[0, 2].T

    contourPlotSol = f"run{run_id}_D{iDec}_contour_solution.{imgExt}"
    contourPlot(
        field=uM, 
        x=xGrid, y=yGrid,
        title="Model(output): "+subtitle,
        refField=uR,
        refTitle=None,
        plot_refField=False,
        saveFig=f"{evalDir}/{contourPlotSol}", 
        refScales=True, 
        closeFig=True)
    
    contourPlotUpdate = f"run{run_id}_D{iDec}_contour_update.{imgExt}"
    contourPlot(
        field=uM-uI,
        refField=uR-uI,
        x=xGrid, y=yGrid,
        title="Model(update): "+subtitle,
        refTitle=None,
        plot_refField=False,
        saveFig=f"{evalDir}/{contourPlotUpdate}", 
        refScales=True, 
        closeFig=True)
    
    contourPlotErr = f"run{run_id}_D{iDec}_contour_err.{imgExt}"
    contourPlot(
        field=np.abs(uM-uR), 
        refField=None, 
        plot_refField=False,
        x=xGrid, y=yGrid,
        title="Error: |Model - Dedalus|, Grid: "+subtitle,
        saveFig=f"{evalDir}/{contourPlotErr}",
        closeFig=True)

    if iDec == 0:
        contourPlotSolRef = f"run{run_id}_D{iDec}_contour_ref_solution.{imgExt}"
        contourPlot(
            field=uR, 
            refField=None, 
            plot_refField=False,
            x=xGrid, y=yGrid,
            title="Reference: Dedalus",
            saveFig=f"{evalDir}/{contourPlotSolRef}",
            closeFig=True)
        
        contourPlotUpdateRef = f"run{run_id}_D{iDec}_contour_ref_update.{imgExt}"
        contourPlot(
            field=uR-uI, 
            refField=None, 
            plot_refField=False,
            x=xGrid, y=yGrid,
            title="Reference: Dedalus",
            saveFig=f"{evalDir}/{contourPlotUpdateRef}",
            closeFig=True)

    # -------------------------------------------------------------------------
    # -- Averaged spectrum
    # -------------------------------------------------------------------------
    sxRef, szRef = computeMeanSpectrum(uRef)
    sxPred, szPred = computeMeanSpectrum(uPred)
    k = getModes(dataset.grid[0])

    plt.figure(f"D{iDec}_spectrum")
    p = plt.loglog(k, sxRef.mean(axis=0), '--', label="sx (ref)")
    plt.loglog(k, sxPred.mean(axis=0), c=p[0].get_color(), label="sx (model)")

    p = plt.loglog(k, szRef.mean(axis=0), '--', label="sz (ref)")
    plt.loglog(k, szPred.mean(axis=0), c=p[0].get_color(), label="sz (model)")

    plt.legend()
    plt.grid()
    plt.ylabel("spectrum")
    plt.xlabel("wavenumber")
    plt.ylim(bottom=1e-10)
    plt.tight_layout()
    spectrumPlot = f"run{run_id}_D{iDec}_spectrum.{imgExt}"
    plt.savefig(f"{evalDir}/{spectrumPlot}")

    plt.xlim(left=50)
    plt.ylim(top=1e-5)
    spectrumPlotHF = f"run{run_id}_D{iDec}_spectrum_HF.{imgExt}"
    plt.savefig(f"{evalDir}/{spectrumPlotHF}")


    # -------------------------------------------------------------------------
    # -- Write slices evaluation in summary
    # -------------------------------------------------------------------------
    summary.write(TEMPLATE.format(
        iDec=iDec,
        device=device_name,
        slices=str([(sliceToStr(sX), sliceToStr(sZ)) for sX, sZ in slices]).replace("'", ""),
        errorPlot=errorPlot,
        errors=errors.to_markdown(floatfmt="1.1e"),
        avg_inferenceTime=avg_inferenceTime,
        tSteps=tSteps,
        dt=model_dt,
        inferenceTime=inferenceTime,
        contourPlotSol=contourPlotSol,
        contourPlotUpdate=contourPlotUpdate,
        contourPlotErr=contourPlotErr,
        contourPlotSolRef=contourPlotSolRef,
        contourPlotUpdateRef=contourPlotUpdateRef,
        spectrumPlot=spectrumPlot,
        spectrumPlotHF=spectrumPlotHF
        ))

summary.close()


