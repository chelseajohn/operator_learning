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
import pyvista as pv
import torch
from timeit import default_timer
import matplotlib.pyplot as plt
from operator_learning.data import HDF5Dataset
from operator_learning.utils.misc import readConfig
from training.train_fno import FourierNeuralOperator
from pySDC.implementations.problem_classes.RayleighBenard3D import RayleighBenard3D


# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Evaluate a model on a given dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataFile", default="dataset.h5", help="name of the dataset HDF5 file")
parser.add_argument(
    "--Rayleigh", default=1e5, type=float, help="Rayeligh Number of the simulation")
parser.add_argument(
    "--Prandtl", default=0.7, type=float, help="Prandtl Number of the simulation")
parser.add_argument(
    "--resFactor", default=32, type=float, help="z=resFactor, x,y = 2*z")
parser.add_argument(
    "--tSteps", default="1",type=int, help="number of autoregressive steps")
parser.add_argument(
    "--model_dt", default="1e-3", type=float, help="model timestep")
parser.add_argument(
    "--checkpoint",help="model checkpoint")
parser.add_argument(
    "--nSamples", default=100, type=int, help="number of simulations to eval with")
parser.add_argument(
    "--imgExt", default="png", help="extension for figure files")
parser.add_argument(
    "--evalDir", default="eval", help="directory to store the evaluation results")
parser.add_argument(
    "--runId", default="1",type=int,  help="run index")
parser.add_argument(
    "--subtitle", default="(64,64,32)",type=str,  help="subtitle for contour plot") 
parser.add_argument(
    "--config", default=None, help="configuration file")
args = parser.parse_args()

if args.config is not None:
    config = readConfig(args.config)
    if "simu" in config:
        args.__dict__.update(**config["simu"])
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
nSamples = args.nSamples
imgExt = args.imgExt
evalDir = args.evalDir
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
tSteps = args.tSteps
model_dt = args.model_dt
subtitle = args.subtitle
run_id = args.runId
nz = args.resFactor
prob = RayleighBenard3D(nx=2*nz, ny=2*nz, nz=nz, Rayleigh=float(args.Rayleigh), Prandtl=float(args.Prandtl), spectral_space=False)  # the spectral_space=False argument means the problem expects input in physical space and returns output in physical space as well
prob.setUpFieldsIO

def norm(x):
    return np.sqrt(np.sum(x**2, axis=(-3, -2, -1)))

def computeError(uPred, uRef):
    diff = norm(uPred-uRef)
    nPred = norm(uPred)
    return diff/nPred

def contourPlot3D(field, x, y, z,
                    title="3D Buoyancy Field",
                    saveFig=None,
                    n_isosurfaces=5,
                    opacity=0.6,
                    cmap="RdBu_r",
                    show=True):
    """
    3D iso-surface visualization with PyVista

    Parameters
    ----------
    field : ndarray
        3D array [nx, ny, nz] of buoyancy (or any scalar).
    x, y, z : 1D arrays
        Grid coordinates along each dimension.
    title : str
        Window/plot title.
    saveFig : str
        If not None, path to save a screenshot.
    n_isosurfaces : int
        Number of positive isosurfaces (total surfaces = 2*n_isosurfaces).
    opacity : float
        Transparency for the surfaces.
    cmap : str
        Colormap.
    show : bool
        Whether to open interactive plot window.
    """

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    grid = pv.StructuredGrid(X, Y, Z)
    grid["field"] = field.flatten(order="F")

    max_val = np.max(np.abs(field))
    isos = np.linspace(max_val / n_isosurfaces, max_val, n_isosurfaces)
    isos = np.concatenate([-isos[::-1], isos])

    # Extract contours
    contour = grid.contour(isosurfaces=isos, scalars="field")

    # Plot
    plotter = pv.Plotter()
    pv.global_theme.colorbar_orientation = 'horizontal'
    scalar_bar_args = {"title":"", "vertical":False, "bold":True, "position_x":0.2, "position_y":0.82, "n_labels":5,
                       "title_font_size": 16, "label_font_size": 16} 
    plotter.add_mesh(contour, cmap=cmap, opacity=opacity, scalar_bar_args=scalar_bar_args)
 
    
    plotter.add_axes()
    plotter.show_axes()
    # plotter.set_scale(xscale=1, yscale=1, zscale=1) 
    plotter.view_isometric()
    plotter.add_title(title, font_size=12)
    plotter.show_grid()
    plotter.show_bounds(
            axes_ranges=(np.min(X),np.max(X),np.min(Y),np.max(Y),np.min(Z),np.max(Z)),
            show_xaxis=True,
            show_yaxis=True,
            show_zaxis=True,
            show_xlabels=True,
            show_ylabels=True,
            show_zlabels=True,
            font_size=16,
            bold=True,
            xtitle='X Axis',
            ytitle='Y Axis',
            ztitle='Z Axis',
            n_xlabels=5,
            n_ylabels=5,
            n_zlabels=3,
            grid=True,
            all_edges=True,
            corner_factor=0.5,
            fmt=None,
            ticks='both',
            location='outer',
            minor_ticks=False,
            padding=0.0,
            use_3d_text=True,
            render=None)

    if saveFig:
        plotter.show(screenshot=saveFig)
    elif show:
        plotter.show()
    else:
        return plotter

HEADER = """
# FNO evaluation on validation dataset

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
model = FourierNeuralOperator(checkpoint=checkpoint, eval_only=True, device=device, data_class='rbc')
os.makedirs(evalDir, exist_ok=True)

nSamplesTotal = dataset.infos["nSamplesTotal"][()]
if nSamples < nSamplesTotal:
    print(f'Evaluating for nSamples: {nSamples}')
else:
    print(f'data file does not have {nSamples}')

indices = slice(nSamplesTotal-nSamples, nSamplesTotal)

# Initial solution for all samples
u0 = dataset.inputs[indices]
print(f'u0: {u0.shape}')
# Reference solution for all samples
uRef = dataset.outputs[indices].copy()
if dataset.outType == "update":
    uRef /= dataset.outScaling
    uRef += u0

# Create summary file, and write header
def fmt(hdfFloat): return float(hdfFloat[()])

summary = open(f"{evalDir}/eval_run{run_id}.md", "w")
summary.write(HEADER.format(
    checkpoint=checkpoint, dataFile=dataFile, nSamples=nSamples,
    dtInput=fmt(dataset.infos["dtInput"]), dtSample=fmt(dataset.infos["dtSample"]),
    outType=dataset.outType, outScaling=dataset.outScaling))

decomps = [
    [(slice(None), slice(None), slice(None))],   # full domain evaluation
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
       return np.sqrt(np.sum(x**2, axis=(-3, -2, -1)))


    def computeError(uPred, uRef):
        diff = norm(uPred-uRef)
        nPred = norm(uPred)
        return diff/nPred

    err = computeError(uPred, uRef)
    errId = computeError(u0, uRef)

    varNames = ["v_x", "v_y", "v_z", "b", "p"]
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
    plt.close()

    avgErr = err.mean(axis=0)
    avgErrId = errId.mean(axis=0)
    errors = pd.DataFrame(data={"model": avgErr, "id": avgErrId}, index=varNames)
    errors.loc["avg"] = errors.mean(axis=0)


    # -------------------------------------------------------------------------
    # -- Contour plots
    # -------------------------------------------------------------------------
    xGrid = dataset.infos["xGrid"][:]
    yGrid = dataset.infos["yGrid"][:]
    zGrid = dataset.infos["zGrid"][:]
    print(f'zGrid: {zGrid[0], zGrid[-1]}')


    uI = u0[0, 3].T
    uM = uPred[0, 3].T
    uR = uRef[0, 3].T

    error_meanxy = np.mean(uPred[0, 3] - uRef[0, 3],axis=(0, 1))  # shape (32,)
    z_levels = np.arange(uRef[0,3].shape[2])  # shape(32)

    errorProfile = f"run{run_id}_D{iDec}_vertical_error_profile.{imgExt}"
    plt.figure(figsize=(10, 5))
    plt.scatter( z_levels[0],error_meanxy[0], color="red", s=100, zorder=5, label="Bottom boundary")
    plt.scatter( z_levels[-1],error_meanxy[-1], color="blue", s=100, zorder=5, label="Top boundary")
    plt.plot(z_levels,error_meanxy, marker='o')  
    # plt.gca().invert_yaxis()
    plt.ylabel("Error (x-y. avg.)")
    plt.xlabel("Vertical Level (z)")
    plt.title("Vertical Error Profile of (Model - pySDC)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'{evalDir}/{errorProfile}')
    plt.close()

    contourPlotSol = f"run{run_id}_D{iDec}_contour_solution.{imgExt}"
    contourPlot3D(
        field=uM, 
        x=xGrid, y=yGrid,z=zGrid,
        title="Model(output): "+subtitle,
        saveFig=f"{evalDir}/{contourPlotSol}")
    
    contourPlotUpdate = f"run{run_id}_D{iDec}_contour_update.{imgExt}"
    contourPlot3D(
        field=uM-uI,
        x=xGrid, y=yGrid,z=zGrid,
        title="Model(update): "+subtitle,
        saveFig=f"{evalDir}/{contourPlotUpdate}")
    
    contourPlotErr = f"run{run_id}_D{iDec}_contour_err.{imgExt}"
    contourPlot3D(
        field=np.abs(uM-uR), 
        x=xGrid, y=yGrid,z=zGrid,
        title="Error: |Model - pySDC|, Grid: "+subtitle,
        saveFig=f"{evalDir}/{contourPlotErr}")

    if iDec == 0:
        contourPlotSolRef = f"run{run_id}_D{iDec}_contour_ref_solution.{imgExt}"
        contourPlot3D(
            field=uR, 
            x=xGrid, y=yGrid,z=zGrid,
            title="Reference: PySDC",
            saveFig=f"{evalDir}/{contourPlotSolRef}")
        
        contourPlotUpdateRef = f"run{run_id}_D{iDec}_contour_ref_update.{imgExt}"
        contourPlot3D(
            field=uR-uI, 
            x=xGrid, y=yGrid,z=zGrid,
            title="Reference: pySDC",
            saveFig=f"{evalDir}/{contourPlotUpdateRef}")
        
    # -------------------------------------------------------------------------
    # -- Averaged Frequency spectrum over Z
    # https://github.com/brownbaerchen/pySDC/blob/24149e8b730f2926869b496d0d9f9c9655652bf2/pySDC/implementations/problem_classes/RayleighBenard3D.py#L378
    # -------------------------------------------------------------------------
    modesRef, spectrumRef = prob.get_frequency_spectrum(uRef[0])
    modesPred, spectrumPred = prob.get_frequency_spectrum(uPred[0])

    sRef_meanZ = spectrumRef.mean(axis=1)
    sPred_meanZ = spectrumPred.mean(axis=1)

    plt.figure(f"D{iDec}_spectrum")
    p = plt.loglog(modesRef, sRef_meanZ[0], '--', label="sx (ref)")
    plt.loglog(modesPred, sPred_meanZ[0], c=p[0].get_color(), label="sx (model)")

    p = plt.loglog(modesRef, sRef_meanZ[1], '--', label="sy (ref)")
    plt.loglog(modesPred, sPred_meanZ[1], c=p[0].get_color(), label="sy (model)")

    plt.legend()
    plt.grid()
    plt.ylabel("spectrum")
    plt.xlabel("wavenumber")
    plt.tight_layout()
    spectrumPlot = f"run{run_id}_D{iDec}_spectrum.{imgExt}"
    plt.savefig(f"{evalDir}/{spectrumPlot}")
    plt.close()

    # -------------------------------------------------------------------------
    # -- Compute Nusselt Number
    # https://github.com/brownbaerchen/pySDC/blob/24149e8b730f2926869b496d0d9f9c9655652bf2/pySDC/implementations/problem_classes/RayleighBenard3D.py#L304
    # -------------------------------------------------------------------------

    nusRef = []
    nusPred = []
    for sample in range(len(uRef)):
        NuRef = prob.compute_Nusselt_numbers(uRef[sample])   # Nu: {'V': .., 't': .., 'b': ..}
        NuPred = prob.compute_Nusselt_numbers(uPred[sample])
        nusRef.append(list(NuRef.values()))
        nusPred.append(list(NuPred.values()))
    nusPred = np.array(nusPred)  # [nSamples,3]
    nusRef = np.array(nusRef)
    
    plt.figure(f"D{iDec}_Nu_v")
    n = plt.plot(nusRef[:,0], '--', label=r"$Nu_v$ (ref)")
    plt.plot(nusPred[:,0], c=n[0].get_color(), label=r"$Nu_v$ (model)")

    n = plt.plot(nusRef[:,1], '--', label=r"$Nu_t$ (ref)")
    plt.plot(nusPred[:,1], c=n[0].get_color(), label=r"$Nu_t$ (model)")

    n = plt.plot(nusRef[:,2], '--', label=r"$Nu_b$ (ref)")
    plt.plot(nusPred[:,2], c=n[0].get_color(), label=r"$Nu_b$ (model)")

    plt.legend(fontsize='x-small')
    plt.grid()
    plt.ylabel("Nu")
    plt.xlabel("time")
    plt.tight_layout()
    nusPlot = f"run{run_id}_D{iDec}_nusPlot.{imgExt}"
    plt.savefig(f"{evalDir}/{nusPlot}")
    plt.close()

    # -------------------------------------------------------------------------
    # -- Write slices evaluation in summary
    # -------------------------------------------------------------------------
    if errorProfile is not None:
        TEMPLATE += f"Vertical Error Profile :\n- [Vertical Error Profile]({errorProfile})\n\n"
    if nusPlot is not None:
        TEMPLATE += f"Nusselt Number :\n- [Nusselt Number Plot]({nusPlot})\n"
    summary.write(TEMPLATE.format(
        iDec=iDec,
        device=device_name,
        slices=str([(sliceToStr(sX), sliceToStr(sY), sliceToStr(sZ)) for sX, sY,sZ in slices]).replace("'", ""),
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
        errorProfile=errorProfile,
        spectrumPlot=spectrumPlot,
        nusPlot=nusPlot
        ))
    

summary.close()


