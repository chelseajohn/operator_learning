#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))
import numpy as np
from energy import kinetic
from initial_conditions import findsource

def accelerate(M: np.ndarray, 
               E: np.ndarray,
               Eout: np.ndarray, 
               wp: float, 
               QM: float,
               it: int,
               dim: int):
    """
    Compute particle acceleration from grid electric field and store E at the current timestep.

    Args:
        M (np.ndarray): Projection/interpolation matrix from grid to particle positions.
        E (np.ndarray): Electric field on the grid.
        Eout (np.ndarray): Array to store electric field at each timestep.
        wp (float): Particle weights.
        QM (float): Charge-to-mass ratio (q/m).
        it (int): Current timestep index.
        dim (int): dimension (1D/2D)

    Returns:
        a (np.ndarray): Particle accelerations.
        Eout (np.ndarray): Updated electric field history.
    """
    if dim == 1:
        Etemp = M * E
        a = np.transpose(Etemp) * QM / wp
        Eout[it, :] = Etemp.astype(np.float32)
    else:
        Extemp = M * E[0].flatten()
        Eytemp = M * E[1].flatten()
        a1 = np.transpose(Extemp) * QM / wp
        a2 = np.transpose(Eytemp) * QM / wp
        Eout[it,:,0] = Extemp.astype(np.float32)
        Eout[it,:,1] = Eytemp.astype(np.float32)
        a = np.array([a1, a2])
    
    return a, Eout


def accelerateML(E: np.ndarray, wp: float, QM: float):
    """
    Compute particle acceleration for ML-predicted electric fields.

    Args:
        E (np.ndarray): Electric field at particle positions.
        wp (float): Particle weights.
        QM (float): Charge-to-mass ratio (q/m).

    Returns:
        np.ndarray: Particle accelerations.
    """
    return E * QM / wp


def push(vp: np.ndarray, a: np.ndarray, 
         DT: float, Q: float, 
         QM: float, wp: float,
         it: int):
    """
    Update particle velocities using leapfrog integration and compute kinetic energy.

    Args:
        vp (np.ndarray): Particle velocities.
        a (np.ndarray): Particle accelerations.
        DT (float): Timestep size.
        Q (float): Particle charge.
        QM (float): Charge-to-mass ratio (q/m).
        wp (float): Particle weights.
        it (int): Current timestep index.

    Returns:
        vp_new (np.ndarray): Updated particle velocities.
        kinetic_energy (float): Kinetic energy after update.
    """
    if it == 0:
        return vp + a * DT / 2, kinetic(vp + a * DT / 2, Q, QM, wp)
    else:
        return vp + a * DT, kinetic(vp + a * DT, Q, QM, wp)


def move(xp: np.ndarray, vp: np.ndarray,
        wp: float, DT: float, 
        L: float, it: int = None):
    """
    Update particle positions based on velocities with optional source term.

    Args:
        xp (np.ndarray): Particle positions.
        vp (np.ndarray): Particle velocities.
        wp (float): Particle weights.
        DT (float): Timestep size.
        L (float): Domain length.
        it (int, optional): Current timestep index, used for source term.

    Returns:
        xp_new (np.ndarray): Updated particle positions.
        wp_new (float): Updated particle weights if source term applied.
    """
    if wp == 1:
        return xp + vp * DT, 1
    else:
        return xp + vp * DT, wp + DT * findsource(xp + vp * DT / 2, vp, L, it + 0.5, DT)


def toPeriodic(x: np.ndarray, L: float, discrete: bool=False):
    """
    Apply periodic boundary conditions to particle positions.

    Args:
        x (np.ndarray): Particle positions (or indices).
        L (float): Domain length.
        discrete (bool, optional): Treat positions as discrete indices if True.

    Returns:
        np.ndarray: Particle positions wrapped into [0, L).
    """
    out = (x < 0)
    x[out] = x[out] + L
    if discrete:
        out = (x > L - 1)
    else:
        out = (x >= L)
    x[out] = x[out] - L
    return x

def toPeriodicND(x: np.ndarray, L: float, dim :int=2, discrete: bool=False):
    for i in range(dim):
        x[:,i] = toPeriodic(x[:,i], L[i], discrete)
    return x

def toPeriodicNDTranspose(x: np.ndarray, L: float, dim :int=2, discrete: bool=False):
    for i in range(dim):
        x[i] = toPeriodic(x[i], L[i], discrete)
    return x
