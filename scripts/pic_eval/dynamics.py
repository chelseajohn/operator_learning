#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))
import numpy as np
import cupy as cp
from energy import kinetic
from initial_conditions import findsource

def accelerate(M: cp.ndarray, 
               E: cp.ndarray,
               wp: float, 
               QM: float,
               it: int,
               dim: int):
    """
    Compute particle acceleration from grid electric field and store E at the current timestep.

    Args:
        M (cp.ndarray): Projection/interpolation matrix from grid to particle positions.
        E (cp.ndarray): Electric field on the grid.
        Eout (cp.ndarray): Array to store electric field at each timestep.
        wp (float): Particle weights.
        QM (float): Charge-to-mass ratio (q/m).
        it (int): Current timestep index.
        dim (int): dimension (1D/2D)

    Returns:
        a (cp.ndarray): Particle accelerations.
        Eout (cp.ndarray): Updated electric field history.
    """
    if dim == 1:
        Etemp = M * E
        a = cp.transpose(Etemp) * QM / wp
        Eout = Etemp.astype(cp.float32)
    else:
        Extemp = M * E[0].flatten()
        Eytemp = M * E[1].flatten()
        a1 = cp.transpose(Extemp) * QM / wp
        a2 = cp.transpose(Eytemp) * QM / wp
        #Eout[it,:,0] = Extemp.astype(cp.float32)
        #Eout[it,:,1] = Eytemp.astype(cp.float32)
        a = cp.array([a1, a2])
        Eout = cp.zeros([2, a.shape[1]])
        Eout[0,:] = Extemp.astype(cp.float32) 
        Eout[1,:] = Eytemp.astype(cp.float32)
    
    return a, Eout


def accelerateML(E: cp.ndarray, wp: float, QM: float):
    """
    Compute particle acceleration for ML-predicted electric fields.

    Args:
        E (cp.ndarray): Electric field at particle positions.
        wp (float): Particle weights.
        QM (float): Charge-to-mass ratio (q/m).

    Returns:
        cp.ndarray: Particle accelerations.
    """
    a = E * QM / wp
    
    return a 


def push(vp: cp.ndarray, a: cp.ndarray, 
         DT: float, Q: float, 
         QM: float, wp: float,
         it: int):
    """
    Update particle velocities using leapfrog integration and compute kinetic energy.

    Args:
        vp (cp.ndarray): Particle velocities.
        a (cp.ndarray): Particle accelerations.
        DT (float): Timestep size.
        Q (float): Particle charge.
        QM (float): Charge-to-mass ratio (q/m).
        wp (float): Particle weights.
        it (int): Current timestep index.

    Returns:
        vp_new (cp.ndarray): Updated particle velocities.
        kinetic_energy (float): Kinetic energy after update.
    """
    if it == 0:
        return vp + a * DT / 2, kinetic(vp + a * DT / 2, Q, QM, wp)
    else:
        return vp + a * DT, kinetic(vp + a * DT, Q, QM, wp)


def move(xp: cp.ndarray, vp: cp.ndarray,
        wp: float, DT: float, 
        L: float, it: int = None):
    """
    Update particle positions based on velocities with optional source term.

    Args:
        xp (cp.ndarray): Particle positions.
        vp (cp.ndarray): Particle velocities.
        wp (float): Particle weights.
        DT (float): Timestep size.
        L (float): Domain length.
        it (int, optional): Current timestep index, used for source term.

    Returns:
        xp_new (cp.ndarray): Updated particle positions.
        wp_new (float): Updated particle weights if source term applied.
    """
    if wp == 1:
        return xp + vp * DT, 1
    else:
        return xp + vp * DT, wp + DT * findsource(xp + vp * DT / 2, vp, L, it + 0.5, DT)


def toPeriodic(x: cp.ndarray, L: float, discrete: bool=False):
    """
    Apply periodic boundary conditions to particle positions.

    Args:
        x (cp.ndarray): Particle positions (or indices).
        L (float): Domain length.
        discrete (bool, optional): Treat positions as discrete indices if True.

    Returns:
        cp.ndarray: Particle positions wrapped into [0, L).
    """
    out = (x < 0)
    x[out] = x[out] + L
    if discrete:
        out = (x > L - 1)
    else:
        out = (x >= L)
    x[out] = x[out] - L
    return x

def toPeriodicNDOld(x: cp.ndarray, L: float, dim :int=2):
    for i in range(dim):
        x[i] = toPeriodic(x[i], L[i])
    return x

def toPeriodicND(x: cp.ndarray, L: float, dim :int=2):
    x = cp.mod(x, cp.asarray(L)[:, None])
    return x
