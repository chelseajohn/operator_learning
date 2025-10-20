#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))
import numpy as np

def kinetic(vp: np.ndarray, Q: float, QM: float = -1, wp: float = 1, dim: int = 1) -> float:
    """
    Compute the total kinetic energy of particles.

    Args:
        vp (np.ndarray): Particle velocities.
        Q (float): Particle charge.
        QM (float, optional): Charge-to-mass ratio (q/m). Default is -1.
        wp (float, optional): Particle weight. Default is 1.
        dim (int): Dimension 

    Returns:
        float: Total kinetic energy of the system.
    """
    if dim == 1:
        return np.sum(Q * wp * vp ** 2 * 0.5  / QM)
    else:
        return np.sum(Q * wp * np.sum(vp ** 2, axis=1) * 0.5  / QM)


def potential(rho: np.ndarray, phi: np.ndarray, dx: np.ndarray, dim:int = 1) -> float:
    """
    Compute the total potential energy of the system.

    Args:
        rho (np.ndarray): Charge density at grid points.
        phi (np.ndarray): Electrostatic potential at grid points.
        dx (np.ndarray): Grid spacing.
        dim (int): Dimension

    Returns:
        float: Total potential energy.
    """
    if dim == 1:
        return np.sum(rho * phi * dx[0] / 2)
    else:
        return np.sum(rho * phi * dx[0] * dx[1] / 2)




