#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))
import numpy as np
from scipy import sparse
import finufft
from dynamics import toPeriodic

def interpMatrix(XP: np.ndarray, wp: float, DX: float, N: int, NG: int, p: np.ndarray) -> sparse.csr_matrix:
    """
    Construct the projection (interpolation) matrix from particles to grid.

    Args:
        XP (np.ndarray): Particle positions (1D array, shape: [N]).
        wp (float): Particle weights.
        DX (float): Grid spacing.
        N (int): Number of particles.
        NG (int): Number of grid points.
        p (np.ndarray): Particle indices (0..N-1).

    Returns:
        scipy.sparse.csr_matrix: Sparse interpolation matrix of shape (N, NG).

    Notes:
        - Uses quadratic (3-point) weighting to distribute particle quantities to the grid.
        - Applies periodic boundary conditions on grid indices.
        - Useful for projecting charge, current, or other particle quantities to a uniform grid.
    """
    g1 = np.floor(XP / DX).astype(int)          # primary grid index
    g = np.array([g1 - 1, g1, g1 + 1])         # neighbors for quadratic interpolation
    delta = XP % DX
    fraz = np.array([(1 - delta) ** 2 / 2,
                     1 - ((1 - delta) ** 2 / 2 + delta ** 2 / 2),
                     delta ** 2 / 2] * wp)

    # apply periodic boundary conditions
    g = toPeriodic(g, NG, discrete=True)

    # construct sparse interpolation matrix
    return (sparse.csr_matrix((fraz[0], (p, g[0])), shape=(N, NG)) +
            sparse.csr_matrix((fraz[1], (p, g[1])), shape=(N, NG)) +
            sparse.csr_matrix((fraz[2], (p, g[2])), shape=(N, NG)))


def interpolate(M: sparse.csr_matrix, DX: float, NG: int, Q: float, rho_back: float) -> np.ndarray:
    """
    Interpolate particle quantities to grid and compute grid density.

    Args:
        M (sparse.csr_matrix): Particle-to-grid interpolation matrix (N x NG).
        DX (float): Grid spacing.
        NG (int): Number of grid points.
        Q (float): Particle charge.
        rho_back (float): Background charge density.

    Returns:
        np.ndarray: Grid charge density of shape (NG,).

    Notes:
        - Computes ρ = Q / DX * sum(M) + background density.
        - Useful to compute total charge on each grid cell from particles.
    """
    return np.asarray((Q / DX) * M.sum(0) + rho_back * np.ones([1, NG]))[0]


def specInterpolate(XP: np.ndarray, Shat: np.ndarray, NG: tuple[int, int], N: int, Q: float, L: tuple[float, float], wp: float = 1) -> np.ndarray:
    """
    Spectrally interpolate particle charges to Fourier-space grid using NUFFT.

    Args:
        XP (np.ndarray): Particle positions (2D array, shape [2, N]).
        Shat (np.ndarray): Shape factors in Fourier space.
        NG (tuple[int, int]): Number of grid points in x and y directions.
        N (int): Number of particles.
        Q (float): Particle charge.
        L (tuple[float, float]): Domain lengths in x and y.
        wp (float, optional): Particle weights. Default is 1.

    Returns:
        np.ndarray: Fourier-space charge density rhoHat.

    Notes:
        - Uses NUFFT (non-uniform FFT) to map irregular particle positions to uniform Fourier grid.
        - Useful in spectral Poisson solvers or FNO-based PIC implementations.
    """
    rhoHat = np.conjugate(
        Q * Shat * finufft.nufft2d1(
            XP[0] * 2 * np.pi / L[0],
            XP[1] * 2 * np.pi / L[1],
            0j + np.zeros(N) + wp,
            tuple(NG),
            eps=1e-12,
            modeord=1
        )
    )
    return rhoHat
