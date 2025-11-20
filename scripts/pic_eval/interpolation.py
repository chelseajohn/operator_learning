#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))
import numpy as np
from scipy import sparse
#import finufft
from dynamics import toPeriodic

def interpMatrix(XP: np.ndarray, wp: float, DX: np.ndarray, N: int, NG: int, p: np.ndarray,L: np.ndarray, dim:int) -> sparse.csr_matrix:
    """
    Construct the projection (interpolation) matrix from particles to grid.

    Args:
        XP (np.ndarray): Particle positions (1D array, shape: [N]).
        wp (float): Particle weights.
        DX (np.ndarray): Grid spacing.
        N (int): Number of particles.
        NG (int): Number of grid points.
        p (np.ndarray): Particle indices (0..N-1).
        L (np.ndarray): Length of container.
        dim (int): Dimension

    Returns:
        scipy.sparse.csr_matrix: Sparse interpolation matrix of shape (N, NG).

    Notes:
        - Uses quadratic (3-point) weighting to distribute particle quantities to the grid.
        - Applies periodic boundary conditions on grid indices.
        - Useful for projecting charge, current, or other particle quantities to a uniform grid.
    """

    if dim == 1:
        g1 = np.floor(XP / DX[0]).astype(int)          # primary grid index
        g = np.array([g1 - 1, g1, g1 + 1])             # neighbors for quadratic interpolation
        delta = XP % DX[0]
        fraz = np.array([(1 - delta) ** 2 / 2,
                        1 - ((1 - delta) ** 2 / 2 + delta ** 2 / 2),
                        delta ** 2 / 2] * wp)

        # apply periodic boundary conditions
        g = toPeriodic(g, NG, discrete=True)

        # construct sparse interpolation matrix
        return (sparse.csr_matrix((fraz[0], (p, g[0])), shape=(N, NG)) +
                sparse.csr_matrix((fraz[1], (p, g[1])), shape=(N, NG)) +
                sparse.csr_matrix((fraz[2], (p, g[2])), shape=(N, NG)))
    else:
        g0, g1 = np.floor(XP[0] / DX[0]).astype(int), np.floor(XP[1] / DX[1]).astype(int)
        g = np.array([[g0 - 1, g0, g0 + 1],[g1 - 1, g1, g1 + 1]])
        a, b = XP[0] % DX[0], XP[1] % DX[1]
        c1, c2, c3, c4 = (DX[0]-a)**2, (DX[1]-b)**2, DX[0]**2 + 2 * DX[0] * a - 2 * a**2, DX[1]**2 + 2 * DX[1] * b - 2 * b**2
        tot = (DX[0] * DX[1]) ** 2
        A = c1 * c2 / (4*tot)
        B = c2 * c3 / (4*tot)
        C = a**2 * c2/ (4*tot)
        D = c1 * c4 / (4*tot)
        F = a**2 * c4 / (4*tot)
        G = b**2 * c1 / (4*tot)
        H = b**2 * c3 / (4*tot)
        I = a**2 * b**2 / (4*tot)
        E = 1 - A - B - C - D - F - G - H - I
        fraz = np.array([A, B, C, D, E, F, G, H, I] * wp)
        g[0] = toPeriodic(g[0], int(L[0]/DX[0]), True)
        g[1] = toPeriodic(g[1], int(L[1]/DX[1]), True)
        matrices = sparse.csr_matrix((N, NG**2))
        for i in range(3):
            for j in range(3):
                matrices = matrices + sparse.csr_matrix((fraz[3*i+j], (p, int(L[1]/DX[1]) * g[0,i] + g[1,j])),shape=(N, NG**2))

        return matrices



def interpolate(M: sparse.csr_matrix, DX: np.ndarray, L: np.ndarray, NG: int, Q: float, rho_back: float, dim:int) -> np.ndarray:
    """
    Interpolate particle quantities to grid and compute grid density.

    Args:
        M (sparse.csr_matrix): Particle-to-grid interpolation matrix (N x NG).
        DX (np.ndarray): Grid spacing.
        L (np.ndarray): Length of container.
        NG (int): Number of grid points.
        Q (float): Particle charge.
        rho_back (float): Background charge density.
        dim (int): Dimension

    Returns:
        np.ndarray: Grid charge density of shape (NG,).

    Notes:
        - Computes ρ = Q / DX * sum(M) + background density.
        - Useful to compute total charge on each grid cell from particles.
    """
    if dim == 1:
        return np.asarray((Q / DX[0]) * M.sum(0) + rho_back * np.ones([1, NG]))[0]
    else:
        return (Q / (DX[0]*DX[1])) * M.sum(0).reshape([int(L[0]/DX[0]), int(L[1]/DX[1])])

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
