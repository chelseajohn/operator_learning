#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))
import numpy as np
import cupy as cp
from scipy import sparse
from cupyx.scipy import sparse as sparsecp
from cupyx import scatter_add
import cufinufft
from dynamics import toPeriodic

def interpMatrix(XP: cp.ndarray, wp: float, DX: cp.ndarray, N: int, NG: int, p: cp.ndarray,L: cp.ndarray, dim:int) -> sparsecp.csr_matrix:
    """
    Construct the projection (interpolation) matrix from particles to grid.

    Args:
        XP (cp.ndarray): Particle positions (1D array, shape: [N]).
        wp (float): Particle weights.
        DX (cp.ndarray): Grid spacing.
        N (int): Number of particles.
        NG (int): Number of grid points.
        p (cp.ndarray): Particle indices (0..N-1).
        L (cp.ndarray): Length of container.
        dim (int): Dimension

    Returns:
        cupyx.scipy.sparse.csr_matrix: Sparse interpolation matrix of shape (N, NG).

    Notes:
        - Uses quadratic (3-point) weighting to distribute particle quantities to the grid.
        - Applies periodic boundary conditions on grid indices.
        - Useful for projecting charge, current, or other particle quantities to a uniform grid.
    """

    if dim == 1:
        g1 = cp.floor(XP / DX[0]).astype(int)          # primary grid index
        g = cp.array([g1 - 1, g1, g1 + 1])             # neighbors for quadratic interpolation
        delta = XP % DX[0]
        fraz = cp.array([(1 - delta) ** 2 / 2,
                        1 - ((1 - delta) ** 2 / 2 + delta ** 2 / 2),
                        delta ** 2 / 2] * wp)

        # apply periodic boundary conditions
        g = toPeriodic(g, NG, discrete=True)

        # construct sparse interpolation matrix
        return (sparsecp.csr_matrix((fraz[0], (p, g[0])), shape=(N, NG)) +
                sparsecp.csr_matrix((fraz[1], (p, g[1])), shape=(N, NG)) +
                sparsecp.csr_matrix((fraz[2], (p, g[2])), shape=(N, NG)))
    else:
        g0, g1 = cp.floor(XP[0] / DX[0]).astype(int), cp.floor(XP[1] / DX[1]).astype(int)
        g = cp.array([[g0 - 1, g0, g0 + 1],[g1 - 1, g1, g1 + 1]])
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
        fraz = cp.array([A, B, C, D, E, F, G, H, I] * wp)
        g[0] = toPeriodic(g[0], int(L[0]/DX[0]), True)
        g[1] = toPeriodic(g[1], int(L[1]/DX[1]), True)
        matrices = sparsecp.csr_matrix((N, NG**2))
        for i in range(3):
            for j in range(3):
                matrices = matrices + sparsecp.csr_matrix((fraz[3*i+j], (p, int(L[1]/DX[1]) * g[0,i] + g[1,j])),shape=(N, NG**2))

        return matrices



def interpolate(M: sparsecp.csr_matrix, DX: cp.ndarray, L: cp.ndarray, NG: int, Q: float, rho_back: float, dim:int) -> cp.ndarray:
    """
    Interpolate particle quantities to grid and compute grid density.

    Args:
        M (sparsecp.csr_matrix): Particle-to-grid interpolation matrix (N x NG).
        DX (cp.ndarray): Grid spacing.
        L (cp.ndarray): Length of container.
        NG (int): Number of grid points.
        Q (float): Particle charge.
        rho_back (float): Background charge density.
        dim (int): Dimension

    Returns:
        cp.ndarray: Grid charge density of shape (NG,).

    Notes:
        - Computes ρ = Q / DX * sum(M) + background density.
        - Useful to compute total charge on each grid cell from particles.
    """
    if dim == 1:
        return cp.asarray((Q / DX[0]) * M.sum(0) + rho_back * cp.ones([1, NG]))[0]
    else:
        return (Q / (DX[0]*DX[1])) * M.sum(0).reshape([int(L[0]/DX[0]), int(L[1]/DX[1])])

def scatterFourier(XP, SHat, NG, N, Q, L, dim, wp=1):
    """
    Spectrally interpolate particle charges to Fourier-space grid using NUFFT.

    Args:
        XP (cp.ndarray): Particle positions (ND array, shape [dim, N]).
        Shat (cp.ndarray): Shape factors in Fourier space.
        NG: Number of grid points in each dimension (currently same in all dimensions).
        N (int): Number of particles.
        Q (float): Particle charge.
        L (cp.ndarray): Domain lengths in each dimension as an array.
        wp (float, optional): Particle weights. Default is 1.

    Returns:
        cp.ndarray: Fourier-space charge density rhoHat.

    Notes:
        - Uses NUFFT (non-uniform FFT) to map irregular particle positions to uniform Fourier grid.
        - Useful in spectral Poisson solvers or FNO-based PIC implementations.
    """
    if dim == 1:
        rhoHat = Q * SHat * (cufinufft.nufft1d1(
                XP[0] * 2 * cp.pi / L[0],
                0j + cp.zeros(N) + wp,
                n_modes=NG,
                eps=1e-12,
                isign=-1,
                modeord=1)) / L[0]

    else:
        rhoHat = Q * SHat * (cufinufft.nufft2d1(
                XP[0] * 2 * cp.pi / L[0],
                XP[1] * 2 * cp.pi / L[1],
                0j + cp.zeros(N) + wp,
                n_modes=(NG,NG),
                eps=1e-12,
                isign=-1,
                modeord=1)) / (L[0] * L[1])

    return rhoHat


def gatherFourier(XP, EHat, SHat, QM, L, dim, wp=1):
    coeff1 = EHat[0] * SHat
    if dim == 1:
        Ep = cp.real(cufinufft.nufft1d2(XP[0] * 2 * cp.pi / L[0], coeff1, eps=1e-12, isign=1, modeord=1))
        a = (QM / wp) * Ep
    else:
        Exp = cp.real(cufinufft.nufft2d2(XP[0] * 2 * cp.pi / L[0], XP[1] * 2 * cp.pi / L[1], coeff1, eps=1e-12, isign=1, modeord=1))
        coeff2 = EHat[1] * SHat
        Eyp = cp.real(cufinufft.nufft2d2(XP[0] * 2 * cp.pi / L[0], XP[1] * 2 * cp.pi / L[1], coeff2, eps=1e-12, isign=1, modeord=1))
        a1 = (QM / wp) * Exp
        a2 = (QM / wp) * Eyp
        Ep = cp.stack([Exp, Eyp], axis=0)
        a = cp.stack([a1, a2], axis=0)
        

    return Ep, a


def p2g_g2p_nostencil_arrays(XP, DX, NG, L, dim,
                             Q=None, rho_back=0.0,
                             E=None, QM=None,
                             return_Ep=True):
    """
    Unified matrix-free PIC scatter/gather without 3N/9N arrays.

    Fully vectorized on GPU, works for 1D and 2D.
    """
    #N = wp.shape[0]
    #wp = wp.astype(dtype, copy=False)

    rho = None
    Ep = None
    a = None

    if dim == 1:
        x = XP
        NGx = int(NG)
        dx = float(DX[0])

        # primary grid index
        g1 = cp.floor(x / dx).astype(cp.int32)
        #For periodic wrapping
        g = cp.stack([g1 - 1, g1, g1 + 1], axis=0) % NGx

        delta = x % dx
        w0 = (1.0 - delta) ** 2 / 2.0
        w2 = delta ** 2 / 2.0
        w1 = 1.0 - (w0 + w2)

        if Q is not None:
            rho = cp.zeros(NGx)
            scatter_add(rho, g[0], w0)
            scatter_add(rho, g[1], w1)
            scatter_add(rho, g[2], w2)
            rho = (Q / dx) * rho + rho_back

        if E is not None:
            Egrid = E.reshape(-1)
            # gather & accumulate per stencil
            Ep_vals = w0*Egrid[g[0]] + w1*Egrid[g[1]] + w2*Egrid[g[2]]
            Ep_ = Ep_vals
            Ep = Ep_.astype(cp.float32) if return_Ep else None
            if QM is not None:
                a = QM * Ep_
            else:
                a = None

        return rho, Ep, a

    elif dim == 2:
        x = XP[0,:]
        y = XP[1,:]
        NGx, NGy = int(NG[0]), int(NG[1])
        dx, dy = float(DX[0]), float(DX[1])

        gx0 = cp.floor(x / dx).astype(cp.int32)
        gy0 = cp.floor(y / dy).astype(cp.int32)

        #For periodic wrapping
        gx = cp.stack([gx0-1, gx0, gx0+1], axis=0) % NGx
        gy = cp.stack([gy0-1, gy0, gy0+1], axis=0) % NGy
        
        a_ = x % dx
        b_ = y % dy
        tot = (dx*dy)**2

        c1 = (dx - a_)**2
        c2 = (dy - b_)**2
        c3 = dx**2 + 2*dx*a_ - 2*a_**2
        c4 = dy**2 + 2*dy*b_ - 2*b_**2

        # stencil weights
        A = c1*c2/(4*tot)
        B = c2*c3/(4*tot)
        C = a_**2*c2/(4*tot)
        D = c1*c4/(4*tot)
        F = a_**2*c4/(4*tot)
        G = b_**2*c1/(4*tot)
        H = b_**2*c3/(4*tot)
        I = a_**2*b_**2/(4*tot)
        E0 = 1 - A - B - C - D - F - G - H - I

        if Q is not None:
            rho_flat = cp.zeros(NGx*NGy)

            def flat(ix, iy): return NGy*ix + iy

            # scatter each stencil individually
            scatter_add(rho_flat, flat(gx[0], gy[0]), A)
            scatter_add(rho_flat, flat(gx[1], gy[0]), B)
            scatter_add(rho_flat, flat(gx[2], gy[0]), C)
            scatter_add(rho_flat, flat(gx[0], gy[1]), D)
            scatter_add(rho_flat, flat(gx[1], gy[1]), E0)
            scatter_add(rho_flat, flat(gx[2], gy[1]), F)
            scatter_add(rho_flat, flat(gx[0], gy[2]), G)
            scatter_add(rho_flat, flat(gx[1], gy[2]), H)
            scatter_add(rho_flat, flat(gx[2], gy[2]), I)

            rho = (Q / (dx*dy)) * rho_flat.reshape(NGx, NGy) + rho_back

        if E is not None:
            Ex = E[0].reshape(-1)
            Ey = E[1].reshape(-1)
            def flat(ix, iy): return NGy*ix + iy

            # accumulate stencil contributions directly
            Exp = (A*Ex[flat(gx[0],gy[0])] + B*Ex[flat(gx[1],gy[0])] + C*Ex[flat(gx[2],gy[0])] +
                   D*Ex[flat(gx[0],gy[1])] + E0*Ex[flat(gx[1],gy[1])] + F*Ex[flat(gx[2],gy[1])] +
                   G*Ex[flat(gx[0],gy[2])] + H*Ex[flat(gx[1],gy[2])] + I*Ex[flat(gx[2],gy[2])])

            Eyp = (A*Ey[flat(gx[0],gy[0])] + B*Ey[flat(gx[1],gy[0])] + C*Ey[flat(gx[2],gy[0])] +
                   D*Ey[flat(gx[0],gy[1])] + E0*Ey[flat(gx[1],gy[1])] + F*Ey[flat(gx[2],gy[1])] +
                   G*Ey[flat(gx[0],gy[2])] + H*Ey[flat(gx[1],gy[2])] + I*Ey[flat(gx[2],gy[2])])

            if return_Ep:
                Ep = cp.stack([Exp, Eyp], axis=0)

            if QM is not None:
                a = QM * cp.stack([Exp,Eyp], axis=0)

        return rho, Ep, a

    else:
        raise ValueError("dim must be 1 or 2")
