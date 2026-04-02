#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))
import numpy as np
import cupy as cp

def f(x: float, alpha: float, kd: float, u: float) -> float:
    """
    Evaluate the nonlinear transformation function for inverse sampling.

    Args:
        x (float): Current guess.
        alpha (float): Amplitude parameter.
        kd (float): Wave number.
        u (float): Uniform random variable mapped to [0, L].

    Returns:
        float: Value of the function f(x) = x + alpha * sin(kd*x)/kd - u.
    """
    return x + (alpha * (np.sin(kd * x) / kd)) - u


def fprime(x: float, alpha: float, kd: float) -> float:
    """
    Evaluate the derivative of the nonlinear transformation function.

    Args:
        x (float): Current guess.
        alpha (float): Amplitude parameter.
        kd (float): Wave number.

    Returns:
        float: Derivative f'(x) = 1 + alpha * cos(kd*x).
    """
    return 1 + (alpha * np.cos(kd * x))


def Newton1d(xi: float, alpha: float, kd: float, u: float) -> tuple[float, int]:
    """
    Solve f(x) = 0 using Newton-Raphson iteration in 1D.

    Args:
        xi (float): Initial guess for x.
        alpha (float): Amplitude parameter.
        kd (float): Wave number.
        u (float): Target value for the transformation.

    Returns:
        x (float): Root of f(x) = 0.
        k (int): Number of iterations performed.

    Raises:
        RuntimeError: If maximum number of iterations is reached without convergence.
    """
    tol = 1e-12
    max_iter = 20
    k = 0
    x = 0
    while (k <= max_iter) and (np.abs(f(xi, alpha, kd, u)) > tol):
        x = xi - f(xi, alpha, kd, u) / fprime(xi, alpha, kd)
        xi = x
        k += 1
    if k == max_iter:
        raise RuntimeError("Newton iterations did not converge")
    return x, k


def InvTransSampling(alpha: float, k: np.ndarray, L: float, N: int, dim: int, label='tsi') -> np.ndarray:
    """
    Generate particle positions using inverse transform sampling for a sinusoidal perturbation.

    Args:
        alpha (float): Amplitude of perturbation.
        k (np.ndarray): Wave number.
        L (float): Domain length.
        N (int): Number of particles to sample.
        dim (int): Dimension

    Returns:
        np.ndarray: Array of particle positions sampled according to x + (alpha*sin(k*x)/k).
    """
    if dim == 1:
        xp = np.zeros(N)
        u0 = np.random.rand(N)
        vp = np.random.randn(self.N)
        for i in range(N):
            print(i)
            u = L[0] * u0[i]
            x = u / (1 + alpha)  # initial guess
            xp[i], _ = Newton1d(x, alpha, k[0], u)
        return xp,vp
    else:
        xp = np.zeros([2, N])
        if((label == 'weakLandau') or (label == 'strongLandau')): 
            vp = np.random.randn(2, N)
            u0 = np.random.rand(2, N)
            for i in range(N):
                print(i)
                for d in range(2):
                    u =  L[d] * u0[d, i]
                    x = u / (1+alpha)
                    xp[d,i],niter = Newton1d(x,alpha,k[d],u)
        elif(label == 'tsi'):
            vp = np.zeros([2, N])
            vp[0,:] = np.random.randn(1, N)
            Nhalf = int(N/2)
            vp[1,:Nhalf] = -np.pi/2.0 + 0.1 * np.random.randn(Nhalf)
            vp[1,Nhalf:] =  np.pi/2.0 + 0.1 * np.random.randn(Nhalf)
            u0 = np.random.rand(2, N)
            xp[0,:] = L[0] * u0[0,:]
            for i in range(N):
                print(i)
                u =  L[1] * u0[1, i]
                x = u / (1+alpha)
                xp[1,i],niter = Newton1d(x,alpha,k[1],u)
        elif(label == 'bti'):
            vp = np.zeros([2, N])
            vp[0,:] = np.random.randn(1, N)
            sigma = 1 / np.sqrt(2)
            ninetypercent = int(0.9*N)
            rem = N - ninetypercent
            vp[1,:ninetypercent] = sigma * np.random.randn(ninetypercent)
            vp[1,ninetypercent:] =  4.0 + sigma * np.random.randn(rem)
            u0 = np.random.rand(2, N)
            xp[0,:] = L[0] * u0[0,:]
            for i in range(N):
                print(i)
                u =  L[1] * u0[1, i]
                x = u / (1+alpha)
                xp[1,i],niter = Newton1d(x,alpha,k[1],u)
        


        return xp,vp



def findsource():
    """
    Placeholder function for a particle source term.

    Returns:
        None: No source term is implemented.
    """
    return None


def inv_trans_sampling_gpu(alpha, k, L, N, dim=1,
                           max_iter=12, tol=1e-12,
                           dtype=cp.float64,
                           out_dtype=cp.float64,
                           label='weakLandau',
                           ref='pif'):
    """
    GPU inverse-transform sampling for initial condition generation.

    Returns:
        XP: shape (dim, N)
        VP: shape (dim, N)
    """

    # uniform u in [0, L_d)
    U0 = cp.random.rand(dim, N, dtype=dtype)
    Larr = cp.asarray(L, dtype=dtype).reshape(dim, 1)
    U = U0 * Larr
    if((label == 'weakLandau') or (label == 'strongLandau')): 
        # velocities: Maxwellian
        VP = cp.random.randn(dim, N, dtype=dtype)


        # initial guess
        X = U / (1.0 + alpha)

        # reshape k to (dim,1) for broadcasting
        karr = cp.asarray(k, dtype=dtype).reshape(dim, 1)

        # Newton iteration (vectorized over ALL particles)
        for _ in range(max_iter):
            f  = X + alpha * (cp.sin(karr * X) / karr) - U
            fp = 1.0 + alpha * cp.cos(karr * X)

            dX = f / fp
            Xnew = X - dX

            # global convergence check
            if cp.max(cp.abs(dX)) < tol:
                X = Xnew
                break

            X = Xnew
    elif((label == 'tsi') or (label == 'bti')):
        # uniform u in [0, L_d)
        X = U.copy()
        # initial guess
        X[dim-1] = U[dim-1] / (1.0 + alpha)

        # reshape k to (dim,1) for broadcasting
        karr = cp.asarray(k, dtype=dtype).reshape(dim, 1)

        # Newton iteration (vectorized over ALL particles)
        for _ in range(max_iter):
            f  = X[dim-1] + alpha * (cp.sin(karr[dim-1] * X[dim-1]) / karr[dim-1]) - U[dim-1]
            fp = 1.0 + alpha * cp.cos(karr[dim-1] * X[dim-1])

            dX = f / fp
            Xnew = X[dim-1] - dX

            # global convergence check
            if cp.max(cp.abs(dX)) < tol:
                X[dim-1] = Xnew
                break

            X[dim-1] = Xnew

        VP = cp.zeros([dim, N])
        if(dim > 1):
            VP[0] = cp.random.randn(1, N)
        if(dim > 2):
            VP[1] = cp.random.randn(1, N)
        if(label == 'tsi'):
            Nhalf = int(N/2)
            VP[dim-1,:Nhalf] = -cp.pi/2.0 + 0.1 * cp.random.randn(Nhalf)
            VP[dim-1,Nhalf:] =  cp.pi/2.0 + 0.1 * cp.random.randn(Nhalf)
        elif(label=='bti'):
            sigma = 1 / cp.sqrt(2)
            ninetypercent = int(0.9*N)
            rem = N - ninetypercent
            VP[dim-1,:ninetypercent] = sigma * cp.random.randn(ninetypercent)
            VP[dim-1,ninetypercent:] =  4.0 + sigma * cp.random.randn(rem)

    elif(label == 'cyclotron'):
        assert dim == 2, 'Cyclotron test case only for 2D' 
        sigmas = cp.array([Larr[0]/10,Larr[1]/30]) / cp.sqrt(2) # Control the shape of the beam
        X = cp.random.randn(dim, N) * sigmas
        if(ref == 'pic'):
            X = X + cp.array([0.5*Larr[0], 0.5*Larr[1]])
            X = cp.mod(X, Larr)

        VP = cp.random.randn(dim, N)
        XP = X

    if(label != 'cyclotron'):
        # periodic wrap to [0,L)
        XP = cp.mod(X, Larr)

    return XP.astype(out_dtype), VP.astype(out_dtype)
