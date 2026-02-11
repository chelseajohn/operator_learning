#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))
import numpy as np

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
