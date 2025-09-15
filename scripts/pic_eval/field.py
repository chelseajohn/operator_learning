#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))
import numpy as np

def fieldInFourier(rhoHat: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Fourier coefficients of the electrostatic potential and electric field from charge density.

    Args:
        rhoHat (np.ndarray): Fourier transform of the charge density (1D array).
        L (float): Domain length.

    Returns:
        phiHat (np.ndarray): Fourier coefficients of the electrostatic potential.
        EHat (np.ndarray): Fourier coefficients of the electric field.

    Notes:
        This function solves the 1D Poisson equation in Fourier space:
        - phiHat[k] = rhoHat[k] * (L / (2*pi*k))^2 for k != 0
        - EHat[k] = rhoHat[k] * L / (2j*pi*k)
        The zero-frequency and Nyquist modes are set to zero.
    """
    N = rhoHat.size
    Ka = np.arange(1, N // 2)                # 1,2,...,(N/2 - 1)
    Kb = Ka[::-1]                            # reversed
    K = np.concatenate([Ka, [N // 2], -Kb])  # length N-1

    rho_modes = rhoHat[1:]                   # skip zero mode → length N-1
    phiHat = np.concatenate([[0], rho_modes * (L / (2 * np.pi * K)) ** 2])
    EHat   = np.concatenate([[0], rho_modes * (L / (2j * np.pi * K))])
    EHat[N // 2] = 0

    return phiHat, EHat


def field(rho: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the real-space electrostatic potential and electric field from charge density.

    Args:
        rho (np.ndarray): Charge density in real space (1D array).
        L (float): Domain length.

    Returns:
        phi (np.ndarray): Electrostatic potential in real space.
        E (np.ndarray): Electric field in real space.

    Notes:
        - Performs FFT on the charge density.
        - Solves Poisson equation in Fourier space via `fieldInFourier`.
        - Converts back to real space using inverse FFT.
    """
    rhoHat = np.fft.fft(rho).ravel()
    phiHat, EHat = fieldInFourier(rhoHat, L)
    phi = np.real(np.fft.ifft(phiHat)).ravel()
    E = np.real(np.fft.ifft(EHat)).ravel()
    return phi, E




