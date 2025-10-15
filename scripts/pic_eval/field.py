#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))
import numpy as np

def fieldInFourier(rhoHat: np.ndarray, L: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Fourier coefficients of the electrostatic potential and electric field from charge density.

    Args:
        rhoHat (np.ndarray): Fourier transform of the charge density (1D array).
        L (np.ndarray): Domain length.
        dim (int): Dimension.

    Returns:
        phiHat (np.ndarray): Fourier coefficients of the electrostatic potential.
        EHat (np.ndarray): Fourier coefficients of the electric field.

    Notes:
        This function solves the 1D Poisson equation in Fourier space:
        - phiHat[k] = rhoHat[k] * (L / (2*pi*k))^2 for k != 0
        - EHat[k] = rhoHat[k] * L / (2j*pi*k)
        The zero-frequency and Nyquist modes are set to zero.
    """
    if dim == 1:
        N = rhoHat.size
        Ka = np.arange(1, N // 2)                # 1,2,...,(N/2 - 1)
        Kb = Ka[::-1]                            # reversed
        K = np.concatenate([Ka, [N // 2], -Kb])  # length N-1

        rho_modes = rhoHat[1:]                   # skip zero mode → length N-1
        phiHat = np.concatenate([[0], rho_modes * (L / (2 * np.pi * K)) ** 2])
        EHat   = np.concatenate([[0], rho_modes * (L / (2j * np.pi * K))])
        EHat[N // 2] = 0
    else:
        Ja = np.arange(rhoHat.shape[0] // 2)
        Jb = Ja[:0:-1]
        J = np.append(np.append(Ja, [-rhoHat.shape[0] // 2]), - Jb)
        Ka = np.arange(rhoHat.shape[1] // 2)
        Kb = Ka[:0:-1]
        K = np.append(np.append(Ka, [-rhoHat.shape[1] // 2]), - Kb)
        J = np.transpose(np.expand_dims(J, 0).repeat(rhoHat.shape[1], axis=0)) * 2 * np.pi / L[0]
        K = np.expand_dims(K, 0).repeat(rhoHat.shape[0], axis=0) * 2 * np.pi / L[1]
        absolute = J ** 2 + K ** 2
        absolute[0,0] = 1
        phiHat = rhoHat / absolute
        phiHat[0,0] = 0
        E0 = phiHat * -1j * J
        E1 = phiHat * -1J * K
        EHat = np.array([E0, E1])

    return phiHat, EHat


def field(rho: np.ndarray, L: np.ndarray, dim:int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the real-space electrostatic potential and electric field from charge density.

    Args:
        rho (np.ndarray): Charge density in real space (1D array).
        L (np.ndarray): Domain length.
        dim (int): Dimension
    Returns:
        phi (np.ndarray): Electrostatic potential in real space.
        E (np.ndarray): Electric field in real space.

    Notes:
        - Performs FFT on the charge density.
        - Solves Poisson equation in Fourier space via `fieldInFourier`.
        - Converts back to real space using inverse FFT.
    """
    if dim == 1:
        rhoHat = np.fft.fft(rho).ravel()
        phiHat, EHat = fieldInFourier(rhoHat, L[0], dim)
        phi = np.real(np.fft.ifft(phiHat)).ravel()
        E = np.real(np.fft.ifft(EHat)).ravel()
    else:
        rhoHat = np.fft.fft2(rho)
        phiHat, EHat = fieldInFourier(rhoHat, L, dim)
        phi = np.real(np.fft.ifft2(phiHat))
        E = np.real(np.array([np.fft.ifft2(EHat[0]), np.fft.ifft2(EHat[1])]))

    return phi, E




