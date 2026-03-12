#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))
import numpy as np
import cupy as cp

def fieldInFourier(rhoHat, L, dim, testCase, ref, J, T1, Q=None, T2=None):
    """
    Compute the Fourier coefficients of the electrostatic potential and electric field from charge density.

    Args:
        rhoHat (cp.ndarray): Fourier transform of the charge density (1D array).
        L (cp.ndarray): Domain length.
        dim (int): Dimension.
        testCase (str): Testcase label.

    Returns:
        phiHat (cp.ndarray): Fourier coefficients of the electrostatic potential.
        EHat (cp.ndarray): Fourier coefficients of the electric field.

    Notes:
        This function solves the 1D Poisson equation in Fourier space:
        - phiHat[k] = rhoHat[k] * (L / (2*pi*k))^2 for k != 0
        - EHat[k] = rhoHat[k] * L / (2j*pi*k)
        The zero-frequency and Nyquist modes are set to zero.
    """
    if dim == 1:
        N = rhoHat.size
        Ka = cp.arange(1, N // 2)                # 1,2,...,(N/2 - 1)
        Kb = Ka[::-1]                            # reversed
        K = cp.append(cp.append(Ka, [rhoHat.size // 2]), - Kb)
        rhoHat  = cp.squeeze(rhoHat)
        phiHat = cp.append([0], rhoHat[1:] * (L / (2 * cp.pi * K)) ** 2)
        EHat = cp.append([0], rhoHat[1:] * L / (2j * cp.pi * K))
        EHat[N // 2] = 0
        #K = cp.concatenate([Ka, [N // 2], -Kb])  # length N-1

        #rho_modes = rhoHat[1:]                   # skip zero mode → length N-1
        #phiHat = cp.concatenate([[0], rho_modes * (L / (2 * cp.pi * K)) ** 2])
        #EHat   = cp.concatenate([[0], rho_modes * (L / (2j * cp.pi * K))])
        #EHat[N // 2] = 0
    else:
        if(testCase == 'cyclotron'):
            if(ref == 'pif'):
                phiHat = Q * T1 * rhoHat # /phi in Fourier space
                coeff1 = Q * T2 * rhoHat * -1j * cp.transpose(J)[::2, ::2] # Not exactly Electric field! Notice it convolutes twice with shape function
                coeff2 = Q * T2 * rhoHat * -1j * J[::2, ::2] 
                coeff1 = cp.array(coeff1, order="C")
                coeff2 = cp.array(coeff2, order="C")
                EHat = cp.array([coeff1, coeff2])
            else:
                phiHat = T1 * rhoHat
                E1_Hat = phiHat * -1j * cp.transpose(J)[::2, ::2]
                E2_Hat = phiHat * -1j * J[::2, ::2]
                EHat = cp.array([E1_Hat, E2_Hat])

        else:
            Ja = cp.arange(rhoHat.shape[0] // 2)
            Jb = Ja[:0:-1]
            J = cp.append(cp.append(Ja, [-rhoHat.shape[0] // 2]), - Jb)
            Ka = cp.arange(rhoHat.shape[1] // 2)
            Kb = Ka[:0:-1]
            K = cp.append(cp.append(Ka, [-rhoHat.shape[1] // 2]), - Kb)
            J = cp.transpose(cp.expand_dims(J, 0).repeat(rhoHat.shape[1], axis=0)) * 2 * cp.pi / L[0]
            K = cp.expand_dims(K, 0).repeat(rhoHat.shape[0], axis=0) * 2 * cp.pi / L[1]
            absolute = J ** 2 + K ** 2
            absolute[0,0] = 1
            phiHat = rhoHat / absolute
            phiHat[0,0] = 0
            E0 = phiHat * -1j * J
            E1 = phiHat * -1J * K
            # Zero out Nyquist modes to ensure real symmetry and avoid spurious anisotropy
            phiHat[rhoHat.shape[0]//2, :] = 0
            phiHat[:, rhoHat.shape[1]//2] = 0
            E0[rhoHat.shape[0]//2, :] = 0
            E0[:, rhoHat.shape[1]//2] = 0
            E1[rhoHat.shape[0]//2, :] = 0
            E1[:, rhoHat.shape[1]//2] = 0
            EHat = cp.array([E0, E1])

    return phiHat, EHat


def field(rho, L, dim, J, T1, testCase, NG):
    """
    Compute the real-space electrostatic potential and electric field from charge density.

    Args:
        rho (cp.ndarray): Charge density in real space (1D array).
        L (cp.ndarray): Domain length.
        dim (int): Dimension
    Returns:
        phi (cp.ndarray): Electrostatic potential in real space.
        E (cp.ndarray): Electric field in real space.

    Notes:
        - Performs FFT on the charge density.
        - Solves Poisson equation in Fourier space via `fieldInFourier`.
        - Converts back to real space using inverse FFT.
    """
    if dim == 1:
        rhoHat = cp.fft.fft(rho).ravel()
        phiHat, EHat = fieldInFourier(rhoHat, L[0], dim)
        phi = cp.real(cp.fft.ifft(phiHat)).ravel()
        E = cp.real(cp.fft.ifft(EHat)).ravel()
    else:
        if(testCase == 'cyclotron'):
            extension = 4
            rhoHat = cp.fft.fft2(rho, s=[extension*NG//2, extension*NG//2]) * 4
        else:
            rhoHat = cp.fft.fft2(rho)
        phiHat, EHat = fieldInFourier(rhoHat, L, dim, testCase, 'pic', J, T1)
        phi = cp.real(cp.fft.ifft2(phiHat))
        if(testCase == 'cyclotron'):
            extension = 4
            E = cp.real(cp.array([cp.fft.ifft2(EHat[0])[extension*NG//4:extension*NG//2, extension*NG//4:extension*NG//2], cp.fft.ifft2(EHat[1])[extension*NG//4:extension*NG//2, extension*NG//4:extension*NG//2]]))
        else:
            E = cp.real(cp.array([cp.fft.ifft2(EHat[0]), cp.fft.ifft2(EHat[1])]))

    return phi, E




