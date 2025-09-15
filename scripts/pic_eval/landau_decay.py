#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))
import numpy as np
import math, cmath
from mpmath import findroot, erfc

def decayRate(k: float) -> float:
    """
    Compute the Landau damping decay rate for a given wave number k.

    Args:
        k (float): Wave number (must be positive).

    Returns:
        float: Landau damping decay rate gamma = sqrt(2) * Im(root).

    Raises:
        ValueError: If k <= 0.

    Notes:
        - Solves the linearized 1D Vlasov-Poisson dispersion relation using 
          Newton-like root-finding (Muller's method) in the complex plane.
        - The root corresponds to the complex frequency ω = ω_r + i γ; the 
          decay rate is γ = Im(ω) * sqrt(2).
    """
    if k <= 0:
        raise ValueError("k must be positive")

    func = lambda x: 1 + 1 / k**2 + 1j * x * cmath.exp(
        -x**2 / (2 * k**2)
    ) * erfc(-1j * x / (math.sqrt(2) * k)) / (math.sqrt(2 / math.pi) * k**3)

    root = findroot(func, 0.01j, solver="muller")
    return math.sqrt(2.0) * float(np.imag(root))


def period(k: float) -> float:
    """
    Compute the approximate oscillation period for a wave with wave number k.

    Args:
        k (float): Wave number (must be positive).

    Returns:
        float: Approximate period T = 2*pi / Re(root).

    Notes:
        - Uses the real part of the root of the linearized Vlasov-Poisson
          dispersion relation to estimate the wave oscillation period.
        - Solves the same function as `decayRate`.
    """
    func = lambda x: 1 + 1 / k**2 + 1j * x * cmath.exp(
        -x**2 / (2 * k**2)
    ) * erfc(-1j * x / (math.sqrt(2) * k)) / (math.sqrt(2 / math.pi) * k**3)

    root = findroot(func, 0.01j, solver="muller")
    return 2.0 * math.pi / float(np.real(root))




