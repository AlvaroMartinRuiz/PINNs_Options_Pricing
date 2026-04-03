"""
Synthetic Local Volatility surface for Phase 2.

The LV surface is defined in log-moneyness coordinates (m, tau):
    sigma_loc(m, tau) = 0.2 + 0.1 * exp(-m^2 / 0.3) + 0.05 * tau

Properties:
    - Base level: 20% vol
    - Smile: 10% bump at ATM (m=0), decaying with moneyness
    - Term structure: vol increases linearly with maturity (+5% per year)
"""

import numpy as np
import torch


def synthetic_lv_numpy(m, tau):
    """
    Evaluate the synthetic local volatility surface (NumPy version).

    Parameters
    ----------
    m   : ndarray -- log-moneyness ln(S/K)
    tau : ndarray -- time to maturity (T - t)

    Returns
    -------
    sigma : ndarray -- local volatility at each (m, tau) point
    """
    return 0.2 + 0.1 * np.exp(-m**2 / 0.3) + 0.05 * tau


def synthetic_lv_torch(m, tau):
    """
    Evaluate the synthetic local volatility surface (PyTorch version).
    Differentiable -- can be used inside autograd computations.
    """
    return 0.2 + 0.1 * torch.exp(-m**2 / 0.3) + 0.05 * tau


def generate_observation_grid(S0=100.0, r=0.05, q=0.015,
                               n_strikes=25, n_maturities=10):
    """
    Generate a grid of "market-like" observation points (m, tau).

    Mimics realistic market data:
    - Strikes concentrated near ATM with wider spacing further out
    - Multiple maturities from short-dated to 1 year

    Parameters
    ----------
    S0           : float -- current spot price
    r, q         : float -- risk-free rate and dividend yield
    n_strikes    : int   -- number of strikes per maturity
    n_maturities : int   -- number of maturities

    Returns
    -------
    m_obs   : 1D array of log-moneyness values
    tau_obs : 1D array of time-to-maturity values
    (These are flattened grids, total points = n_strikes * n_maturities)
    """
    # Strikes: tighter near ATM, wider in the wings
    # m = ln(S/K): m > 0 is ITM, m < 0 is OTM for a call
    m_values = np.linspace(-0.3, 0.3, n_strikes)

    # Maturities: from 1 month to 1 year
    tau_values = np.linspace(0.08, 1.0, n_maturities)

    m_grid, tau_grid = np.meshgrid(m_values, tau_values, indexing='ij')

    return m_grid.flatten(), tau_grid.flatten(), m_values, tau_values
