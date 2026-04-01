"""
Black-Scholes analytical formulae for European options with continuous dividends.

Provides:
    - bs_call / bs_put:  closed-form prices
    - bs_greeks:         Delta, Gamma, Theta, Vega, Rho
    - bs_call_torch:     vectorised PyTorch version (differentiable, for validation)
"""

import math
import numpy as np
from scipy.stats import norm


# ── Scalar / NumPy versions ──────────────────────────────────────────────────

def _d1d2(S: np.ndarray, K: float, r: float, q: float,
          sigma: float, tau: np.ndarray):
    """Compute d1, d2.  tau = T - t  (time to maturity)."""
    sqrt_tau = np.sqrt(np.maximum(tau, 1e-12))  # avoid div-by-zero
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return d1, d2


def bs_call(S, K, r, q, sigma, tau):
    """
    European call price under BS with continuous dividend yield q.

    Parameters
    ----------
    S     : array-like  – spot price(s)
    K     : float       – strike
    r     : float       – risk-free rate
    q     : float       – continuous dividend yield
    sigma : float       – volatility
    tau   : array-like  – time to maturity (T - t)

    Returns
    -------
    price : ndarray
    """
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    d1, d2 = _d1d2(S, K, r, q, sigma, tau)
    price = S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    # At expiry (tau ≈ 0) the intrinsic value should be max(S-K, 0)
    at_expiry = tau < 1e-12
    price = np.where(at_expiry, np.maximum(S - K, 0.0), price)
    return price


def bs_put(S, K, r, q, sigma, tau):
    """European put price via put-call parity."""
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    d1, d2 = _d1d2(S, K, r, q, sigma, tau)
    price = K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)
    at_expiry = tau < 1e-12
    price = np.where(at_expiry, np.maximum(K - S, 0.0), price)
    return price


def bs_greeks(S, K, r, q, sigma, tau, option_type="call"):
    """
    Compute BS Greeks for European options.

    Returns dict: {delta, gamma, theta, vega, rho}
    All values are per-unit (not per-contract).
    """
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    d1, d2 = _d1d2(S, K, r, q, sigma, tau)
    sqrt_tau = np.sqrt(np.maximum(tau, 1e-12))
    n_d1 = norm.pdf(d1)  # standard normal density

    if option_type == "call":
        delta = np.exp(-q * tau) * norm.cdf(d1)
        theta = (- S * np.exp(-q * tau) * n_d1 * sigma / (2 * sqrt_tau)
                 + q * S * np.exp(-q * tau) * norm.cdf(d1)
                 - r * K * np.exp(-r * tau) * norm.cdf(d2))
        rho = K * tau * np.exp(-r * tau) * norm.cdf(d2)
    else:  # put
        delta = np.exp(-q * tau) * (norm.cdf(d1) - 1)
        theta = (- S * np.exp(-q * tau) * n_d1 * sigma / (2 * sqrt_tau)
                 - q * S * np.exp(-q * tau) * norm.cdf(-d1)
                 + r * K * np.exp(-r * tau) * norm.cdf(-d2))
        rho = -K * tau * np.exp(-r * tau) * norm.cdf(-d2)

    gamma = np.exp(-q * tau) * n_d1 / (S * sigma * sqrt_tau)
    vega = S * np.exp(-q * tau) * n_d1 * sqrt_tau

    return {"delta": delta, "gamma": gamma, "theta": theta,
            "vega": vega, "rho": rho}


# ── PyTorch version (for validation inside training loop) ────────────────────

def bs_call_torch(S, K, r, q, sigma, tau):
    """
    Vectorised BS call price in pure PyTorch (differentiable).

    All inputs are torch tensors (or broadcastable scalars).
    Uses the error-function approximation of the normal CDF:
        Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    """
    import torch

    sqrt_tau = torch.sqrt(torch.clamp(tau, min=1e-12))
    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    sqrt2 = math.sqrt(2.0)
    Nd1 = 0.5 * (1.0 + torch.erf(d1 / sqrt2))
    Nd2 = 0.5 * (1.0 + torch.erf(d2 / sqrt2))

    price = S * torch.exp(-q * tau) * Nd1 - K * torch.exp(-r * tau) * Nd2

    # At expiry
    at_expiry = tau < 1e-12
    intrinsic = torch.clamp(S - K, min=0.0)
    price = torch.where(at_expiry, intrinsic, price)
    return price
