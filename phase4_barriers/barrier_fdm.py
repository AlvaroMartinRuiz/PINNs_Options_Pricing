"""
Phase 4 -- Crank-Nicolson FDM solver for barrier options.

Solves the same Dupire PDE as Phase 2/3 but with modified boundary
conditions that enforce the knock-out constraint at the barrier.

Supports:
    - Down-and-Out Call  (left boundary = barrier, v = 0)
    - Up-and-Out Call    (right boundary = barrier, v = 0)

The local volatility sigma(m, tau) can come from:
    1. A constant (for sanity checks against BS closed-form)
    2. The Phase 3 PINN checkpoint (for real market calibration)
"""

import numpy as np
from scipy.linalg import solve_banded
import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── Volatility Bridge: Phase 3 PINN -> callable sigma_func ─────────────────

def sigma_from_pinn(model_path='results/phase3/pinn_phase3.pt', device=None):
    """
    Load the Phase 3 PINN and return a callable sigma_func(m_array, tau_scalar).

    This bridges Phase 3 (calibration) to Phase 4 (barrier pricing):
    the calibrated volatility surface is now used as a known coefficient
    in the barrier option PDE.

    Returns
    -------
    sigma_func : callable(m_array, tau_scalar) -> sigma_array
    domain     : dict with m_min, m_max, tau_max
    bs_params  : dict with r_pde, q_pde
    """
    from phase2_inverse.pinn_dual import PINN_Dual
    from utils.normalization import LogMoneynessNormalizer

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    domain = checkpoint['domain']

    model = PINN_Dual().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    normalizer = LogMoneynessNormalizer(m_scale=0.5, tau_max=domain['tau_max'])

    r_pde = checkpoint.get('r_pde', 0.05)
    q_pde = checkpoint.get('q_pde', 0.015)

    def sigma_func(m_array, tau_scalar):
        """Evaluate PINN local vol at an array of m values and a single tau."""
        m_np = np.asarray(m_array, dtype=np.float64)
        m_t = torch.tensor(m_np, dtype=torch.float32, device=device).unsqueeze(-1)
        tau_t = torch.full_like(m_t, float(tau_scalar))

        with torch.no_grad():
            m_n, tau_n = normalizer.normalize(m_t, tau_t)
            _, sigma_hat = model(m_n, tau_n)

        return sigma_hat.cpu().numpy().flatten()

    return sigma_func, domain, {'r': r_pde, 'q': q_pde}


def sigma_constant(sigma_val=0.20):
    """Return a constant-vol sigma_func for sanity checks."""
    def sigma_func(m_array, tau_scalar):
        return np.full_like(np.asarray(m_array, dtype=np.float64), sigma_val)
    return sigma_func


# ── Pre-computed Interpolation Grid ─────────────────────────────────────────

def precompute_sigma_grid(sigma_func, m_grid, tau_grid):
    """
    Pre-evaluate sigma on the full FDM grid for fast lookup.
    Returns a 2D array sigma_grid[i, n] = sigma(m_i, tau_n).
    """
    N_m = len(m_grid)
    N_tau = len(tau_grid)
    sigma_grid = np.zeros((N_m, N_tau))
    for n in range(N_tau):
        sigma_grid[:, n] = sigma_func(m_grid, tau_grid[n])
    return sigma_grid


# ── Barrier FDM Solver ──────────────────────────────────────────────────────

def crank_nicolson_barrier(sigma_func, r, q, barrier_m, m_spot, barrier_type='down-out',
                           m_min=-1.5, m_max=1.5, N_m=400,
                           tau_max=1.0, N_tau=400):
    """
    Crank-Nicolson solver for barrier option pricing.

    The domain is truncated at the barrier, and v = 0 is enforced there.

    Parameters
    ----------
    sigma_func   : callable(m_array, tau_scalar) -> sigma_array
    r, q         : risk-free rate, dividend yield
    barrier_m    : barrier level in log-moneyness coordinates: m_B = ln(B/K)
    m_spot       : spot level in log-moneyness coordinates: m_spot = ln(S0/K)
    barrier_type : 'down-out' or 'up-out'
    m_min, m_max : full domain bounds (will be truncated at barrier)
    N_m          : number of spatial grid points
    tau_max      : maximum maturity
    N_tau        : number of time steps

    Returns
    -------
    m_grid   : 1D array (N_m,)
    tau_grid : 1D array (N_tau+1,)
    V        : 2D array (N_m, N_tau+1) -- normalized barrier option price
    """
    # Truncate domain at barrier
    if barrier_type == 'down-out':
        m_lo = barrier_m     # left boundary = barrier (v = 0)
        m_hi = m_max
    elif barrier_type == 'up-out':
        m_lo = m_min
        m_hi = barrier_m     # right boundary = barrier (v = 0)
    else:
        raise ValueError(f"Unknown barrier_type: {barrier_type}")

    dm = (m_hi - m_lo) / (N_m - 1)
    dtau = tau_max / N_tau
    m_grid = np.linspace(m_lo, m_hi, N_m)
    tau_grid = np.linspace(0, tau_max, N_tau + 1)

    V = np.zeros((N_m, N_tau + 1))

    # Terminal condition (tau = 0): call payoff
    V[:, 0] = np.maximum(np.exp(m_grid) - 1.0, 0.0)

    # Enforce barrier at tau = 0 too
    if barrier_type == 'down-out':
        V[0, 0] = 0.0
    else:
        V[-1, 0] = 0.0

    N_int = N_m - 2

    # Pre-compute sigma on the interior grid for efficiency
    # Note: the FDM grid is m = ln(S/K).
    # The PINN expects m_pinn = ln(S0/K_dupire). By Dupire, K_dupire = S, 
    # so m_pinn = ln(S0/S) = ln(S0/K) - ln(S/K) = m_spot - m_grid.
    sigma_cache = np.zeros((N_int, N_tau + 1))
    for n in range(N_tau + 1):
        sigma_cache[:, n] = sigma_func(m_spot - m_grid[1:-1], tau_grid[n])

    for n in range(N_tau):
        tau_np1 = tau_grid[n + 1]

        sigma_n = sigma_cache[:, n]
        sigma_np1 = sigma_cache[:, n + 1]

        # Coefficients at tau_n (explicit/RHS)
        a_n = 0.5 * sigma_n**2 / dm**2
        b_n = (r - q - 0.5 * sigma_n**2) / (2.0 * dm)
        sub_n = a_n - b_n
        main_n = -2.0 * a_n - r
        sup_n = a_n + b_n

        # Coefficients at tau_{n+1} (implicit/LHS)
        a_np1 = 0.5 * sigma_np1**2 / dm**2
        b_np1 = (r - q - 0.5 * sigma_np1**2) / (2.0 * dm)
        sub_np1 = a_np1 - b_np1
        main_np1 = -2.0 * a_np1 - r
        sup_np1 = a_np1 + b_np1

        # RHS: (I + dtau/2 * L_n) * v^n
        v_int = V[1:-1, n]
        rhs = np.zeros(N_int)
        for i in range(N_int):
            rhs[i] = v_int[i] + 0.5 * dtau * main_n[i] * v_int[i]
            if i > 0:
                rhs[i] += 0.5 * dtau * sub_n[i] * v_int[i - 1]
            else:
                rhs[i] += 0.5 * dtau * sub_n[i] * V[0, n]
            if i < N_int - 1:
                rhs[i] += 0.5 * dtau * sup_n[i] * v_int[i + 1]
            else:
                rhs[i] += 0.5 * dtau * sup_n[i] * V[-1, n]

        # Boundary conditions at tau_{n+1}
        if barrier_type == 'down-out':
            V[0, n + 1] = 0.0    # barrier BC
            V[-1, n + 1] = (np.exp(m_hi) * np.exp(-q * tau_np1)
                            - np.exp(-r * tau_np1))   # deep ITM
        else:  # up-out
            V[0, n + 1] = 0.0    # deep OTM
            V[-1, n + 1] = 0.0   # barrier BC

        # Implicit boundary contributions
        rhs[0] -= (-0.5 * dtau * sub_np1[0]) * V[0, n + 1]
        rhs[-1] -= (-0.5 * dtau * sup_np1[-1]) * V[-1, n + 1]

        # LHS tridiagonal system
        ab = np.zeros((3, N_int))
        ab[1, :] = 1.0 - 0.5 * dtau * main_np1
        ab[0, 1:] = -0.5 * dtau * sup_np1[:-1]
        ab[2, :-1] = -0.5 * dtau * sub_np1[1:]

        V[1:-1, n + 1] = solve_banded((1, 1), ab, rhs)

    return m_grid, tau_grid, V


# ── Price extraction at specific (S, K, B, T) ──────────────────────────────

def price_barrier_fdm(S, K, B, T, r, q, sigma_func, barrier_type='down-out',
                      N_m=600, N_tau=600):
    """
    Price a single barrier option using FDM.

    Parameters
    ----------
    S, K     : spot, strike
    B        : barrier level
    T        : time to expiry
    r, q     : rates
    sigma_func : callable
    barrier_type : 'down-out' or 'up-out'

    Returns
    -------
    price : float (un-normalized: actual dollar price)
    """
    from scipy.interpolate import RegularGridInterpolator

    barrier_m = np.log(B / K)
    m_spot = np.log(S / K)

    # Domain bounds: extend enough beyond spot in the "safe" direction,
    # but NOT wastefully far. The barrier is the hard boundary on one side.
    if barrier_type == 'down-out':
        m_min_fdm = barrier_m
        # Extend the ITM side enough for the deep ITM BC to be accurate
        m_max_fdm = max(m_spot + 1.0, barrier_m + 2.0)
    else:  # up-out
        # For up-out, the barrier caps the right side.
        # Extend OTM side modestly -- the call value is ~0 for m << 0
        m_min_fdm = min(m_spot - 0.8, barrier_m - 2.0)
        m_max_fdm = barrier_m

    m_grid, tau_grid, V = crank_nicolson_barrier(
        sigma_func, r, q, barrier_m, m_spot, barrier_type,
        m_min=m_min_fdm, m_max=m_max_fdm,
        N_m=N_m, N_tau=N_tau, tau_max=T
    )

    # Interpolate to get v(m_spot, tau=T)
    interp = RegularGridInterpolator(
        (m_grid, tau_grid), V, method='linear',
        bounds_error=False, fill_value=0.0
    )
    v_norm = float(interp(np.array([[m_spot, T]]))[0])

    return v_norm * K   # un-normalize


# ── Self-test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from phase4_barriers.bs_barrier_closedform import (
        down_and_out_call, up_and_out_call, bs_call
    )

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.015, 0.20
    sf = sigma_constant(sigma)

    print("=" * 70)
    print("  Barrier FDM -- Validation against BS Closed-Form")
    print("=" * 70)

    # Vanilla check
    vanilla_bs = bs_call(S, K, T, r, q, sigma)
    print(f"\n  BS Vanilla Call: {vanilla_bs:.6f}")

    for label, B_pcts, btype, bs_fn in [
        ("Down-and-Out", [0.85, 0.90, 0.95], 'down-out', down_and_out_call),
        ("Up-and-Out",   [1.05, 1.10, 1.15], 'up-out',   up_and_out_call),
    ]:
        print(f"\n  --- {label} Call ---")
        for B_pct in B_pcts:
            B = B_pct * S
            bs_price = bs_fn(S, K, B, T, r, q, sigma)
            fdm_price = price_barrier_fdm(S, K, B, T, r, q, sf, btype,
                                          N_m=800, N_tau=800)
            rel_err = abs(fdm_price - bs_price) / max(bs_price, 1e-10) * 100
            print(f"    B={B:6.1f} | BS={bs_price:.6f} | FDM={fdm_price:.6f} "
                  f"| RelErr={rel_err:.4f}%")
