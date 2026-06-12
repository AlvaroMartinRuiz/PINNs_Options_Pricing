"""
Phase 4 — Monte Carlo pricer for barrier options under local volatility.

Features:
    - Euler-Maruyama discretization of the SDE in log-spot coordinates
    - Pre-computed PINN volatility grid for fast interpolation
    - Antithetic variates for variance reduction
    - Broadie-Glasserman-Kou (1997) continuity correction for discrete
      monitoring bias

Usage:
    python -m phase4_barriers.barrier_mc
"""

import numpy as np
import os, sys
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ── Pre-compute the PINN vol surface for fast MC lookup ─────────────────────

def build_sigma_interpolator(sigma_func, m_min=-0.5, m_max=0.5,
                             tau_min=0.001, tau_max=1.0,
                             N_m=200, N_tau=100):
    """
    Pre-evaluate the PINN local vol on a dense regular grid and return
    a fast 2D interpolator.

    In the MC simulation, we need sigma_loc(S_t, t).  In Dupire's framework:
        sigma_loc(S, t) = sigma_Dupire(K=S, T=t)
    In PINN coordinates (m = ln(S_0/K), tau = T):
        K = S  =>  m = ln(S_0/S)  and  tau = T - t  (time remaining in PDE sense)
    But since the MC simulates forward in time and the PINN's tau is
    time-to-maturity, we need:
        m = ln(S_0 / S_t)   (note: negative when S_t > S_0)
        tau = ... (the maturity of the option, not the simulation time)

    IMPORTANT: The PINN sigma(m, tau) gives the instantaneous local vol
    at the point (m, tau) in the PDE grid.  For MC simulation of a specific
    option, we evaluate sigma at m = ln(S_0/S_t) and at the current
    time-to-maturity tau_remaining = T - t.

    Parameters
    ----------
    sigma_func : callable(m_array, tau_scalar) -> sigma_array
    m_min, m_max : range of log-moneyness to pre-compute
    tau_min, tau_max : range of maturities
    N_m, N_tau : grid resolution

    Returns
    -------
    interp : RegularGridInterpolator for sigma(m, tau)
    """
    m_grid = np.linspace(m_min, m_max, N_m)
    tau_grid = np.linspace(tau_min, tau_max, N_tau)

    sigma_grid = np.zeros((N_m, N_tau))
    for j in range(N_tau):
        sigma_grid[:, j] = sigma_func(m_grid, tau_grid[j])

    # Clip to physically reasonable range
    sigma_grid = np.clip(sigma_grid, 0.01, 5.0)

    interp = RegularGridInterpolator(
        (m_grid, tau_grid), sigma_grid,
        method='linear', bounds_error=False, fill_value=None
    )
    return interp, m_grid, tau_grid


# ── Monte Carlo Engine ──────────────────────────────────────────────────────

def monte_carlo_barrier(sigma_func, S0, K, B, T, r, q,
                        barrier_type='down-out',
                        n_paths=100_000, n_steps=500,
                        antithetic=True, bgk_correction=True,
                        seed=42, use_precomputed=True):
    """
    Price a barrier option via Monte Carlo under local volatility.

    Simulates the SDE:
        d(ln S) = (r - q - sigma^2/2) dt + sigma dW

    where sigma = sigma_loc(S_t, t) is evaluated from the calibrated
    PINN volatility surface.

    Parameters
    ----------
    sigma_func  : callable or RegularGridInterpolator
    S0          : spot price
    K           : strike
    B           : barrier level
    T           : time to expiry
    r, q        : risk-free rate, dividend yield
    barrier_type: 'down-out' or 'up-out'
    n_paths     : number of MC paths
    n_steps     : number of Euler time steps
    antithetic  : use antithetic variates
    bgk_correction : apply Broadie-Glasserman-Kou correction
    seed        : random seed

    Returns
    -------
    result : dict with price, se, n_survived, survival_rate
    """
    np.random.seed(seed)
    dt = T / n_steps

    # Build fast interpolator if sigma_func is a regular callable
    if use_precomputed and callable(sigma_func) and not isinstance(sigma_func, RegularGridInterpolator):
        interp, _, _ = build_sigma_interpolator(
            sigma_func,
            m_min=-0.6, m_max=0.6,
            tau_min=dt, tau_max=T,
            N_m=200, N_tau=100
        )
    elif isinstance(sigma_func, RegularGridInterpolator):
        interp = sigma_func
    else:
        interp = None

    def get_sigma(S_arr, tau_remaining):
        """Get local vol for an array of spot prices at a given tau."""
        # In Dupire framework: sigma_loc(S, t) = sigma(m=ln(S0/S), tau=T-t)
        # We use ln(S0/S_t) = -ln(S_t/S0) for the moneyness coordinate
        m_arr = np.log(S0 / S_arr)
        tau_arr = np.full_like(m_arr, max(tau_remaining, 1e-6))
        if interp is not None:
            pts = np.column_stack([m_arr, tau_arr])
            sig = interp(pts)
        else:
            sig = sigma_func(m_arr, tau_remaining)
        return np.clip(sig, 0.01, 5.0)

    # Initialize paths
    effective_paths = n_paths
    if antithetic:
        half = n_paths // 2
        effective_paths = 2 * half
    else:
        half = n_paths
        effective_paths = n_paths

    S = np.full(effective_paths, S0)
    alive = np.ones(effective_paths, dtype=bool)

    # Simulate forward in time
    for step in range(n_steps):
        t_now = step * dt
        tau_remaining = T - t_now

        # Get local vol at current (S, t)
        sigma = get_sigma(S[alive], tau_remaining)

        # Generate random numbers
        n_alive = alive.sum()
        if antithetic and step == 0:
            Z_half = np.random.randn(half)
            Z = np.concatenate([Z_half, -Z_half])
            # Shuffle to avoid correlation artifacts
        else:
            Z = np.random.randn(n_alive)

        if antithetic and step == 0:
            Z_alive = Z[:n_alive]
        else:
            Z_alive = Z

        # Euler step in log-spot
        drift = (r - q - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z_alive
        S[alive] *= np.exp(drift + diffusion)

        # Check barrier crossing with BGK correction
        if barrier_type == 'down-out':
            if bgk_correction:
                # BGK: shift barrier up to account for discrete monitoring
                B_shift = B * np.exp(0.5826 * sigma * np.sqrt(dt))
            else:
                B_shift = np.full(n_alive, B)
            knocked = S[alive] <= B_shift
        else:  # up-out
            if bgk_correction:
                B_shift = B * np.exp(-0.5826 * sigma * np.sqrt(dt))
            else:
                B_shift = np.full(n_alive, B)
            knocked = S[alive] >= B_shift

        # Kill knocked-out paths
        alive_indices = np.where(alive)[0]
        alive[alive_indices[knocked]] = False

    # Compute payoffs for surviving paths
    payoff = np.zeros(effective_paths)
    payoff[alive] = np.maximum(S[alive] - K, 0.0)  # call payoff

    # Discount and compute statistics
    disc_payoff = np.exp(-r * T) * payoff
    price = np.mean(disc_payoff)
    se = np.std(disc_payoff) / np.sqrt(effective_paths)
    survival_rate = alive.sum() / effective_paths

    return {
        'price': price,
        'se': se,
        'n_survived': alive.sum(),
        'n_paths': effective_paths,
        'survival_rate': survival_rate,
    }


# ── Self-test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from phase4_barriers.bs_barrier_closedform import (
        down_and_out_call, up_and_out_call, bs_call
    )
    from phase4_barriers.barrier_fdm import sigma_constant

    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.015, 0.20
    sf = sigma_constant(sigma)

    print("=" * 70)
    print("  Barrier MC -- Validation against BS Closed-Form")
    print("=" * 70)

    vanilla = bs_call(S, K, T, r, q, sigma)
    print(f"\n  BS Vanilla Call: {vanilla:.6f}")

    for label, B_pcts, btype, bs_fn in [
        ("Down-and-Out", [0.85, 0.90, 0.95], 'down-out', down_and_out_call),
        ("Up-and-Out",   [1.05, 1.10, 1.15], 'up-out',   up_and_out_call),
    ]:
        print(f"\n  --- {label} Call ---")
        for B_pct in B_pcts:
            B_val = B_pct * S
            bs_price = bs_fn(S, K, B_val, T, r, q, sigma)
            result = monte_carlo_barrier(
                sf, S, K, B_val, T, r, q, btype,
                n_paths=200_000, n_steps=500,
                antithetic=True, bgk_correction=True,
                use_precomputed=False
            )
            mc_price = result['price']
            mc_se = result['se']
            rel_err = abs(mc_price - bs_price) / max(bs_price, 1e-10) * 100
            within_2se = abs(mc_price - bs_price) < 2 * mc_se
            print(f"    B={B_val:6.1f} | BS={bs_price:.4f} | "
                  f"MC={mc_price:.4f}+/-{mc_se:.4f} | "
                  f"RelErr={rel_err:.2f}% | 2sig={'OK' if within_2se else 'X'} | "
                  f"surv={result['survival_rate']:.1%}")
