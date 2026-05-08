"""
Phase 3 — Market Data Loader.

Loads the preprocessed SPY options CSV and converts to PINN-ready tensors
in log-moneyness coordinates.

Key responsibilities:
    1. Load CSV and compute m = ln(S/K)
    2. Convert OTM put prices to call-equivalent via put-call parity
    3. Compute vega-weighted loss weights (with 95th-percentile cap)
    4. Run arbitrage diagnostics on the data
"""

import numpy as np
import pandas as pd
import os


# ── Black-Scholes Vega ──────────────────────────────────────────────────────

def _compute_bs_vega(S, K, r, q, sigma, tau):
    """
    Compute Black-Scholes vega for each option.

    Vega = S * exp(-q*tau) * phi(d1) * sqrt(tau)
    where phi is the standard normal pdf.

    Same formula for calls and puts.
    """
    sqrt_tau = np.sqrt(np.maximum(tau, 1e-12))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    phi_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    vega = S * np.exp(-q * tau) * phi_d1 * sqrt_tau
    return vega


# ── Put-Call Parity Conversion ──────────────────────────────────────────────

def _convert_to_call_prices(V_market, S, K, r, q, tau, option_type):
    """
    Convert all option prices to normalized call prices v = C/K.

    For OTM calls (K > S):
        v = C / K = V_market / K

    For OTM puts (K < S), use put-call parity:
        C = P + S*exp(-q*tau) - K*exp(-r*tau)
        v = C/K = P/K + (S/K)*exp(-q*tau) - exp(-r*tau)

    This ensures the PINN output is always a normalized call price,
    consistent with the payoff condition v(m, 0) = max(exp(m) - 1, 0).
    """
    v = np.zeros_like(V_market, dtype=np.float64)

    call_mask = (option_type == 'call')
    put_mask = (option_type == 'put')

    # Calls: direct normalization
    v[call_mask] = V_market[call_mask] / K[call_mask]

    # Puts: put-call parity conversion
    v[put_mask] = (V_market[put_mask] / K[put_mask]
                   + (S[put_mask] / K[put_mask]) * np.exp(-q[put_mask] * tau[put_mask])
                   - np.exp(-r[put_mask] * tau[put_mask]))

    return v


# ── Vega Weights ────────────────────────────────────────────────────────────

def _compute_vega_weights(S, K, r, q, sigma, tau, epsilon=1e-6, cap_pctl=95):
    """
    Compute inverse-vega weights with percentile cap.

    Deep OTM short-dated options have near-zero vega, producing extreme
    inverse weights that would hijack the loss function.  The cap at the
    95th percentile prevents any single outlier from dominating.

    Returns weights normalised so that mean(w) = 1.
    """
    vega = _compute_bs_vega(S, K, r, q, sigma, tau)
    raw_w = 1.0 / (vega + epsilon)

    # Adaptive cap at 95th percentile
    cap = np.percentile(raw_w, cap_pctl)
    w = np.clip(raw_w, None, cap)

    # Normalize so mean = 1 (keeps loss scale comparable to unweighted MSE)
    w /= w.mean()

    return w


# ── Arbitrage Diagnostics ───────────────────────────────────────────────────

def _diagnose_arbitrage_violations(df, v_call):
    """
    Scan the dataset for common no-arbitrage violations.

    After put-call parity conversion, all prices are normalized call prices.
    For European calls:
        1. Calendar: for a fixed strike, v should increase with tau
        2. Butterfly: for a fixed expiry, call price C should decrease with K

    Reports counts and worst offenders to inform lambda_arb tuning.
    """
    print("\n  -- Arbitrage Diagnostics --")

    # Add v_call to a working copy
    work = df[['K', 'DTE', 'tau', 'EXPIRE_DATE']].copy()
    work['v_call'] = v_call
    work['C'] = v_call * df['K'].values  # un-normalized call price

    # --- 1. Calendar violations ---
    # For each unique strike, call prices should increase with tau
    calendar_violations = 0
    calendar_total_pairs = 0
    unique_strikes = work['K'].unique()

    for K_val in unique_strikes:
        chain = work[work['K'] == K_val].sort_values('tau')
        if len(chain) < 2:
            continue
        C_vals = chain['C'].values
        diffs = np.diff(C_vals)
        n_violations = np.sum(diffs < -1e-8)
        calendar_violations += n_violations
        calendar_total_pairs += len(diffs)

    pct = 100.0 * calendar_violations / max(calendar_total_pairs, 1)
    print(f"    Calendar violations: {calendar_violations}/{calendar_total_pairs} "
          f"pairs ({pct:.1f}%)")

    # --- 2. Butterfly violations ---
    # For each expiry, call prices C should decrease with K
    butterfly_violations = 0
    butterfly_total_pairs = 0
    unique_expiries = work['EXPIRE_DATE'].unique()

    for exp in unique_expiries:
        chain = work[work['EXPIRE_DATE'] == exp].sort_values('K')
        if len(chain) < 2:
            continue
        C_vals = chain['C'].values
        diffs = np.diff(C_vals)
        n_violations = np.sum(diffs > 1e-8)  # C should decrease with K
        butterfly_violations += n_violations
        butterfly_total_pairs += len(diffs)

    pct = 100.0 * butterfly_violations / max(butterfly_total_pairs, 1)
    print(f"    Butterfly violations: {butterfly_violations}/{butterfly_total_pairs} "
          f"pairs ({pct:.1f}%)")

    if calendar_violations + butterfly_violations == 0:
        print("    OK: No arbitrage violations detected in the data.")
    else:
        total = calendar_violations + butterfly_violations
        print(f"    WARNING: {total} total violations -- lambda_arb=1.0 should handle these softly.")

    return calendar_violations, butterfly_violations


# ── Main Data Loader ────────────────────────────────────────────────────────

def load_market_data(csv_path, verbose=True):
    """
    Load preprocessed SPY options CSV and prepare PINN-ready data.

    Parameters
    ----------
    csv_path : str
        Path to the preprocessed CSV (e.g., spy_otm_2021-09-20.csv)
    verbose : bool
        Print summary statistics and diagnostics

    Returns
    -------
    data : dict with keys:
        m           : (N,) ndarray — log-moneyness ln(S/K)
        tau         : (N,) ndarray — time to maturity (years)
        v_call      : (N,) ndarray — normalized call price (after put-call parity)
        r           : (N,) ndarray — risk-free rate (per option)
        q           : (N,) ndarray — dividend yield (per option)
        vega_weights: (N,) ndarray — inverse-vega weights (capped, normalised)
        sigma_market: (N,) ndarray — market implied volatility
        option_type : (N,) ndarray — 'call' or 'put'
        moneyness   : (N,) ndarray — K/S ratio
        S0          : float        — underlying price
        K           : (N,) ndarray — strike prices
        DTE         : (N,) ndarray — days to expiration
    """
    if verbose:
        print("=" * 70)
        print("Loading market data for PINN training")
        print("=" * 70)

    df = pd.read_csv(csv_path)

    S = df['S'].values.astype(np.float64)
    K = df['K'].values.astype(np.float64)
    tau = df['tau'].values.astype(np.float64)
    r = df['r'].values.astype(np.float64)
    q = df['q'].values.astype(np.float64)
    sigma_market = df['sigma_market'].values.astype(np.float64)
    V_market = df['V_market'].values.astype(np.float64)
    option_type = df['option_type'].values

    # 1. Log-moneyness
    m = np.log(S / K)

    # 2. Put-call parity conversion → normalized call prices
    v_call = _convert_to_call_prices(V_market, S, K, r, q, tau, option_type)

    # 3. Vega weights (inverse, capped at 95th percentile)
    vega_weights = _compute_vega_weights(S, K, r, q, sigma_market, tau)

    # 4. Arbitrage diagnostics
    if verbose:
        _diagnose_arbitrage_violations(df, v_call)

    data = {
        'm': m,
        'tau': tau,
        'v_call': v_call,
        'r': r,
        'q': q,
        'vega_weights': vega_weights,
        'sigma_market': sigma_market,
        'option_type': option_type,
        'moneyness': df['moneyness'].values.astype(np.float64),
        'S0': S[0],
        'K': K,
        'DTE': df['DTE'].values,
    }

    if verbose:
        n_calls = np.sum(option_type == 'call')
        n_puts = np.sum(option_type == 'put')
        print(f"  -- Dataset Summary --")
        print(f"    Total options:  {len(m)} ({n_calls} calls, {n_puts} puts)")
        print(f"    Underlying S0:  ${S[0]:.2f}")
        print(f"    m (log-money):  [{m.min():.4f}, {m.max():.4f}]")
        print(f"    tau (years):    [{tau.min():.4f}, {tau.max():.4f}]")
        print(f"    v_call range:   [{v_call.min():.6f}, {v_call.max():.6f}]")
        print(f"    r range:        [{r.min()*100:.4f}%, {r.max()*100:.4f}%]")
        print(f"    q (constant):   {q[0]*100:.2f}%")
        print(f"    sigma_market:   [{sigma_market.min():.4f}, {sigma_market.max():.4f}]")
        print(f"    Vega weights:   [{vega_weights.min():.4f}, {vega_weights.max():.4f}]"
              f"  (mean=1.00)")

    return data


# ── Standalone test ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, 'spy_otm_2021-09-20.csv')
    data = load_market_data(csv_path)
    print("\n  Data loaded successfully.")
    print(f"  Keys: {list(data.keys())}")
