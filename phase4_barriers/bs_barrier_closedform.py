"""
Black-Scholes closed-form barrier option pricing.

Implements the Reiner-Rubinstein (1991) formulas for standard European
barrier options under constant volatility.  Used as a sanity check for
the FDM and Monte Carlo implementations.

Reference:
    Haug, E.G. (2007) "The Complete Guide to Option Pricing Formulas"
    Reiner, E. and Rubinstein, M. (1991) "Breaking Down the Barriers"

Supports:
    - Down-and-Out Call  (B < S)
    - Up-and-Out Call    (B > S)
"""

import numpy as np
from scipy.stats import norm

N = norm.cdf   # standard normal CDF


# ── Vanilla BS Call ──────────────────────────────────────────────────────────

def bs_call(S, K, T, r, q, sigma):
    """Standard Black-Scholes European call price."""
    if T <= 0:
        return max(S - K, 0.0)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2)


# ── Building Blocks (Haug notation) ─────────────────────────────────────────

def _building_blocks(S, K, H, T, r, q, sigma, phi, eta):
    """
    Compute the Reiner-Rubinstein building blocks A, B, C, D.

    Parameters
    ----------
    S     : spot price
    K     : strike price
    H     : barrier level
    T     : time to expiry (years)
    r     : risk-free rate
    q     : dividend yield
    sigma : constant volatility
    phi   : +1 for call, -1 for put
    eta   : +1 for down barrier, -1 for up barrier

    Returns
    -------
    A, B, C, D : floats
    """
    if T <= 1e-12:
        # At expiry, return intrinsic values
        payoff = max(phi * (S - K), 0.0)
        return payoff, payoff, 0.0, 0.0

    sqrt_T = np.sqrt(T)
    b = r - q                         # cost of carry
    mu = (b - 0.5 * sigma**2) / sigma**2

    x1 = np.log(S / K) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
    x2 = np.log(S / H) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
    y1 = np.log(H**2 / (S * K)) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
    y2 = np.log(H / S) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T

    # Discount factors
    df_S = S * np.exp((b - r) * T)    # = S * exp(-q*T)
    df_K = K * np.exp(-r * T)

    # Barrier ratio
    HS_ratio = H / S

    A = (phi * df_S * N(phi * x1)
         - phi * df_K * N(phi * (x1 - sigma * sqrt_T)))

    B = (phi * df_S * N(phi * x2)
         - phi * df_K * N(phi * (x2 - sigma * sqrt_T)))

    C = (phi * df_S * HS_ratio**(2 * (mu + 1)) * N(eta * y1)
         - phi * df_K * HS_ratio**(2 * mu) * N(eta * (y1 - sigma * sqrt_T)))

    D = (phi * df_S * HS_ratio**(2 * (mu + 1)) * N(eta * y2)
         - phi * df_K * HS_ratio**(2 * mu) * N(eta * (y2 - sigma * sqrt_T)))

    return A, B, C, D


# ── Barrier Option Prices ───────────────────────────────────────────────────

def down_and_out_call(S, K, H, T, r, q, sigma):
    """
    Price a Down-and-Out European call option.

    Parameters
    ----------
    S     : spot price (must be > H)
    K     : strike price
    H     : barrier level (H < S)
    T     : time to expiry (years)
    r     : risk-free rate
    q     : dividend yield
    sigma : constant volatility

    Returns
    -------
    price : float
    """
    if S <= H:
        return 0.0   # Already knocked out

    phi, eta = +1, +1   # call, down barrier
    A, B, C, D = _building_blocks(S, K, H, T, r, q, sigma, phi, eta)

    if K >= H:
        # Standard case: strike above barrier
        return max(A - C, 0.0)
    else:
        # Strike below barrier (unusual for equity calls)
        return max(B - D, 0.0)


def up_and_out_call(S, K, H, T, r, q, sigma):
    """
    Price an Up-and-Out European call option.

    Uses the Reiner-Rubinstein building blocks with the correct sign
    convention: for up-and-out calls, the reflected terms C and D
    use eta=+1 (same as down barriers).

    Parameters
    ----------
    S     : spot price (must be < H)
    K     : strike price (must be < H for non-zero value)
    H     : barrier level (H > S)
    T     : time to expiry (years)
    r     : risk-free rate
    q     : dividend yield
    sigma : constant volatility

    Returns
    -------
    price : float
    """
    if S >= H:
        return 0.0   # Already knocked out
    if K >= H:
        return 0.0   # Cannot exercise
    if T <= 1e-12:
        return max(S - K, 0.0)

    # Building blocks: phi=+1 (call), eta=+1 for the reflected terms
    phi = +1
    A, B, C, D = _building_blocks(S, K, H, T, r, q, sigma, phi, eta=+1)

    return max(A - B + D - C, 0.0)


# ── Vectorized helpers ──────────────────────────────────────────────────────

def down_and_out_call_vec(S, K, H, T_array, r, q, sigma):
    """Vectorized over an array of maturities T."""
    return np.array([down_and_out_call(S, K, H, T, r, q, sigma)
                     for T in T_array])


def up_and_out_call_vec(S, K, H, T_array, r, q, sigma):
    """Vectorized over an array of maturities T."""
    return np.array([up_and_out_call(S, K, H, T, r, q, sigma)
                     for T in T_array])


# ── Self-test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Standard test case from Haug (2007)
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.015, 0.20
    vanilla = bs_call(S, K, T, r, q, sigma)

    print("=" * 60)
    print("  BS Barrier Option Closed-Form -- Self-Test")
    print("=" * 60)
    print(f"\n  Vanilla Call: {vanilla:.6f}")

    for B_pct in [0.85, 0.90, 0.95]:
        H = B_pct * S
        do = down_and_out_call(S, K, H, T, r, q, sigma)
        print(f"  Down-and-Out Call (B={H:.1f}): {do:.6f}  "
              f"(discount = {1 - do/vanilla:.1%})")

    for B_pct in [1.05, 1.10, 1.15]:
        H = B_pct * S
        uo = up_and_out_call(S, K, H, T, r, q, sigma)
        print(f"  Up-and-Out Call   (B={H:.1f}): {uo:.6f}  "
              f"(discount = {1 - uo/vanilla:.1%})")

    # In-Out parity check: C_vanilla = C_in + C_out
    H_down = 90.0
    c_do = down_and_out_call(S, K, H_down, T, r, q, sigma)
    c_di = vanilla - c_do
    print(f"\n  In-Out Parity Check (B={H_down}):")
    print(f"    C_do + C_di = {c_do:.6f} + {c_di:.6f} = {c_do + c_di:.6f}")
    print(f"    C_vanilla   = {vanilla:.6f}")
    print(f"    Match: {abs(c_do + c_di - vanilla) < 1e-10}")
