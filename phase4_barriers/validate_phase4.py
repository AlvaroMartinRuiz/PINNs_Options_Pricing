"""
Phase 4 -- Barrier Option Validation & Thesis Figure Generation.

Runs FDM, MC, and PINN barrier option pricing using the calibrated
local volatility from Phase 3. Generates publication-quality figures.

Usage:
    python -m phase4_barriers.validate_phase4
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from phase4_barriers.bs_barrier_closedform import (
    down_and_out_call, up_and_out_call, bs_call
)
from phase4_barriers.barrier_fdm import (
    sigma_from_pinn, sigma_constant, price_barrier_fdm, crank_nicolson_barrier
)
from phase4_barriers.barrier_mc import monte_carlo_barrier
from phase4_barriers.barrier_pinn import (
    train_barrier_pinn, price_barrier_pinn, compute_greeks, BarrierPriceNet
)
from utils.normalization import LogMoneynessNormalizer

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'phase4')
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Phase 3 calibrated parameters (SPY 2021-09-20)
S0 = 434.01
K_ATM = 434.0   # ATM strike
T = 0.5          # 6-month maturity

# Barrier levels
DOWN_BARRIERS = [0.85, 0.90, 0.95]  # fractions of S0
UP_BARRIERS = [1.05, 1.10, 1.15]


# ═══════════════════════════════════════════════════════════════════════════
#  PART 1: LOAD CALIBRATED VOLATILITY
# ═══════════════════════════════════════════════════════════════════════════

def load_phase3():
    """Load the Phase 3 PINN volatility surface."""
    model_path = os.path.join(os.path.dirname(__file__), '..',
                              'results', 'phase3', 'pinn_phase3.pt')
    sigma_func, domain, params = sigma_from_pinn(model_path)
    print(f"  Loaded Phase 3 PINN: r={params['r']:.4f}, q={params['q']:.4f}")
    print(f"  Domain: m in [{domain['m_min']:.3f}, {domain['m_max']:.3f}], "
          f"tau_max={domain['tau_max']:.3f}")
    return sigma_func, domain, params


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2: THREE-WAY PRICE COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════

def run_price_comparison(sigma_func, r, q, sigma_const=0.20):
    """
    Run FDM, MC, and BS closed-form for all barrier configurations.
    """
    print("\n" + "=" * 75)
    print("  THREE-WAY BARRIER PRICE COMPARISON")
    print("=" * 75)

    results = []
    sf_const = sigma_constant(sigma_const)

    for barrier_type, barriers_pct, bs_fn, label in [
        ('down-out', DOWN_BARRIERS, down_and_out_call, 'Down-Out Call'),
        ('up-out',   UP_BARRIERS,   up_and_out_call,   'Up-Out Call'),
    ]:
        print(f"\n  --- {label} (S0={S0:.0f}, K={K_ATM:.0f}, T={T}) ---")
        print(f"  {'B':>7s} | {'BS(const)':>10s} | {'FDM(LV)':>10s} | "
              f"{'MC(LV)':>14s} | {'Diff%':>7s}")
        print("  " + "-" * 65)

        for B_pct in barriers_pct:
            B = B_pct * S0

            # BS closed-form (constant vol reference)
            bs_price = bs_fn(S0, K_ATM, B, T, r, q, sigma_const)

            # FDM with calibrated LV
            fdm_price = price_barrier_fdm(S0, K_ATM, B, T, r, q,
                                          sigma_func, barrier_type,
                                          N_m=600, N_tau=600)

            # MC with calibrated LV
            mc_result = monte_carlo_barrier(
                sigma_func, S0, K_ATM, B, T, r, q, barrier_type,
                n_paths=100_000, n_steps=500,
                antithetic=True, bgk_correction=True,
                use_precomputed=True
            )
            mc_price = mc_result['price']
            mc_se = mc_result['se']

            # Difference: LV vs constant vol
            diff_pct = (fdm_price - bs_price) / max(abs(bs_price), 1e-10) * 100

            print(f"  {B:7.1f} | {bs_price:10.4f} | {fdm_price:10.4f} | "
                  f"{mc_price:8.4f}+/-{mc_se:.4f} | {diff_pct:+7.1f}%")

            results.append({
                'type': barrier_type, 'B': B, 'B_pct': B_pct,
                'bs': bs_price, 'fdm': fdm_price,
                'mc': mc_price, 'mc_se': mc_se,
            })

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3: BARRIER SENSITIVITY PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_barrier_sensitivity(sigma_func, r, q):
    """
    Plot barrier option price as a function of barrier level.
    Key thesis figure: shows the price collapsing as B -> S0.
    """
    print("\n  Generating barrier sensitivity plot...")

    B_pcts_down = np.linspace(0.80, 0.99, 20)
    B_pcts_up = np.linspace(1.01, 1.20, 20)

    prices_down = []
    prices_up = []

    for B_pct in B_pcts_down:
        B = B_pct * S0
        p = price_barrier_fdm(S0, K_ATM, B, T, r, q, sigma_func, 'down-out',
                              N_m=400, N_tau=400)
        prices_down.append(p)

    for B_pct in B_pcts_up:
        B = B_pct * S0
        p = price_barrier_fdm(S0, K_ATM, B, T, r, q, sigma_func, 'up-out',
                              N_m=400, N_tau=400)
        prices_up.append(p)

    # Vanilla reference
    vanilla_fdm = price_barrier_fdm(S0, K_ATM, 0.5*S0, T, r, q,
                                     sigma_func, 'down-out', N_m=400, N_tau=400)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(B_pcts_down * 100, prices_down, 'b-o', markersize=4, linewidth=2)
    ax1.axhline(vanilla_fdm, color='gray', linestyle='--', alpha=0.7,
                label=f'Vanilla Call ({vanilla_fdm:.2f})')
    ax1.set_xlabel('Barrier Level (% of Spot)', fontsize=12)
    ax1.set_ylabel('Option Price ($)', fontsize=12)
    ax1.set_title('Down-and-Out Call: Barrier Sensitivity', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(B_pcts_up * 100, prices_up, 'r-o', markersize=4, linewidth=2)
    ax2.set_xlabel('Barrier Level (% of Spot)', fontsize=12)
    ax2.set_ylabel('Option Price ($)', fontsize=12)
    ax2.set_title('Up-and-Out Call: Barrier Sensitivity', fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'barrier_sensitivity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 4: BARRIER PRICE SURFACE (3D)
# ═══════════════════════════════════════════════════════════════════════════

def plot_barrier_surface_3d(sigma_func, r, q):
    """
    Generate 3D surface plots of the barrier option price v(K/S, tau).
    """
    print("\n  Generating 3D barrier price surfaces...")

    B_down = 0.90 * S0
    B_up = 1.10 * S0

    taus = np.linspace(0.05, 1.0, 20)
    moneyness = np.linspace(0.85, 1.15, 30)

    fig = plt.figure(figsize=(16, 5))

    for idx, (B, btype, title) in enumerate([
        (B_down, 'down-out', f'Down-and-Out Call (B={B_down:.0f})'),
        (B_up, 'up-out', f'Up-and-Out Call (B={B_up:.0f})'),
        (None, None, 'Vanilla Call'),
    ]):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        TAU, MON = np.meshgrid(taus, moneyness)
        prices = np.zeros_like(TAU)

        for i in range(len(moneyness)):
            for j in range(len(taus)):
                K_val = moneyness[i] * S0
                tau_val = taus[j]

                if btype is None:
                    # Vanilla: use down-out with very low barrier
                    p = price_barrier_fdm(S0, K_val, 0.5*S0, tau_val, r, q,
                                          sigma_func, 'down-out', N_m=200, N_tau=200)
                else:
                    p = price_barrier_fdm(S0, K_val, B, tau_val, r, q,
                                          sigma_func, btype, N_m=200, N_tau=200)
                prices[i, j] = max(p, 0)

        surf = ax.plot_surface(TAU, MON, prices, cmap='viridis', alpha=0.9)
        ax.set_xlabel('Tau', fontsize=9)
        ax.set_ylabel('K/S', fontsize=9)
        ax.set_zlabel('Price ($)', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'barrier_surfaces_3d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 5: GREEKS COMPARISON (PINN autograd vs FDM bump-and-reset)
# ═══════════════════════════════════════════════════════════════════════════

def plot_greeks_comparison(sigma_func, r, q):
    """
    Compare PINN autograd Greeks vs FDM bump-and-reset Greeks.
    Key thesis figure: shows the noise-free advantage of PINN near the barrier.
    """
    print("\n  Training PINN for Greeks demonstration...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 0.90 * S0
    barrier_m = np.log(B / K_ATM)
    normalizer = LogMoneynessNormalizer(m_scale=0.5, tau_max=T)

    # Train PINN with calibrated LV
    m_spot = np.log(S0 / K_ATM)
    model, _ = train_barrier_pinn(
        barrier_m=barrier_m, barrier_type='down-out',
        sigma_func_frozen=sigma_func,
        r=r, q=q, m_spot=m_spot,
        m_domain=(barrier_m, 0.3),
        tau_max=T, normalizer=normalizer, device=device,
        n_epochs=5000, lr=1e-3, n_pde=2000, n_ic=500,
    )

    # Compute Greeks over a range of spot prices
    S_range = np.linspace(B + 1, 1.15 * S0, 60)

    # PINN Greeks (analytical via autograd)
    pinn_deltas = []
    pinn_gammas = []
    for S_val in S_range:
        d, g = compute_greeks(model, normalizer, S_val, K_ATM, T)
        pinn_deltas.append(d)
        pinn_gammas.append(g)

    # FDM Greeks (bump-and-reset)
    dS = 0.5  # bump size
    fdm_deltas = []
    fdm_gammas = []
    for S_val in S_range:
        if S_val - dS <= B:
            fdm_deltas.append(np.nan)
            fdm_gammas.append(np.nan)
            continue
        V_up = price_barrier_fdm(S_val + dS, K_ATM, B, T, r, q,
                                  sigma_func, 'down-out', N_m=300, N_tau=300)
        V_dn = price_barrier_fdm(S_val - dS, K_ATM, B, T, r, q,
                                  sigma_func, 'down-out', N_m=300, N_tau=300)
        V_mid = price_barrier_fdm(S_val, K_ATM, B, T, r, q,
                                   sigma_func, 'down-out', N_m=300, N_tau=300)
        delta = (V_up - V_dn) / (2 * dS)
        gamma = (V_up - 2 * V_mid + V_dn) / (dS ** 2)
        fdm_deltas.append(delta)
        fdm_gammas.append(gamma)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Delta
    ax1.plot(S_range / S0, pinn_deltas, 'b-', linewidth=2, label='PINN (Autograd)')
    ax1.plot(S_range / S0, fdm_deltas, 'r--', linewidth=1.5, label='FDM (Bump-Reset)')
    ax1.axvline(B / S0, color='gray', linestyle=':', alpha=0.5, label=f'Barrier ({B/S0:.0%})')
    ax1.set_xlabel('S / S0', fontsize=12)
    ax1.set_ylabel('Delta', fontsize=12)
    ax1.set_title('Delta: PINN vs FDM', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Gamma
    ax2.plot(S_range / S0, pinn_gammas, 'b-', linewidth=2, label='PINN (Autograd)')
    ax2.plot(S_range / S0, fdm_gammas, 'r--', linewidth=1.5, label='FDM (Bump-Reset)')
    ax2.axvline(B / S0, color='gray', linestyle=':', alpha=0.5, label=f'Barrier ({B/S0:.0%})')
    ax2.set_xlabel('S / S0', fontsize=12)
    ax2.set_ylabel('Gamma', fontsize=12)
    ax2.set_title('Gamma: PINN vs FDM (Near Barrier)', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'greeks_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return model, normalizer


# ═══════════════════════════════════════════════════════════════════════════
#  PART 6: PRICING SPEED BENCHMARK (PINN vs FDM)
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_pricing_speed(model, normalizer, sigma_func, r, q):
    """
    Compare the execution time of pricing a batch of options using
    the trained PINN (fast forward pass) vs the FDM solver (sequential grid).
    """
    print("\n" + "=" * 75)
    print("  PRICING SPEED BENCHMARK: PINN vs FDM")
    print("=" * 75)

    B = 0.90 * S0
    N_options = 1000
    
    # Generate random spot prices and maturities
    np.random.seed(42)
    spots = np.random.uniform(B * 1.05, S0 * 1.2, N_options)
    maturities = np.random.uniform(0.1, 1.0, N_options)
    
    print(f"  Benchmarking pricing of {N_options} different barrier options...")

    # 1. FDM Benchmark
    # FDM has to be solved iteratively in time for each individual configuration
    print("  Running FDM (this may take a moment)...")
    t0_fdm = time.time()
    for i in range(10):  # Only run 10 and extrapolate because FDM is slow
        _ = price_barrier_fdm(spots[i], K_ATM, B, maturities[i], r, q,
                              sigma_func, 'down-out', N_m=200, N_tau=200)
    t1_fdm = time.time()
    fdm_time_per_option = (t1_fdm - t0_fdm) / 10
    fdm_total_est = fdm_time_per_option * N_options

    # 2. PINN Benchmark
    # PINN can price all 1000 options simultaneously in a single batched tensor operation
    print("  Running PINN (Batched)...")
    device = next(model.parameters()).device
    
    m_phys = torch.tensor(np.log(spots / K_ATM), dtype=torch.float32, device=device)
    tau_t = torch.tensor(maturities, dtype=torch.float32, device=device)
    
    t0_pinn = time.time()
    m_n, tau_n = normalizer.normalize(m_phys, tau_t)
    with torch.no_grad():
        _ = model(m_n, tau_n, m_phys)
    t1_pinn = time.time()
    
    pinn_total_time = t1_pinn - t0_pinn
    pinn_time_per_option = pinn_total_time / N_options
    
    speedup = fdm_total_est / max(pinn_total_time, 1e-10)

    print(f"\n  Results ({N_options} options):")
    print(f"    FDM Time (Estimated): {fdm_total_est:.4f} seconds ({fdm_time_per_option*1000:.2f} ms/option)")
    print(f"    PINN Time (Batched):  {pinn_total_time:.4f} seconds ({pinn_time_per_option*1000000:.2f} microseconds/option)")
    print(f"    Speedup Factor:       {speedup:,.0f}x faster")
    print("\n  Conclusion: PINN shifts computational cost entirely to the offline training")
    print("  phase. Once trained, real-time pricing and hedging is virtually instantaneous.")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 75)
    print("  PHASE 4: Barrier Option Pricing with Calibrated Local Volatility")
    print("=" * 75)

    t_start = time.time()

    # Load Phase 3 calibrated LV
    sigma_func, domain, params = load_phase3()
    r = params['r']
    q = params['q']

    # Part 2: Price comparison table
    results = run_price_comparison(sigma_func, r, q)

    # Part 3: Barrier sensitivity plot
    plot_barrier_sensitivity(sigma_func, r, q)

    # Part 4: 3D surface plots
    plot_barrier_surface_3d(sigma_func, r, q)

    # Part 5: Greeks comparison
    model, normalizer = plot_greeks_comparison(sigma_func, r, q)

    # Part 6: Pricing speed benchmark
    benchmark_pricing_speed(model, normalizer, sigma_func, r, q)

    elapsed = time.time() - t_start
    print(f"\n{'='*75}")
    print(f"  Phase 4 validation complete in {elapsed/60:.1f} minutes")
    print(f"  Plots saved to: {OUT_DIR}")
    print(f"{'='*75}")
