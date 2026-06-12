"""
Phase 3 — Dupire Benchmark Comparison.

Generates a 3-panel figure comparing:
    A) Raw FDM Dupire (central finite differences on RBF-interpolated IV)
    B) Smoothed Dupire (SVI-parametrised IV, then Dupire)
    C) PINN local volatility surface

Usage:
    python -m phase3_market.benchmark_dupire
"""

import sys, os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase2_inverse.pinn_dual import PINN_Dual
from utils.normalization import LogMoneynessNormalizer
from phase3_market.data_loader import load_market_data


# ═══════════════════════════════════════════════════════════════════════════
#  PART 1:  Extract raw market (k, tau, IV) scatter data
# ═══════════════════════════════════════════════════════════════════════════

def _get_market_scatter(market_data):
    """
    Return the raw scatter of (k, tau, IV) from market data,
    averaging duplicates.
    """
    m   = market_data['m']
    tau = market_data['tau']
    sig = market_data['sigma_market']
    k   = -m   # k = ln(K/S)

    df = pd.DataFrame({'k': k, 'tau': tau, 'sigma': sig})
    df = df.groupby(['k', 'tau'], as_index=False).mean()
    return df['k'].values, df['tau'].values, df['sigma'].values


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2:  Dupire FDM engine (operates on any W(k,tau) grid)
# ═══════════════════════════════════════════════════════════════════════════

def _dupire_from_total_variance(k_grid, tau_grid, W):
    """
    Apply the discrete Dupire formula via finite differences on a regular
    total-variance surface W(k, tau) = sigma_IV^2 * tau.

    Returns sigma2_loc (Nk, Nt) — local variance.  May contain NaN or
    negative values where the formula is ill-posed.
    """
    Nk, Nt = W.shape
    sigma2_loc = np.full((Nk, Nt), np.nan)

    for j in range(Nt):
        for i in range(Nk):
            # ── dw/dtau ──
            if j == 0:
                dw_dtau = (W[i, j+1] - W[i, j]) / (tau_grid[j+1] - tau_grid[j])
            elif j == Nt - 1:
                dw_dtau = (W[i, j] - W[i, j-1]) / (tau_grid[j] - tau_grid[j-1])
            else:
                dw_dtau = (W[i, j+1] - W[i, j-1]) / (tau_grid[j+1] - tau_grid[j-1])

            # ── dw/dk ──
            if i == 0:
                dw_dk = (W[i+1, j] - W[i, j]) / (k_grid[i+1] - k_grid[i])
            elif i == Nk - 1:
                dw_dk = (W[i, j] - W[i-1, j]) / (k_grid[i] - k_grid[i-1])
            else:
                dw_dk = (W[i+1, j] - W[i-1, j]) / (k_grid[i+1] - k_grid[i-1])

            # ── d2w/dk2 ──
            if i == 0 or i == Nk - 1:
                d2w_dk2 = 0.0
            else:
                h1 = k_grid[i] - k_grid[i-1]
                h2 = k_grid[i+1] - k_grid[i]
                d2w_dk2 = (2.0 * (W[i+1, j]*h1 + W[i-1, j]*h2
                           - W[i, j]*(h1+h2)) / (h1*h2*(h1+h2)))

            w = W[i, j]
            kv = k_grid[i]

            if w < 1e-12:
                continue

            denom = (1.0
                     - (kv / w) * dw_dk
                     + 0.25 * (-0.25 - 1.0/w + kv**2 / w**2) * dw_dk**2
                     + 0.5 * d2w_dk2)

            if abs(denom) < 1e-10:
                continue

            sigma2_loc[i, j] = dw_dtau / denom

    return sigma2_loc


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2-A:  Raw FDM Dupire  (RBF interpolation → Dupire)
# ═══════════════════════════════════════════════════════════════════════════

def dupire_raw(k_scatter, tau_scatter, iv_scatter, k_eval, tau_eval):
    """
    1. RBF-interpolate the raw IV scatter to a regular grid (no smoothing).
    2. Compute total variance W = IV^2 * tau.
    3. Apply Dupire FDM.
    """
    # ── RBF interpolation (faithful to raw data → thin_plate_spline) ──
    pts = np.column_stack([k_scatter, tau_scatter])
    rbf = RBFInterpolator(pts, iv_scatter, kernel='thin_plate_spline',
                          smoothing=0.0)

    KK, TT = np.meshgrid(k_eval, tau_eval, indexing='ij')
    targets = np.column_stack([KK.ravel(), TT.ravel()])
    IV_grid = rbf(targets).reshape(KK.shape)
    IV_grid = np.clip(IV_grid, 0.01, None)

    W = IV_grid**2 * tau_eval[np.newaxis, :]
    sigma2 = _dupire_from_total_variance(k_eval, tau_eval, W)
    return sigma2


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2-B:  SVI Parametrisation + Smoothed Dupire
# ═══════════════════════════════════════════════════════════════════════════

def _svi_total_variance(k, params):
    """
    SVI parametrisation of total variance:
        w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sig^2))
    """
    a, b, rho, m_svi, sig = params
    return a + b * (rho * (k - m_svi) + np.sqrt((k - m_svi)**2 + sig**2))


def _fit_svi_slice(k_data, iv_data, tau_val):
    """Fit SVI to one maturity slice.  Returns 5 SVI parameters."""
    w_data = iv_data**2 * tau_val
    a0 = np.mean(w_data)

    def residuals(params):
        return _svi_total_variance(k_data, params) - w_data

    lb = [1e-6,  1e-6, -0.999, -0.5, 1e-4]
    ub = [2.0,   5.0,   0.999,  0.5, 2.0]
    result = least_squares(residuals, [a0, 0.1, -0.5, 0.0, 0.1],
                           bounds=(lb, ub), method='trf', max_nfev=5000)
    return result.x


def dupire_svi(k_scatter, tau_scatter, iv_scatter, k_eval, tau_eval):
    """
    1. Group data by maturity (nearest tau_eval bucket).
    2. Fit SVI per maturity slice.
    3. Evaluate SVI on (k_eval, tau_eval) grid → smooth W surface.
    4. Apply Dupire FDM.
    """
    Nk = len(k_eval)
    Nt = len(tau_eval)

    # Assign each data point to its nearest tau_eval bucket
    tau_data_unique = np.unique(tau_scatter)

    svi_params = {}
    for tau_val in tau_data_unique:
        mask = tau_scatter == tau_val
        if mask.sum() < 5:
            continue
        k_sl  = k_scatter[mask]
        iv_sl = iv_scatter[mask]
        svi_params[tau_val] = _fit_svi_slice(k_sl, iv_sl, tau_val)

    if len(svi_params) < 2:
        raise ValueError("Not enough maturity slices with data for SVI fitting.")

    # Evaluate SVI on the dense grid.
    # For each tau_eval, interpolate SVI params from the two nearest fitted slices.
    fitted_taus = np.array(sorted(svi_params.keys()))
    fitted_params = np.array([svi_params[t] for t in fitted_taus])  # (N_fitted, 5)

    W_smooth = np.zeros((Nk, Nt))
    for j, tau_val in enumerate(tau_eval):
        # Find nearest fitted tau
        idx = np.argmin(np.abs(fitted_taus - tau_val))
        W_smooth[:, j] = _svi_total_variance(k_eval, fitted_params[idx])
        W_smooth[:, j] = np.clip(W_smooth[:, j], 1e-8, None)

    sigma2 = _dupire_from_total_variance(k_eval, tau_eval, W_smooth)
    return sigma2


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3:  PINN Local Volatility
# ═══════════════════════════════════════════════════════════════════════════

def pinn_local_vol(k_eval, tau_eval, model, normalizer, device):
    """Evaluate PINN on the (k, tau) grid.  k = ln(K/S), m = -k."""
    m_eval = -k_eval
    KK, TT = np.meshgrid(m_eval, tau_eval, indexing='ij')

    m_flat   = torch.tensor(KK.ravel(), dtype=torch.float32, device=device).unsqueeze(-1)
    tau_flat = torch.tensor(TT.ravel(), dtype=torch.float32, device=device).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        mn, tn = normalizer.normalize(m_flat, tau_flat)
        _, sigma = model(mn, tn)
    return sigma.cpu().numpy().flatten().reshape(KK.shape)


# ═══════════════════════════════════════════════════════════════════════════
#  PART 4:  Plotting
# ═══════════════════════════════════════════════════════════════════════════

def _sigma_from_sigma2(sigma2):
    """Convert local variance to local vol, marking invalid cells as NaN."""
    sigma = np.full_like(sigma2, np.nan)
    valid = np.isfinite(sigma2) & (sigma2 > 0)
    sigma[valid] = np.sqrt(sigma2[valid])
    return sigma


def plot_three_way(k_eval, tau_eval, sigma_raw, sigma_svi, sigma_pinn,
                   market_data, results_dir='results/phase3'):
    """3-panel heatmap comparison."""
    os.makedirs(results_dir, exist_ok=True)
    moneyness = np.exp(k_eval)   # K/S = exp(k)

    vmin, vmax = 0.05, 0.55

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
    titles = [
        '(A)  Raw FDM Dupire',
        '(B)  Smoothed Dupire (SVI)',
        '(C)  PINN Local Volatility'
    ]

    for idx, (ax, title, surf) in enumerate(zip(axes, titles,
                                                 [sigma_raw, sigma_svi, sigma_pinn])):
        # For panels A/B, use a special colormap that shows NaN as gray
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='#2a2a2a')   # dark gray for NaN / negative var

        im = ax.pcolormesh(tau_eval, moneyness, surf,
                           cmap=cmap, shading='auto',
                           vmin=vmin, vmax=vmax)
        ax.set_xlabel('τ (years to expiry)', fontsize=13)
        if idx == 0:
            ax.set_ylabel('Moneyness K/S', fontsize=13)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Overlay data points
        ax.scatter(market_data['tau'], market_data['moneyness'],
                   s=1, c='white', alpha=0.12)
        ax.set_ylim([moneyness.min(), moneyness.max()])

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('Local Vol σ', fontsize=12)

    fig.subplots_adjust(left=0.05, right=0.91, wspace=0.08)
    out = os.path.join(results_dir, 'benchmark_comparison.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\n  [OK] Saved: {out}")
    plt.close(fig)


def plot_three_way_3d(k_eval, tau_eval, sigma_raw, sigma_svi, sigma_pinn, results_dir='results/phase3'):
    """3-panel 3D surface comparison."""
    os.makedirs(results_dir, exist_ok=True)
    moneyness = np.exp(k_eval)
    
    # Create meshgrid for 3D plotting
    TT, MM = np.meshgrid(tau_eval, moneyness, indexing='ij')

    vmin, vmax = 0.05, 0.55

    fig = plt.figure(figsize=(24, 8))
    titles = [
        '(A)  Raw FDM Dupire',
        '(B)  Smoothed Dupire (SVI)',
        '(C)  PINN Local Volatility'
    ]

    for idx, (title, surf) in enumerate(zip(titles, [sigma_raw, sigma_svi, sigma_pinn])):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        # For panels A/B, use a special colormap that shows NaN as gray
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='#2a2a2a')
        
        # We plot the surface. Note: surf is (Nk, Nt). We need it to match TT, MM which are (Nt, Nk) if indexing='ij'.
        # Actually in plot_three_way we did ax.pcolormesh(tau_eval, moneyness, surf), where surf is (Nk, Nt).
        # Let's create proper meshgrid:
        TT_grid, MM_grid = np.meshgrid(tau_eval, moneyness)
        # surf is (Nk, Nt), so surf matches TT_grid and MM_grid which are (Nk, Nt).
        
        # To avoid ValueError with NaN in 3D surface, we can clip or mask, but plot_surface handles NaN by not rendering the polygon.
        surf_plot = np.clip(surf, vmin, vmax) # Clip for better visual range in Z axis
        
        im = ax.plot_surface(TT_grid, MM_grid, surf_plot, cmap=cmap, vmin=vmin, vmax=vmax,
                             linewidth=0, antialiased=False, alpha=0.9)
        
        ax.set_xlabel('τ (years to expiry)', fontsize=11, labelpad=10)
        ax.set_ylabel('Moneyness K/S', fontsize=11, labelpad=10)
        ax.set_zlabel('Local Vol σ', fontsize=11, labelpad=10)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_zlim(vmin, vmax)
        ax.view_init(elev=25, azim=-125) # Adjust viewing angle

    # Shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.2, 0.015, 0.6])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('Local Vol σ', fontsize=12)

    fig.subplots_adjust(left=0.02, right=0.91, wspace=0.1)
    out = os.path.join(results_dir, 'benchmark_comparison_3d.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  [OK] Saved: {out}")
    plt.close(fig)


def plot_cross_sections(k_eval, tau_eval, sigma_raw, sigma_svi, sigma_pinn,
                        market_data, results_dir='results/phase3'):
    """Cross-section plots at 4 representative maturities."""
    os.makedirs(results_dir, exist_ok=True)
    moneyness = np.exp(k_eval)

    target_taus = [0.03, 0.10, 0.33, 0.74]
    tau_indices = [np.argmin(np.abs(tau_eval - t)) for t in target_taus]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax_idx, j in enumerate(axes.flatten()):
        ax = j
        j_idx = tau_indices[ax_idx]
        tau_val = tau_eval[j_idx]
        dte = int(tau_val * 365)

        # Raw
        y = sigma_raw[:, j_idx]
        ok = np.isfinite(y) & (y > 0)
        ax.plot(moneyness[ok], y[ok], 'o-', color='#d62728', ms=3, lw=1,
                alpha=0.7, label='Raw FDM Dupire')

        # SVI
        y = sigma_svi[:, j_idx]
        ok = np.isfinite(y) & (y > 0)
        ax.plot(moneyness[ok], y[ok], 's-', color='#ff7f0e', ms=3, lw=1.3,
                alpha=0.8, label='SVI Dupire')

        # PINN
        ax.plot(moneyness, sigma_pinn[:, j_idx], '-', color='#1f77b4',
                lw=2.2, label='PINN')

        # Market IV scatter
        mask = np.abs(market_data['tau'] - tau_val) < 0.005
        if mask.sum() > 0:
            ax.scatter(market_data['moneyness'][mask],
                       market_data['sigma_market'][mask],
                       s=25, c='gray', alpha=0.5, zorder=1, label='Market IV')

        ax.set_xlabel('K/S', fontsize=11)
        ax.set_ylabel('σ_local', fontsize=11)
        ax.set_title(f'τ ≈ {tau_val:.3f} yr  (~{dte} DTE)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.0, 0.7])

    fig.suptitle('Local Volatility Cross-Sections by Maturity',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(results_dir, 'benchmark_cross_sections.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  [OK] Saved: {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Dupire Benchmark Comparison")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load market data ──
    csv_path = os.path.join(os.path.dirname(__file__), 'spy_otm_2021-09-20.csv')
    market_data = load_market_data(csv_path, verbose=False)

    # ── Raw scatter data ──
    k_scatter, tau_scatter, iv_scatter = _get_market_scatter(market_data)
    print(f"\n  Market data: {len(k_scatter)} unique (k, tau) points")
    print(f"  k in [{k_scatter.min():.4f}, {k_scatter.max():.4f}]")
    print(f"  tau in [{tau_scatter.min():.4f}, {tau_scatter.max():.4f}]")

    # ── Dense evaluation grid ──
    k_eval  = np.linspace(k_scatter.min(), k_scatter.max(), 120)
    tau_eval = np.linspace(tau_scatter.min(), tau_scatter.max(), 80)

    # ── Benchmark A: Raw FDM Dupire ──
    print("\n  Computing Raw FDM Dupire...")
    sigma2_raw = dupire_raw(k_scatter, tau_scatter, iv_scatter, k_eval, tau_eval)
    n_neg = np.sum(sigma2_raw[np.isfinite(sigma2_raw)] < 0)
    n_fin = np.sum(np.isfinite(sigma2_raw))
    print(f"    Negative variance cells: {n_neg}/{n_fin} "
          f"({100*n_neg/max(n_fin,1):.1f}%)")
    sigma_raw = _sigma_from_sigma2(sigma2_raw)

    # ── Benchmark B: SVI Dupire ──
    print("  Computing SVI Dupire...")
    sigma2_svi = dupire_svi(k_scatter, tau_scatter, iv_scatter, k_eval, tau_eval)
    n_neg_s = np.sum(sigma2_svi[np.isfinite(sigma2_svi)] < 0)
    n_fin_s = np.sum(np.isfinite(sigma2_svi))
    print(f"    Negative variance cells: {n_neg_s}/{n_fin_s} "
          f"({100*n_neg_s/max(n_fin_s,1):.1f}%)")
    sigma_svi = _sigma_from_sigma2(sigma2_svi)

    # ── PINN ──
    print("  Computing PINN local vol...")
    ckpt = torch.load('results/phase3/pinn_phase3.pt', map_location=device)
    domain = ckpt['domain']
    model = PINN_Dual().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    normalizer = LogMoneynessNormalizer(m_scale=0.5, tau_max=domain['tau_max'])
    sigma_pinn = pinn_local_vol(k_eval, tau_eval, model, normalizer, device)

    # ── Summary ──
    print("\n  -- Surface Statistics --")
    for name, surf in [("Raw FDM", sigma_raw), ("SVI Dupire", sigma_svi),
                        ("PINN", sigma_pinn)]:
        v = surf[np.isfinite(surf)]
        pct = 100 * len(v) / surf.size
        if len(v) > 0:
            print(f"    {name:12s}:  valid={pct:5.1f}%  "
                  f"s in [{v.min():.4f}, {v.max():.4f}]  mean={v.mean():.4f}")
        else:
            print(f"    {name:12s}:  valid={pct:5.1f}%  (all NaN)")

    # ── Plots ──
    print("\n  Generating comparison plots...")
    plot_three_way(k_eval, tau_eval, sigma_raw, sigma_svi, sigma_pinn,
                   market_data)
    plot_cross_sections(k_eval, tau_eval, sigma_raw, sigma_svi, sigma_pinn,
                        market_data)
    plot_three_way_3d(k_eval, tau_eval, sigma_raw, sigma_svi, sigma_pinn)
    print("\n  Benchmark complete.\n")


if __name__ == '__main__':
    main()
