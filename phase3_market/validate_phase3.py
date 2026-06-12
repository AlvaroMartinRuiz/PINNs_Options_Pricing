"""
Phase 3 — Validation and diagnostic plots.

Generates:
    1. Price fit scatter (v_pred vs v_market)
    2. Price residuals vs moneyness
    3. Recovered local volatility surface (heatmap)
    4. Local vol vs market IV at data points
    5. IV smiles at selected DTE buckets
    6. ATM term structure
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase2_inverse.pinn_dual import PINN_Dual, compute_pde_residual_phase2
from utils.normalization import LogMoneynessNormalizer
from phase3_market.data_loader import load_market_data


def validate(model, normalizer, market_data, domain, results_dir='results/phase3'):
    """
    Run full validation suite for Phase 3 calibration.
    """
    os.makedirs(results_dir, exist_ok=True)
    device = next(model.parameters()).device

    m_data = market_data['m']
    tau_data = market_data['tau']
    v_market = market_data['v_call']
    sigma_market = market_data['sigma_market']
    moneyness = market_data['moneyness']
    option_type = market_data['option_type']
    DTE = market_data['DTE']

    # ── Evaluate model at data points ────────────────────────────────────────
    m_t = torch.tensor(m_data, dtype=torch.float32, device=device).unsqueeze(-1)
    tau_t = torch.tensor(tau_data, dtype=torch.float32, device=device).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        m_n, tau_n = normalizer.normalize(m_t, tau_t)
        v_pred_t, sigma_pred_t = model(m_n, tau_n)
        v_pred = v_pred_t.cpu().numpy().flatten()
        sigma_pred = sigma_pred_t.cpu().numpy().flatten()

    # ── Metrics ──────────────────────────────────────────────────────────────
    price_rmse = np.sqrt(np.mean((v_pred - v_market) ** 2))
    price_mae = np.mean(np.abs(v_pred - v_market))
    price_r2 = 1.0 - np.sum((v_pred - v_market)**2) / np.sum((v_market - v_market.mean())**2)

    sigma_rmse = np.sqrt(np.mean((sigma_pred - sigma_market) ** 2))
    sigma_mae = np.mean(np.abs(sigma_pred - sigma_market))

    print("=" * 70)
    print("  Phase 3 Validation: Market Data Calibration")
    print("=" * 70)
    print(f"\n  Price Fit (normalized v = C/K):")
    print(f"    RMSE(v):  {price_rmse:.6f}")
    print(f"    MAE(v):   {price_mae:.6f}")
    print(f"    R2:       {price_r2:.6f}")
    print(f"\n  Vol Comparison (local vol vs market IV):")
    print(f"    RMSE(sigma):  {sigma_rmse:.4f}")
    print(f"    MAE(sigma):   {sigma_mae:.4f}")
    print(f"    Note: Local vol != implied vol conceptually,")
    print(f"          but they should be correlated.\n")

    # ── Plot 1: Price Fit Scatter ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    call_mask = option_type == 'call'
    put_mask = option_type == 'put'

    ax = axes[0]
    ax.scatter(v_market[put_mask], v_pred[put_mask], s=8, alpha=0.4,
               c='#E63946', label='Puts (parity-converted)')
    ax.scatter(v_market[call_mask], v_pred[call_mask], s=8, alpha=0.4,
               c='#457B9D', label='Calls')
    lim = [0, max(v_market.max(), v_pred.max()) * 1.05]
    ax.plot(lim, lim, 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Market v = C/K', fontsize=12)
    ax.set_ylabel('PINN v_pred', fontsize=12)
    ax.set_title(f'Price Fit (RMSE={price_rmse:.5f}, R²={price_r2:.4f})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right panel: residuals vs moneyness
    ax = axes[1]
    residuals = v_pred - v_market
    scatter = ax.scatter(moneyness, residuals, s=8, alpha=0.4,
                         c=np.log10(DTE), cmap='viridis')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Moneyness K/S', fontsize=12)
    ax.set_ylabel('Residual (v_pred - v_market)', fontsize=12)
    ax.set_title('Price Residuals', fontsize=13, fontweight='bold')
    cb = plt.colorbar(scatter, ax=ax)
    cb.set_label('log₁₀(DTE)', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'price_fit.png'), dpi=150)
    print(f"  Saved: price_fit.png")
    plt.close(fig)

    # ── Plot 2: Recovered LV Surface ─────────────────────────────────────────
    m_eval = np.linspace(domain['m_min'], domain['m_max'], 100)
    tau_eval = np.linspace(0.02, domain['tau_max'], 80)
    m_grid, tau_grid = np.meshgrid(m_eval, tau_eval, indexing='ij')

    m_flat = torch.tensor(m_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(-1)
    tau_flat = torch.tensor(tau_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(-1)

    with torch.no_grad():
        mn, tn = normalizer.normalize(m_flat, tau_flat)
        _, sigma_surface = model(mn, tn)
        sigma_surface = sigma_surface.cpu().numpy().flatten().reshape(m_grid.shape)

    # Convert m to moneyness for readability
    moneyness_eval = np.exp(-m_eval)  # K/S = exp(-m)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(tau_eval, moneyness_eval, sigma_surface,
                       cmap='viridis', shading='auto')
    ax.set_xlabel('τ (years to expiry)', fontsize=13)
    ax.set_ylabel('Moneyness K/S', fontsize=13)
    ax.set_title('Recovered Local Volatility σ(K/S, τ)', fontsize=14, fontweight='bold')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Local Vol σ', fontsize=12)

    # Overlay data points
    ax.scatter(tau_data, market_data['moneyness'], s=2, c='white', alpha=0.15,
               label='Data points')
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'lv_surface.png'), dpi=150)
    print(f"  Saved: lv_surface.png")
    plt.close(fig)

    # ── Plot 3: Local Vol vs Market IV ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sigma_market[put_mask], sigma_pred[put_mask], s=8, alpha=0.4,
               c='#E63946', label='Puts')
    ax.scatter(sigma_market[call_mask], sigma_pred[call_mask], s=8, alpha=0.4,
               c='#457B9D', label='Calls')
    lim = [min(sigma_market.min(), sigma_pred.min()) * 0.9,
           max(sigma_market.max(), sigma_pred.max()) * 1.05]
    ax.plot(lim, lim, 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Market Implied Vol σ_market', fontsize=12)
    ax.set_ylabel('PINN Local Vol σ_hat', fontsize=12)
    ax.set_title(f'Local Vol vs Market IV (RMSE={sigma_rmse:.4f})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'vol_comparison.png'), dpi=150)
    print(f"  Saved: vol_comparison.png")
    plt.close(fig)

    # ── Plot 4: IV Smiles at Selected Tenors ──────────────────────────────────
    dte_buckets = [(7, 14, '7-14d'), (30, 60, '30-60d'),
                   (90, 180, '90-180d'), (181, 365, '181-365d')]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for i, (dte_lo, dte_hi, label) in enumerate(dte_buckets):
        ax = axes_flat[i]
        mask = (DTE >= dte_lo) & (DTE <= dte_hi)
        if mask.sum() == 0:
            ax.set_title(f'{label}: No data', fontsize=12)
            continue

        ax.scatter(moneyness[mask], sigma_market[mask], s=15, alpha=0.5,
                   c='#457B9D', label='Market IV', zorder=2)
        ax.scatter(moneyness[mask], sigma_pred[mask], s=15, alpha=0.5,
                   c='#E63946', marker='x', label='PINN Local Vol', zorder=3)
        ax.set_xlabel('Moneyness K/S', fontsize=11)
        ax.set_ylabel('Volatility', fontsize=11)
        ax.set_title(f'Smile: {label} ({mask.sum()} options)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Volatility Smiles by Tenor Bucket', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'iv_smiles.png'), dpi=150)
    print(f"  Saved: iv_smiles.png")
    plt.close(fig)

    # ── Plot 5: ATM Term Structure ────────────────────────────────────────────
    tau_line = np.linspace(0.02, domain['tau_max'], 200)
    m_atm = np.zeros_like(tau_line)
    m_atm_t = torch.tensor(m_atm, dtype=torch.float32, device=device).unsqueeze(-1)
    tau_line_t = torch.tensor(tau_line, dtype=torch.float32, device=device).unsqueeze(-1)

    with torch.no_grad():
        mn, tn = normalizer.normalize(m_atm_t, tau_line_t)
        _, sigma_atm = model(mn, tn)
        sigma_atm = sigma_atm.cpu().numpy().flatten()

    # Also get market ATM data points (moneyness near 1.0)
    atm_mask = (moneyness >= 0.98) & (moneyness <= 1.02)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tau_line, sigma_atm, 'r-', linewidth=2, label='PINN Local Vol (ATM)')
    if atm_mask.sum() > 0:
        ax.scatter(tau_data[atm_mask], sigma_market[atm_mask], s=20, alpha=0.6,
                   c='#457B9D', label='Market IV (K/S ∈ [0.98, 1.02])', zorder=3)
    ax.set_xlabel('τ (years)', fontsize=13)
    ax.set_ylabel('Volatility', fontsize=13)
    ax.set_title('ATM Volatility Term Structure', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'atm_term_structure.png'), dpi=150)
    print(f"  Saved: atm_term_structure.png")
    plt.close(fig)

    # ── Plot 6: Payoff Check (tau=0) ──────────────────────────────────────────
    m_ic = np.linspace(domain['m_min'], domain['m_max'], 200)
    m_ic_t = torch.tensor(m_ic, dtype=torch.float32, device=device).unsqueeze(-1)
    tau_ic_t = torch.zeros_like(m_ic_t)

    with torch.no_grad():
        mn, tn = normalizer.normalize(m_ic_t, tau_ic_t)
        v_ic_pred, _ = model(mn, tn)
        v_ic_pred = v_ic_pred.cpu().numpy().flatten()

    v_ic_true = np.maximum(np.exp(m_ic) - 1.0, 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(m_ic, v_ic_true, 'b--', linewidth=2, label='True Payoff')
    ax.plot(m_ic, v_ic_pred, 'r-', linewidth=1.5, label='PINN Predicted')
    ax.set_xlabel('m = ln(S/K)', fontsize=13)
    ax.set_ylabel('Normalized Price', fontsize=13)
    ax.set_title('Payoff Condition Check (τ=0)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'payoff_check.png'), dpi=150)
    print(f"  Saved: payoff_check.png")
    plt.close(fig)

    print(f"\n  All validation plots saved to {results_dir}/")

    return {
        'price_rmse': price_rmse, 'price_mae': price_mae, 'price_r2': price_r2,
        'sigma_rmse': sigma_rmse, 'sigma_mae': sigma_mae,
    }


# ── Standalone ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validate Phase 3 PINN')
    parser.add_argument('--model', type=str, default='results/phase3/pinn_phase3.pt')
    parser.add_argument('--csv', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'spy_otm_2021-09-20.csv'))
    parser.add_argument('--results-dir', type=str, default='results/phase3')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    checkpoint = torch.load(args.model, map_location=device)
    domain = checkpoint['domain']

    model = PINN_Dual().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    normalizer = LogMoneynessNormalizer(m_scale=0.5, tau_max=domain['tau_max'])
    market_data = load_market_data(args.csv)

    validate(model, normalizer, market_data, domain, results_dir=args.results_dir)
