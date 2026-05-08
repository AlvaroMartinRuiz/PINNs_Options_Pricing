"""
Training script for Phase 3: Real Market Data Calibration.

Goal: Recover the local volatility surface sigma(m, tau) from real SPY
option mid-prices using the same dual-output PINN architecture as Phase 2.

Pipeline:
    1. Load preprocessed market data (spy_otm_2021-09-20.csv)
    2. Convert to log-moneyness coordinates with put-call parity
    3. Train PINN with: vega-weighted data loss + PDE loss + smoothness + arb + IC
    4. Validate: compare recovered sigma vs market implied vol

Loss:
    L = lam_data * L_data_vega + lam_pde * L_pde + lam_smooth * L_smooth
      + lam_ic * L_ic + lam_arb * L_arb
"""

import sys
import os
import time
import argparse

import torch
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase2_inverse.pinn_dual import (
    PINN_Dual, compute_pde_residual_phase2,
    compute_smoothness_loss, terminal_condition_phase2
)
from phase2_inverse.loss_balancer import LossBalancer
from utils.normalization import LogMoneynessNormalizer
from utils.plotting import plot_loss_history
from phase3_market.data_loader import load_market_data


# ── Default Parameters ───────────────────────────────────────────────────────

DOMAIN = {
    'm_min': -0.25,     # slightly beyond ln(1/1.20) ≈ -0.223
    'm_max':  0.22,     # slightly beyond ln(1/0.80) ≈  0.182
    'tau_max': 1.0,     # max tau in data ≈ 0.989
}

TRAIN = {
    # Collocation points
    'n_pde': 10_000,
    'n_ic': 300,
    # Loss weights
    'lambda_data': 10.0,
    'lambda_pde': 5.0,
    'lambda_smooth': 0.001,
    'lambda_ic': 3.0,
    'lambda_arb': 1.0,       # Lower than Phase 2 — real data may have mild arb violations
    # Anisotropic smoothness weights
    'smooth_weight_m': 0.01,
    'smooth_weight_tau': 1.0,
    # Loss Balancer
    'use_loss_balancer': False,
    'ratio_data': 5.0,
    'ratio_pde': 3.0,
    'ratio_smooth': 0.001,
    'ratio_ic': 1.0,
    # Adam phase
    'adam_epochs': 15_000,
    'adam_lr': 1e-3,
    # L-BFGS phase
    'lbfgs_iters': 5_000,
    'lbfgs_lr': 0.1,
    'print_every': 500,
}


# ── Collocation Sampling ─────────────────────────────────────────────────────

def sample_pde_points(n, m_min, m_max, tau_max, device):
    """Sample random interior points for PDE residual."""
    m = m_min + (m_max - m_min) * torch.rand(n, 1, device=device)
    tau = tau_max * torch.rand(n, 1, device=device)
    tau = torch.clamp(tau, min=0.01)  # stay away from payoff kink
    m.requires_grad_(True)
    tau.requires_grad_(True)
    return m, tau


def sample_ic_points(n, m_min, m_max, device):
    """Sample points at tau = 0 for terminal/initial condition."""
    m = m_min + (m_max - m_min) * torch.rand(n, 1, device=device)
    tau = torch.zeros(n, 1, device=device)
    return m, tau


# ── Loss Computation ─────────────────────────────────────────────────────────

def compute_loss(model, normalizer, data_tensors, domain, train_cfg,
                 device, r_pde, q_pde, fixed_pde=None, fixed_ic=None):
    """
    Compute total loss for Phase 3 market calibration.

    data_tensors = (m_obs, tau_obs, v_obs, vega_w)
    r_pde, q_pde = scalar values for PDE collocation (median r, constant q)
    """
    m_min, m_max = domain['m_min'], domain['m_max']
    tau_max = domain['tau_max']

    m_obs, tau_obs, v_obs, vega_w = data_tensors

    # --- Data loss: vega-weighted MSE on normalized call prices ---
    m_obs_norm, tau_obs_norm = normalizer.normalize(m_obs, tau_obs)
    v_pred, _ = model(m_obs_norm, tau_obs_norm)
    loss_data = torch.mean(vega_w * (v_pred - v_obs) ** 2)

    # --- PDE loss: residual at random interior points ---
    if fixed_pde is not None:
        m_pde, tau_pde = fixed_pde
    else:
        m_pde, tau_pde = sample_pde_points(train_cfg['n_pde'], m_min, m_max,
                                            tau_max, device)
    residual, v_tau, v_m, v_mm, v_hat_pde = compute_pde_residual_phase2(
        model, m_pde, tau_pde, normalizer, r_pde, q_pde
    )
    loss_pde = torch.mean(residual ** 2)

    # --- Arbitrage-Free Constraints ---
    # Dupire calendar: v_tau + r*v - (r-q)*v_m >= 0
    calendar_violation = -(v_tau + r_pde * v_hat_pde - (r_pde - q_pde) * v_m)
    calendar_penalty = torch.mean(torch.relu(calendar_violation)**3)
    # Butterfly: v_mm - v_m >= 0
    butterfly_penalty = torch.mean(torch.relu(-(v_mm - v_m))**3)
    loss_arb = calendar_penalty + butterfly_penalty

    # --- Smoothness loss: Tikhonov on sigma ---
    loss_smooth = compute_smoothness_loss(
        model, m_pde, tau_pde, normalizer,
        weight_m=train_cfg.get('smooth_weight_m', 0.01),
        weight_tau=train_cfg.get('smooth_weight_tau', 1.0)
    )

    # --- IC loss: terminal condition at tau = 0 ---
    if fixed_ic is not None:
        m_ic, tau_ic = fixed_ic
    else:
        m_ic, tau_ic = sample_ic_points(train_cfg['n_ic'], m_min, m_max, device)
    m_ic_norm, tau_ic_norm = normalizer.normalize(m_ic, tau_ic)
    v_ic_pred, _ = model(m_ic_norm, tau_ic_norm)
    v_ic_exact = terminal_condition_phase2(m_ic)
    loss_ic = torch.mean((v_ic_pred - v_ic_exact) ** 2)

    return {
        'data': loss_data,
        'pde': loss_pde,
        'smooth': loss_smooth,
        'ic': loss_ic,
        'arb': loss_arb,
    }


# ── Volatility Pre-training ─────────────────────────────────────────────────

def pretrain_vol_network(model, normalizer, domain, device, epochs=500, target_sigma=0.2):
    """Pre-train vol network to a flat surface (sigma=0.2)."""
    print("=" * 70)
    print(f"  Phase 0: Pre-training Volatility Network (target sigma={target_sigma})")
    print("=" * 70)

    optimizer = torch.optim.Adam(model.vol_net.parameters(), lr=1e-3)
    m_min, m_max = domain['m_min'], domain['m_max']
    tau_max = domain['tau_max']

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        m = (m_max - m_min) * torch.rand(1000, 1, device=device) + m_min
        tau = tau_max * torch.rand(1000, 1, device=device)
        m_norm, tau_norm = normalizer.normalize(m, tau)
        _, sigma_hat = model(m_norm, tau_norm)
        loss = torch.mean((sigma_hat - target_sigma) ** 2)
        loss.backward()
        optimizer.step()
        if (ep + 1) % 100 == 0:
            print(f"  Pre-train Epoch {ep+1}/{epochs} | MSE Loss: {loss.item():.2e}")
    print("  Pre-training complete.\n")


# ── Training Loop ────────────────────────────────────────────────────────────

def train(model, normalizer, market_data, domain, train_cfg,
          device, results_dir='results/phase3'):
    """Full training: Adam + L-BFGS."""
    os.makedirs(results_dir, exist_ok=True)

    # Prepare data tensors
    m_obs_t = torch.tensor(market_data['m'], dtype=torch.float32, device=device).unsqueeze(-1)
    tau_obs_t = torch.tensor(market_data['tau'], dtype=torch.float32, device=device).unsqueeze(-1)
    v_obs_t = torch.tensor(market_data['v_call'], dtype=torch.float32, device=device).unsqueeze(-1)
    vega_w_t = torch.tensor(market_data['vega_weights'], dtype=torch.float32, device=device).unsqueeze(-1)
    data_tensors = (m_obs_t, tau_obs_t, v_obs_t, vega_w_t)

    # Representative r, q for PDE collocation
    r_pde = float(np.median(market_data['r']))
    q_pde = float(market_data['q'][0])
    print(f"  PDE collocation r = {r_pde*100:.4f}%, q = {q_pde*100:.2f}%")

    history = {'total': [], 'data': [], 'pde': [], 'smooth': [], 'ic': [], 'arb': []}
    weight_history = {'data': [], 'pde': [], 'smooth': [], 'ic': [], 'arb': []}

    # Initialize Loss Balancer
    init_weights = {k: train_cfg[f'lambda_{k}'] for k in ['data', 'pde', 'smooth', 'ic', 'arb']}
    target_ratios = {
        'data': train_cfg.get('ratio_data', 1.0),
        'pde': train_cfg.get('ratio_pde', 1.0),
        'smooth': train_cfg.get('ratio_smooth', 1.0),
        'ic': train_cfg.get('ratio_ic', 1.0),
        'arb': train_cfg.get('ratio_arb', 1.0),
    }
    balancer = LossBalancer(model.parameters(), init_weights,
                            target_ratios=target_ratios, alpha=0.9, update_freq=100)

    # Pre-train vol network
    pretrain_vol_network(model, normalizer, domain, device)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase A: Adam
    # ══════════════════════════════════════════════════════════════════════════
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['adam_lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg['adam_epochs'], eta_min=1e-5
    )

    print("=" * 70)
    print(f"  Phase A: Adam ({train_cfg['adam_epochs']} epochs, lr={train_cfg['adam_lr']})")
    print("=" * 70)
    t0 = time.time()

    for epoch in range(1, train_cfg['adam_epochs'] + 1):
        optimizer.zero_grad()
        losses = compute_loss(model, normalizer, data_tensors, domain,
                              train_cfg, device, r_pde, q_pde)

        if train_cfg.get('use_loss_balancer', False):
            grad_norms = balancer.compute_per_component_grad_norms(losses)
            balancer.update(grad_norms)
        weights = balancer.get_weights()

        loss = sum(weights[k] * losses[k] for k in weights)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_item = loss.item()
        history['total'].append(total_item)
        for k in ['data', 'pde', 'smooth', 'ic', 'arb']:
            history[k].append(losses[k].item())
            weight_history[k].append(weights[k])

        if epoch % train_cfg['print_every'] == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:>5d} | Total={total_item:.3e} | "
                  f"Data={losses['data'].item():.3e} | PDE={losses['pde'].item():.3e} | "
                  f"Smooth={losses['smooth'].item():.3e} | IC={losses['ic'].item():.3e} | "
                  f"Arb={losses['arb'].item():.3e} | lr={lr_now:.2e} | {elapsed:.1f}s")

    adam_time = time.time() - t0
    print(f"\n  Adam finished in {adam_time:.1f}s\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase B: L-BFGS
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print(f"  Phase B: L-BFGS ({train_cfg['lbfgs_iters']} max iterations)")
    print("=" * 70)

    # Freeze collocation points for deterministic L-BFGS
    fixed_pde = sample_pde_points(train_cfg['n_pde'],
                                  domain['m_min'], domain['m_max'],
                                  domain['tau_max'], device)
    fixed_ic = sample_ic_points(train_cfg['n_ic'],
                                domain['m_min'], domain['m_max'], device)

    lbfgs_optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=train_cfg['lbfgs_lr'],
        max_iter=train_cfg['lbfgs_iters'],
        max_eval=int(train_cfg['lbfgs_iters'] * 1.25),
        history_size=50,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn='strong_wolfe'
    )

    lbfgs_iter = [0]
    t1 = time.time()

    def closure():
        lbfgs_optimizer.zero_grad()
        losses = compute_loss(model, normalizer, data_tensors, domain,
                              train_cfg, device, r_pde, q_pde,
                              fixed_pde=fixed_pde, fixed_ic=fixed_ic)
        weights = balancer.get_weights()
        loss = sum(weights[k] * losses[k] for k in weights)
        loss.backward()

        lbfgs_iter[0] += 1
        total_item = loss.item()

        if lbfgs_iter[0] % 100 == 0 or lbfgs_iter[0] == 1:
            print(f"  L-BFGS iter {lbfgs_iter[0]:>5d} | Total={total_item:.3e} | "
                  f"Data={losses['data'].item():.3e} | PDE={losses['pde'].item():.3e} | "
                  f"Smooth={losses['smooth'].item():.3e} | Arb={losses['arb'].item():.3e}")

        history['total'].append(total_item)
        for k in ['data', 'pde', 'smooth', 'ic', 'arb']:
            history[k].append(losses[k].item())
            weight_history[k].append(weights[k])

        return loss

    lbfgs_optimizer.step(closure)
    lbfgs_time = time.time() - t1
    print(f"\n  L-BFGS finished in {lbfgs_time:.1f}s ({lbfgs_iter[0]} iterations)\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    total_time = adam_time + lbfgs_time
    print(f"  Total training time: {total_time:.1f}s")

    model_path = os.path.join(results_dir, 'pinn_phase3.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'domain': domain,
        'train_cfg': train_cfg,
        'history': history,
        'weight_history': weight_history,
        'r_pde': r_pde,
        'q_pde': q_pde,
    }, model_path)
    print(f"  Model saved: {model_path}")

    plot_loss_history(history,
                      save_path=os.path.join(results_dir, 'loss_history.png'),
                      show=False)

    return model, history


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train Phase 3 Market PINN')
    parser.add_argument('--adam-epochs', type=int, default=TRAIN['adam_epochs'])
    parser.add_argument('--lbfgs-iters', type=int, default=TRAIN['lbfgs_iters'])
    parser.add_argument('--results-dir', type=str, default='results/phase3')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--csv', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'spy_otm_2021-09-20.csv'))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_cfg = TRAIN.copy()
    train_cfg['adam_epochs'] = args.adam_epochs
    train_cfg['lbfgs_iters'] = args.lbfgs_iters

    # 1. Load market data
    market_data = load_market_data(args.csv)

    # 2. Build model (same architecture as Phase 2)
    model = PINN_Dual().to(device)
    normalizer = LogMoneynessNormalizer(m_scale=0.5, tau_max=DOMAIN['tau_max'])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params} parameters")
    print(f"Domain: m in [{DOMAIN['m_min']}, {DOMAIN['m_max']}], "
          f"tau in [0, {DOMAIN['tau_max']}]")
    print()

    # 3. Train
    model, history = train(model, normalizer, market_data, DOMAIN,
                           train_cfg, device, results_dir=args.results_dir)

    # 4. Validate
    print()
    from phase3_market.validate_phase3 import validate
    validate(model, normalizer, market_data, DOMAIN,
             results_dir=args.results_dir)

    print("\nPhase 3 complete.")


if __name__ == '__main__':
    main()
