"""
Training script for Phase 1: BS PINN with constant sigma.

Strategy:
    1. Adam optimizer (lr=1e-3) for 8,000 epochs  -- global exploration
    2. L-BFGS optimizer for 2,000 iterations       -- fine-tuning

Collocation points:
    - Interior (PDE): 10,000 random points in [0, S_max] x [0, T]
    - Terminal (IC):  500 random points at t = T (initial condition in backward time, because the PDE is solved backwards in time: start at t=T and go to t=0)
    - Boundary (BC):  200 random points (100 at S=0, 100 at S=S_max)

Loss:
    L = lam_pde * L_pde + lam_ic * L_ic + lam_bc * L_bc
"""

import sys
import os
import time
import argparse

import torch
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) # This is done to import the modules from the parent directory.

from phase1_direct.pinn_bs import (
    PINN_BS, compute_pde_residual,
    terminal_condition, boundary_condition_lower, boundary_condition_upper
)
from utils.normalization import Phase1Normalizer
from utils.plotting import plot_loss_history


# ── Default Parameters ───────────────────────────────────────────────────────

PARAMS = {
    'K': 50.0,
    'r': 0.05,
    'q': 0.015,
    'sigma': 0.25,
    'T': 0.5,
    'S_min': 0.0,
    'S_max': 100.0,
}

TRAIN = {
    'n_pde': 10_000,
    'n_ic': 500,
    'n_bc': 200,       # total (split between lower and upper)
    'lambda_pde': 1.0,
    'lambda_ic': 10.0,
    'lambda_bc': 1.0,
    'adam_epochs': 8_000,
    'adam_lr': 1e-3,
    'lbfgs_iters': 2_000,
    'lbfgs_lr': 1.0,
    'print_every': 500,
    'resample_every': 2_000,  # resample collocation points periodically
}


# ── Collocation Point Sampling ───────────────────────────────────────────────
# These points are used to evaluate the PDE and the boundary conditions.
# The points are used as inputs to the neural network.
def sample_interior(n, S_min, S_max, T, device):
    """Random points in the interior: S in (S_min, S_max), t in (0, T)."""
    S = S_min + (S_max - S_min) * torch.rand(n, 1, device=device)
    t = T * torch.rand(n, 1, device=device)
    S.requires_grad_(True)
    t.requires_grad_(True)
    return S, t


def sample_terminal(n, S_min, S_max, T, device):
    """Points at the terminal time t = T."""
    S = S_min + (S_max - S_min) * torch.rand(n, 1, device=device)
    t = T * torch.ones(n, 1, device=device)
    return S, t


def sample_boundary(n, S_min, S_max, T, device):
    """Points at S = S_min and S = S_max (split evenly)."""
    n_half = n // 2
    # Lower boundary: S = S_min
    t_lo = T * torch.rand(n_half, 1, device=device)
    S_lo = S_min * torch.ones(n_half, 1, device=device)
    # Upper boundary: S = S_max
    t_hi = T * torch.rand(n - n_half, 1, device=device)
    S_hi = S_max * torch.ones(n - n_half, 1, device=device)
    return S_lo, t_lo, S_hi, t_hi


# ── Loss Computation ─────────────────────────────────────────────────────────

def compute_loss(model, normalizer, params, train_cfg, device):
    """
    Compute total loss = lam_pde*L_pde + lam_ic*L_ic + lam_bc*L_bc.

    Returns (total_loss, {pde, ic, bc} individual losses as floats).
    """
    K, r, q, sigma, T = params['K'], params['r'], params['q'], params['sigma'], params['T']
    S_min, S_max = params['S_min'], params['S_max']

    # ── PDE residual ──
    S_int, t_int = sample_interior(train_cfg['n_pde'], S_min, S_max, T, device)
    residual = compute_pde_residual(model, S_int, t_int, normalizer, sigma, r, q)
    loss_pde = torch.mean(residual ** 2)

    # ── Terminal condition (IC in backward time) ──
    S_ic, t_ic = sample_terminal(train_cfg['n_ic'], S_min, S_max, T, device)
    S_ic_norm, t_ic_norm = normalizer.normalize(S_ic, t_ic)
    V_pred_ic = model(S_ic_norm, t_ic_norm)
    V_exact_ic = terminal_condition(S_ic, K)
    loss_ic = torch.mean((V_pred_ic - V_exact_ic) ** 2)

    # ── Boundary conditions ──
    S_lo, t_lo, S_hi, t_hi = sample_boundary(train_cfg['n_bc'], S_min, S_max, T, device)

    # Lower boundary: V(0, t) = 0
    S_lo_norm, t_lo_norm = normalizer.normalize(S_lo, t_lo)
    V_pred_lo = model(S_lo_norm, t_lo_norm)
    V_exact_lo = boundary_condition_lower(t_lo, K, r, q, T)
    loss_bc_lo = torch.mean((V_pred_lo - V_exact_lo) ** 2)

    # Upper boundary: V(S_max, t) = S_max*exp(-q*tau) - K*exp(-r*tau)
    S_hi_norm, t_hi_norm = normalizer.normalize(S_hi, t_hi)
    V_pred_hi = model(S_hi_norm, t_hi_norm)
    V_exact_hi = boundary_condition_upper(S_max, t_hi, K, r, q, T)
    loss_bc_hi = torch.mean((V_pred_hi - V_exact_hi) ** 2)

    loss_bc = loss_bc_lo + loss_bc_hi

    # ── Weighted total ──
    total = (train_cfg['lambda_pde'] * loss_pde +
             train_cfg['lambda_ic'] * loss_ic +
             train_cfg['lambda_bc'] * loss_bc)

    return total, {
        'pde': loss_pde.item(),
        'ic': loss_ic.item(),
        'bc': loss_bc.item(),
        'total': total.item(),
    }


# ── Training Loop ────────────────────────────────────────────────────────────

def train(model, normalizer, params, train_cfg, device, results_dir='results/phase1'):
    """
    Full training: Adam phase + L-BFGS phase.

    Returns
    -------
    model   : trained model
    history : dict of loss lists
    """
    os.makedirs(results_dir, exist_ok=True)

    history = {'total': [], 'pde': [], 'ic': [], 'bc': []}

    # ══════════════════════════════════════════════════════════════════════════
    # Phase A: Adam
    # ══════════════════════════════════════════════════════════════════════════
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['adam_lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg['adam_epochs'], eta_min=1e-5
    )

    print("=" * 65)
    print(f"  Phase A: Adam  ({train_cfg['adam_epochs']} epochs, lr={train_cfg['adam_lr']})")
    print("=" * 65)
    t0 = time.time()

    for epoch in range(1, train_cfg['adam_epochs'] + 1):
        optimizer.zero_grad()
        loss, losses = compute_loss(model, normalizer, params, train_cfg, device)
        loss.backward()
        optimizer.step()
        scheduler.step()

        for k in history:
            history[k].append(losses[k])

        if epoch % train_cfg['print_every'] == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:>5d} | Total={losses['total']:.3e} | "
                  f"PDE={losses['pde']:.3e} | IC={losses['ic']:.3e} | "
                  f"BC={losses['bc']:.3e} | lr={lr_now:.2e} | {elapsed:.1f}s")

    adam_time = time.time() - t0
    print(f"\n  Adam finished in {adam_time:.1f}s\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase B: L-BFGS
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print(f"  Phase B: L-BFGS  ({train_cfg['lbfgs_iters']} max iterations)")
    print("=" * 65)

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

    # L-BFGS needs a closure; we use fixed collocation points for stability
    S_int, t_int = sample_interior(train_cfg['n_pde'], params['S_min'],
                                    params['S_max'], params['T'], device)
    S_ic, t_ic = sample_terminal(train_cfg['n_ic'], params['S_min'],
                                  params['S_max'], params['T'], device)
    S_lo, t_lo, S_hi, t_hi = sample_boundary(train_cfg['n_bc'], params['S_min'],
                                              params['S_max'], params['T'], device)

    lbfgs_iter = [0]
    t1 = time.time()

    def closure():
        lbfgs_optimizer.zero_grad()

        # PDE
        residual = compute_pde_residual(model, S_int, t_int, normalizer,
                                        params['sigma'], params['r'], params['q'])
        loss_pde = torch.mean(residual ** 2)

        # Terminal
        S_ic_n, t_ic_n = normalizer.normalize(S_ic, t_ic)
        V_ic = model(S_ic_n, t_ic_n)
        loss_ic = torch.mean((V_ic - terminal_condition(S_ic, params['K'])) ** 2)

        # Boundaries
        S_lo_n, t_lo_n = normalizer.normalize(S_lo, t_lo)
        loss_lo = torch.mean((model(S_lo_n, t_lo_n) -
                              boundary_condition_lower(t_lo, params['K'], params['r'],
                                                      params['q'], params['T'])) ** 2)
        S_hi_n, t_hi_n = normalizer.normalize(S_hi, t_hi)
        loss_hi = torch.mean((model(S_hi_n, t_hi_n) -
                              boundary_condition_upper(params['S_max'], t_hi, params['K'],
                                                      params['r'], params['q'], params['T'])) ** 2)
        loss_bc = loss_lo + loss_hi

        total = (train_cfg['lambda_pde'] * loss_pde +
                 train_cfg['lambda_ic'] * loss_ic +
                 train_cfg['lambda_bc'] * loss_bc)

        total.backward()

        lbfgs_iter[0] += 1
        if lbfgs_iter[0] % 100 == 0 or lbfgs_iter[0] == 1:
            print(f"  L-BFGS iter {lbfgs_iter[0]:>5d} | Total={total.item():.3e} | "
                  f"PDE={loss_pde.item():.3e} | IC={loss_ic.item():.3e} | "
                  f"BC={loss_bc.item():.3e}")

        # Record history
        history['total'].append(total.item())
        history['pde'].append(loss_pde.item())
        history['ic'].append(loss_ic.item())
        history['bc'].append(loss_bc.item())

        return total

    lbfgs_optimizer.step(closure)
    lbfgs_time = time.time() - t1
    print(f"\n  L-BFGS finished in {lbfgs_time:.1f}s  ({lbfgs_iter[0]} iterations)\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    total_time = adam_time + lbfgs_time
    print(f"  Total training time: {total_time:.1f}s")

    # Save model
    model_path = os.path.join(results_dir, 'pinn_bs_phase1.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'params': params,
        'train_cfg': train_cfg,
        'history': history,
    }, model_path)
    print(f"  Model saved: {model_path}")

    # Save loss plot
    plot_loss_history(history,
                      save_path=os.path.join(results_dir, 'loss_history.png'),
                      show=False)

    return model, history


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train Phase 1 BS PINN')
    parser.add_argument('--adam-epochs', type=int, default=TRAIN['adam_epochs'])
    parser.add_argument('--lbfgs-iters', type=int, default=TRAIN['lbfgs_iters'])
    parser.add_argument('--results-dir', type=str, default='results/phase1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Update training config
    train_cfg = TRAIN.copy()
    train_cfg['adam_epochs'] = args.adam_epochs
    train_cfg['lbfgs_iters'] = args.lbfgs_iters

    # Build model
    model = PINN_BS().to(device)
    normalizer = Phase1Normalizer(S_min=PARAMS['S_min'], S_max=PARAMS['S_max'],
                                  t_min=0.0, t_max=PARAMS['T'])

    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"BS params: K={PARAMS['K']}, r={PARAMS['r']}, q={PARAMS['q']}, "
          f"sigma={PARAMS['sigma']}, T={PARAMS['T']}")
    print(f"Domain: S in [{PARAMS['S_min']}, {PARAMS['S_max']}], t in [0, {PARAMS['T']}]")
    print(f"Collocation: PDE={train_cfg['n_pde']}, IC={train_cfg['n_ic']}, "
          f"BC={train_cfg['n_bc']}")
    print()

    model, history = train(model, normalizer, PARAMS, train_cfg, device,
                           results_dir=args.results_dir)

    print("\nTraining complete. Run validate_phase1.py to check accuracy.")


if __name__ == '__main__':
    main()
