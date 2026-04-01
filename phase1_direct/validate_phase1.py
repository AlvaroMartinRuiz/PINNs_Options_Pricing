"""
Validation script for Phase 1: compare PINN predictions vs BS analytical.

Metrics:
    - Max absolute error
    - RMSE
    - Mean absolute error (MAE)
    - Relative error for ITM options

Pass criteria:
    - Max error < 0.05
    - RMSE < 0.005

Outputs:
    - Error heatmap (S × t)
    - 3D surface comparison (PINN vs BS)
    - 1D slice plots at several times
    - Summary statistics printed to console
"""

import sys
import os
import argparse

import torch
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase1_direct.pinn_bs import PINN_BS
from utils.black_scholes import bs_call
from utils.normalization import Phase1Normalizer
from utils.plotting import plot_error_heatmap, plot_surface_3d, plot_slices


def evaluate_on_grid(model, normalizer, params, n_S=200, n_t=100, device='cpu'):
    """
    Evaluate the PINN and BS formula on a regular grid.

    Returns
    -------
    S_1d, t_1d : 1D arrays
    V_pinn     : 2D array (n_S × n_t) – PINN predictions
    V_bs       : 2D array (n_S × n_t) – BS analytical
    """
    K, r, q, sigma, T = params['K'], params['r'], params['q'], params['sigma'], params['T']
    S_min, S_max = params['S_min'], params['S_max']

    # Avoid S=0 exactly (log issues in BS formula, though our implementation handles it)
    S_1d = np.linspace(S_min + 0.5, S_max, n_S)
    t_1d = np.linspace(0.0, T - 1e-6, n_t)  # avoid exact t=T (tau=0)

    S_grid, t_grid = np.meshgrid(S_1d, t_1d, indexing='ij')  # (n_S, n_t)

    # BS analytical
    tau_grid = T - t_grid
    V_bs = bs_call(S_grid, K, r, q, sigma, tau_grid)

    # PINN predictions
    S_flat = torch.tensor(S_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(-1)
    t_flat = torch.tensor(t_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        S_norm, t_norm = normalizer.normalize(S_flat, t_flat)
        V_pinn_flat = model(S_norm, t_norm).cpu().numpy().flatten()

    V_pinn = V_pinn_flat.reshape(n_S, n_t)

    return S_1d, t_1d, S_grid, t_grid, V_pinn, V_bs


def compute_metrics(V_pinn, V_bs, S_grid, K):
    """Compute error metrics."""
    abs_error = np.abs(V_pinn - V_bs)
    # Avoid division by zero for deep OTM (V_bs ≈ 0)
    itm_mask = S_grid > K * 0.8  # roughly ITM + near-ATM

    metrics = {
        'max_abs_error': np.max(abs_error),
        'rmse': np.sqrt(np.mean(abs_error**2)),
        'mae': np.mean(abs_error),
        'max_error_itm': np.max(abs_error[itm_mask]) if itm_mask.any() else 0.0,
        'mean_rel_error_itm': np.mean(
            abs_error[itm_mask] / np.maximum(np.abs(V_bs[itm_mask]), 1e-8)
        ) if itm_mask.any() else 0.0,
    }
    return metrics, abs_error


def validate(model_path: str, results_dir: str = 'results/phase1',
             n_S: int = 200, n_t: int = 100, show: bool = False):
    """
    Full validation pipeline.
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    params = checkpoint['params']

    model = PINN_BS()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    normalizer = Phase1Normalizer(S_min=params['S_min'], S_max=params['S_max'],
                                  t_min=0.0, t_max=params['T'])

    print("=" * 65)
    print("  Phase 1 Validation: PINN vs BS Analytical")
    print("=" * 65)
    print(f"  Model: {model_path}")
    print(f"  BS params: K={params['K']}, r={params['r']}, q={params['q']}, "
          f"sigma={params['sigma']}, T={params['T']}")
    print(f"  Validation grid: {n_S} x {n_t} = {n_S * n_t} points")
    print()

    # Evaluate
    S_1d, t_1d, S_grid, t_grid, V_pinn, V_bs = evaluate_on_grid(
        model, normalizer, params, n_S, n_t
    )
    metrics, abs_error = compute_metrics(V_pinn, V_bs, S_grid, params['K'])

    # Print results
    print("  Error Metrics:")
    print(f"    Max absolute error   : {metrics['max_abs_error']:.6f}")
    print(f"    RMSE                 : {metrics['rmse']:.6f}")
    print(f"    MAE                  : {metrics['mae']:.6f}")
    print(f"    Max error (ITM)      : {metrics['max_error_itm']:.6f}")
    print(f"    Mean rel error (ITM) : {metrics['mean_rel_error_itm']:.4%}")
    print()

    # Pass / Fail
    pass_max = metrics['max_abs_error'] < 0.05
    pass_rmse = metrics['rmse'] < 0.005
    status = "PASS" if (pass_max and pass_rmse) else "FAIL"
    print(f"  Max error < 0.05?  {'YES' if pass_max else 'NO'} ({metrics['max_abs_error']:.6f})")
    print(f"  RMSE < 0.005?      {'YES' if pass_rmse else 'NO'} ({metrics['rmse']:.6f})")
    print(f"\n  Overall: {status}")
    print()

    # ── Plots ────────────────────────────────────────────────────────────────

    os.makedirs(results_dir, exist_ok=True)

    # 1. Error heatmap
    plot_error_heatmap(
        S_1d, t_1d, abs_error,
        title=f"Absolute Error |PINN − BS|  (max={metrics['max_abs_error']:.4f})",
        save_path=os.path.join(results_dir, 'error_heatmap.png'),
        show=show
    )

    # 2. 3D surfaces
    plot_surface_3d(
        S_grid, t_grid, V_pinn,
        title="PINN Price Surface $\\hat{V}(S, t)$",
        save_path=os.path.join(results_dir, 'surface_pinn.png'),
        show=show
    )
    plot_surface_3d(
        S_grid, t_grid, V_bs,
        title="BS Analytical Surface $V(S, t)$",
        save_path=os.path.join(results_dir, 'surface_bs.png'),
        show=show
    )

    # 3. 1D slice comparisons
    T = params['T']
    slice_times = [0.0, T * 0.25, T * 0.5, T * 0.75]
    slice_indices = [np.argmin(np.abs(t_1d - tv)) for tv in slice_times]

    V_pinn_slices = [V_pinn[:, idx] for idx in slice_indices]
    V_bs_slices = [V_bs[:, idx] for idx in slice_indices]
    actual_times = [t_1d[idx] for idx in slice_indices]

    plot_slices(
        S_1d, V_pinn_slices, V_bs_slices, actual_times,
        title="PINN vs BS at Fixed Times",
        save_path=os.path.join(results_dir, 'slices_comparison.png'),
        show=show
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Validate Phase 1 BS PINN')
    parser.add_argument('--model', type=str, default='results/phase1/pinn_bs_phase1.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--results-dir', type=str, default='results/phase1')
    parser.add_argument('--grid-S', type=int, default=200, help='Grid points in S')
    parser.add_argument('--grid-t', type=int, default=100, help='Grid points in t')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    args = parser.parse_args()

    metrics = validate(args.model, args.results_dir, args.grid_S, args.grid_t, args.show)


if __name__ == '__main__':
    main()
