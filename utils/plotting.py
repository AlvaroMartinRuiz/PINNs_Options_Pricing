"""
Plotting utilities for PINN training and validation.

Provides:
    - plot_loss_history:    training loss curves (total + components)
    - plot_surface_3d:     3D surface of V(S, t)
    - plot_error_heatmap:  |V_pinn - V_bs| as a 2D heatmap
    - plot_slice:          1D slice comparisons (V vs S at fixed t)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path


def plot_loss_history(history: dict, save_path: str = None, show: bool = True):
    """
    Plot training loss curves.

    Parameters
    ----------
    history : dict with keys like 'total', 'pde', 'ic', 'bc'
              Each value is a list of (epoch, loss) or just loss values.
    save_path : optional path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {'total': '#E63946', 'pde': '#457B9D', 'ic': '#2A9D8F', 'bc': '#E9C46A'}
    for key, values in history.items():
        c = colors.get(key, None)
        ax.semilogy(values, label=key.upper(), color=c, linewidth=2 if key == 'total' else 1.2)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss (log scale)', fontsize=13)
    ax.set_title('Training Loss History', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_surface_3d(S_grid, t_grid, V_grid, title="Option Price Surface",
                    save_path=None, show=True):
    """
    3D surface plot of V(S, t).

    Parameters
    ----------
    S_grid, t_grid : 2D arrays (meshgrid)
    V_grid         : 2D array of option values
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(S_grid, t_grid, V_grid, cmap=cm.viridis,
                           linewidth=0, antialiased=True, alpha=0.85)
    ax.set_xlabel('Spot Price $S$', fontsize=12)
    ax.set_ylabel('Time $t$', fontsize=12)
    ax.set_zlabel('Option Value $V$', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.colorbar(surf, shrink=0.5, label='$V$')
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_error_heatmap(S_1d, t_1d, error_grid, title="Absolute Error |PINN − BS|",
                       save_path=None, show=True):
    """
    2D heatmap of absolute error.

    Parameters
    ----------
    S_1d, t_1d   : 1D arrays (axes)
    error_grid   : 2D array  (shape = len(S_1d) x len(t_1d))
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    im = ax.pcolormesh(t_1d, S_1d, error_grid, cmap='hot', shading='auto')
    ax.set_xlabel('Time $t$', fontsize=13)
    ax.set_ylabel('Spot Price $S$', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Absolute Error', fontsize=12)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_slices(S_1d, V_pinn_slices, V_bs_slices, t_values,
                title="PINN vs BS at Fixed Times", save_path=None, show=True):
    """
    1D comparison plots: V(S) at several fixed times t.

    Parameters
    ----------
    S_1d           : 1D array of spot prices
    V_pinn_slices  : list of 1D arrays, one per t-value
    V_bs_slices    : list of 1D arrays, one per t-value
    t_values       : list of floats (the fixed t values)
    """
    n = len(t_values)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    colors_pinn = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#6A0572']
    colors_bs = ['#F4A261', '#264653', '#1D6D37', '#B8860B', '#9B59B6']

    for i, (t_val, V_p, V_b) in enumerate(zip(t_values, V_pinn_slices, V_bs_slices)):
        ax = axes[i]
        ax.plot(S_1d, V_b, '--', color=colors_bs[i % len(colors_bs)],
                linewidth=2, label='BS Analytical')
        ax.plot(S_1d, V_p, '-', color=colors_pinn[i % len(colors_pinn)],
                linewidth=1.5, alpha=0.85, label='PINN')
        ax.set_xlabel('$S$', fontsize=13)
        if i == 0:
            ax.set_ylabel('$V(S, t)$', fontsize=13)
        ax.set_title(f'$t = {t_val:.2f}$', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
