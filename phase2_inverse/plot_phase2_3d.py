import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase2_inverse.pinn_dual import PINN_Dual
from phase2_inverse.lv_surface import synthetic_lv_numpy
from utils.normalization import LogMoneynessNormalizer

def plot_phase2_3d():
    results_dir = 'results/phase2'
    model_path = os.path.join(results_dir, 'pinn_dual_phase2.pt')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    fdm_params = checkpoint['fdm_params']
    
    model = PINN_Dual().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    normalizer = LogMoneynessNormalizer(
        m_scale=0.5,
        tau_max=fdm_params['tau_max']
    )
    
    # Evaluation grid
    m_eval = np.linspace(-0.4, 0.4, 100)
    tau_eval = np.linspace(0.05, 0.95, 50)
    m_grid, tau_grid = np.meshgrid(m_eval, tau_eval, indexing='ij')

    m_flat = torch.tensor(m_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(-1)
    tau_flat = torch.tensor(tau_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        m_norm, tau_norm = normalizer.normalize(m_flat, tau_flat)
        _, sigma_pred = model(m_norm, tau_norm)
        sigma_pred_2d = sigma_pred.cpu().numpy().flatten().reshape(m_grid.shape)

    # Ground truth
    sigma_true_2d = synthetic_lv_numpy(m_grid.flatten(), tau_grid.flatten()).reshape(m_grid.shape)

    # --- Plot 3D ---
    fig_3d = plt.figure(figsize=(16, 7))
    
    vmin = min(np.min(sigma_true_2d), np.min(sigma_pred_2d))
    vmax = max(np.max(sigma_true_2d), np.max(sigma_pred_2d))

    # True surface
    ax1 = fig_3d.add_subplot(1, 2, 1, projection='3d')
    im3d_0 = ax1.plot_surface(tau_grid, m_grid, sigma_true_2d, cmap='viridis', 
                              linewidth=0, antialiased=False, alpha=0.9, vmin=vmin, vmax=vmax)
    ax1.set_xlabel('tau (maturity)', fontsize=11, labelpad=10)
    ax1.set_ylabel('m = ln(S/K)', fontsize=11, labelpad=10)
    ax1.set_zlabel('Local Vol sigma', fontsize=11, labelpad=10)
    ax1.set_title('True sigma(m, tau) [3D]', fontsize=14, fontweight='bold', pad=20)
    ax1.view_init(elev=25, azim=-125)
    
    # Recovered surface
    ax2 = fig_3d.add_subplot(1, 2, 2, projection='3d')
    im3d_1 = ax2.plot_surface(tau_grid, m_grid, sigma_pred_2d, cmap='viridis', 
                              linewidth=0, antialiased=False, alpha=0.9, vmin=vmin, vmax=vmax)
    ax2.set_xlabel('tau (maturity)', fontsize=11, labelpad=10)
    ax2.set_ylabel('m = ln(S/K)', fontsize=11, labelpad=10)
    ax2.set_zlabel('Local Vol sigma', fontsize=11, labelpad=10)
    ax2.set_zlim(ax1.get_zlim())
    ax2.set_title('Recovered PINN sigma(m, tau) [3D]', fontsize=14, fontweight='bold', pad=20)
    ax2.view_init(elev=25, azim=-125)
    
    # Shared colorbar
    cbar_ax = fig_3d.add_axes([0.92, 0.2, 0.015, 0.6])
    cb = fig_3d.colorbar(im3d_1, cax=cbar_ax)
    cb.set_label('Local Vol sigma', fontsize=12)

    fig_3d.subplots_adjust(left=0.02, right=0.88, wspace=0.1)
    
    out_path = os.path.join(results_dir, 'vol_surface_comparison_3d.png')
    fig_3d.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"  [OK] Saved: {out_path}")
    plt.close(fig_3d)

if __name__ == '__main__':
    plot_phase2_3d()
