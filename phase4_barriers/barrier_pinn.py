"""
Phase 4 -- PINN barrier option pricer.

Trains a new price network to solve the Dupire PDE with barrier boundary
conditions, using the FROZEN volatility surface from Phase 3.

Key design decisions:
    1. Squashed boundary ansatz: v(m, tau) = g(m) * NN(m, tau)
       where g(m) = 1 - exp(-alpha * (m - m_B)) for down-out
       This exactly satisfies v(m_B, tau) = 0 without soft penalties.

    2. No data loss: there is no market data for barrier options.
       The PINN prices purely from PDE + BCs + frozen calibrated vol.

    3. Greeks via autograd: Delta and Gamma are computed analytically
       through the computational graph — no bumping required.

Usage:
    python -m phase4_barriers.barrier_pinn
"""

import torch
import torch.nn as nn
import numpy as np
import os, sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.normalization import LogMoneynessNormalizer


# ── Barrier Price Network ───────────────────────────────────────────────────

class BarrierPriceNet(nn.Module):
    """
    Price network for barrier options with hard boundary enforcement.

    The output is:
        v(m, tau) = g(m) * NN(m_norm, tau_norm)

    where g(m) is a squashed distance function that equals 0 at the barrier
    and approaches 1 in the interior.

    For down-and-out (barrier at m_B, domain m >= m_B):
        g(m) = 1 - exp(-alpha * (m - m_B))

    For up-and-out (barrier at m_B, domain m <= m_B):
        g(m) = 1 - exp(-alpha * (m_B - m))
    """

    def __init__(self, barrier_m, barrier_type='down-out',
                 alpha=10.0, hidden_layers=None):
        super().__init__()

        self.barrier_m = barrier_m
        self.barrier_type = barrier_type
        self.alpha = alpha

        if hidden_layers is None:
            hidden_layers = [64, 64, 64, 64]

        # Build the NN
        layers = []
        in_dim = 2  # (m_norm, tau_norm)
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.layers = nn.ModuleList(layers)
        self.act = torch.tanh

        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _boundary_mask(self, m):
        """Squashed distance function g(m) that is 0 at barrier, ~1 in interior."""
        if self.barrier_type == 'down-out':
            dist = m - self.barrier_m   # positive in the domain
        else:  # up-out
            dist = self.barrier_m - m   # positive in the domain

        return 1.0 - torch.exp(-self.alpha * torch.clamp(dist, min=0.0))

    def forward(self, m_norm, tau_norm, m_phys):
        """
        Parameters
        ----------
        m_norm, tau_norm : normalized inputs for the NN
        m_phys           : physical (un-normalized) moneyness for the mask

        Returns
        -------
        v : (N, 1) normalized barrier option price
        """
        if m_norm.dim() == 1:
            m_norm = m_norm.unsqueeze(-1)
        if tau_norm.dim() == 1:
            tau_norm = tau_norm.unsqueeze(-1)

        inp = torch.cat([m_norm, tau_norm], dim=-1)

        x = inp
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        nn_out = self.layers[-1](x)   # raw NN output

        # Apply squashed boundary mask
        g = self._boundary_mask(m_phys).unsqueeze(-1) if m_phys.dim() == 1 else self._boundary_mask(m_phys)
        v = g * nn_out

        return v


# ── Training Engine ─────────────────────────────────────────────────────────

def train_barrier_pinn(barrier_m, barrier_type, sigma_func_frozen,
                       r, q, m_spot, m_domain, tau_max,
                       normalizer, device,
                       n_epochs=5000, lr=1e-3,
                       n_pde=2000, n_ic=500,
                       lambda_pde=1.0, lambda_ic=10.0):
    """
    Train the barrier PINN.

    Loss = lambda_pde * L_PDE + lambda_ic * L_IC

    No data loss (no market data for barrier options).
    """
    model = BarrierPriceNet(
        barrier_m=barrier_m,
        barrier_type=barrier_type,
        alpha=10.0
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    m_lo, m_hi = m_domain
    history = {'total': [], 'pde': [], 'ic': []}

    print(f"\n  Training Barrier PINN ({barrier_type}, m_B={barrier_m:.4f})")
    print(f"  Domain: m in [{m_lo:.3f}, {m_hi:.3f}], tau in [0, {tau_max:.3f}]")
    print(f"  Epochs: {n_epochs}, PDE pts: {n_pde}, IC pts: {n_ic}")
    t0 = time.time()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # ── PDE collocation points (random) ──
        m_pde = torch.rand(n_pde, device=device) * (m_hi - m_lo) + m_lo
        tau_pde = torch.rand(n_pde, device=device) * tau_max + 1e-4
        m_pde.requires_grad_(True)
        tau_pde.requires_grad_(True)

        m_n, tau_n = normalizer.normalize(m_pde, tau_pde)
        v = model(m_n, tau_n, m_pde)

        # Get frozen sigma at collocation points
        # COORDINATE FIX: The PINN vol surface expects m_pinn = ln(S0/K_dupire).
        # The FDM/PINN spatial grid uses m = ln(S/K). By Dupire, K_dupire = S,
        # so m_pinn = ln(S0/S) = ln(S0/K) - ln(S/K) = m_spot - m_grid.
        tau_vals_np = tau_pde.detach().cpu().numpy()
        m_vals_np = m_pde.detach().cpu().numpy()
        unique_taus = np.unique(np.round(tau_vals_np, 4))
        sig_all = np.zeros(n_pde)
        for tau_val in unique_taus:
            mask = np.abs(tau_vals_np - tau_val) < 5e-4
            if mask.any():
                m_query = m_spot - m_vals_np[mask]
                sig_all[mask] = sigma_func_frozen(m_query, float(tau_val))

        # For points not covered (unlikely), use mean tau
        remaining = sig_all == 0
        if remaining.any():
            m_query_rem = m_spot - m_vals_np[remaining]
            sig_all[remaining] = sigma_func_frozen(m_query_rem,
                                                    float(tau_vals_np.mean()))

        sigma = torch.tensor(sig_all, dtype=torch.float32, device=device).unsqueeze(-1)

        # PDE derivatives
        v_tau = torch.autograd.grad(v, tau_pde, torch.ones_like(v),
                                    create_graph=True)[0]
        v_m = torch.autograd.grad(v, m_pde, torch.ones_like(v),
                                  create_graph=True)[0]
        v_mm = torch.autograd.grad(v_m, m_pde, torch.ones_like(v_m),
                                   create_graph=True)[0]

        if v_tau.dim() == 1:
            v_tau = v_tau.unsqueeze(-1)
        if v_m.dim() == 1:
            v_m = v_m.unsqueeze(-1)
        if v_mm.dim() == 1:
            v_mm = v_mm.unsqueeze(-1)

        sigma2 = sigma ** 2
        pde_res = (v_tau
                   - 0.5 * sigma2 * v_mm
                   - (r - q - 0.5 * sigma2) * v_m
                   + r * v)
        L_pde = torch.mean(pde_res ** 2)

        # ── Terminal condition (tau = 0): v(m, 0) = max(exp(m) - 1, 0) ──
        m_ic = torch.rand(n_ic, device=device) * (m_hi - m_lo) + m_lo
        tau_ic = torch.full((n_ic,), 1e-5, device=device)
        m_ic_n, tau_ic_n = normalizer.normalize(m_ic, tau_ic)

        v_ic = model(m_ic_n, tau_ic_n, m_ic)
        v_ic_target = torch.clamp(torch.exp(m_ic) - 1.0, min=0.0).unsqueeze(-1)
        L_ic = torch.mean((v_ic - v_ic_target) ** 2)

        # ── Total loss ──
        loss = lambda_pde * L_pde + lambda_ic * L_ic
        loss.backward()
        optimizer.step()
        scheduler.step()

        history['total'].append(loss.item())
        history['pde'].append(L_pde.item())
        history['ic'].append(L_ic.item())

        if (epoch + 1) % 1000 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:5d} | Loss={loss.item():.6f} | "
                  f"PDE={L_pde.item():.6f} | IC={L_ic.item():.6f} | "
                  f"{elapsed:.0f}s")

    return model, history


# ── Price extraction ────────────────────────────────────────────────────────

def price_barrier_pinn(model, normalizer, S, K, B, T, barrier_type='down-out'):
    """
    Extract barrier option price from a trained PINN.

    Returns
    -------
    price : float (un-normalized dollar price)
    """
    device = next(model.parameters()).device
    m_spot = np.log(S / K)
    m_t = torch.tensor([m_spot], dtype=torch.float32, device=device)
    tau_t = torch.tensor([T], dtype=torch.float32, device=device)

    m_n, tau_n = normalizer.normalize(m_t, tau_t)

    with torch.no_grad():
        v = model(m_n, tau_n, m_t)

    return float(v.item()) * K


def compute_greeks(model, normalizer, S, K, T):
    """
    Compute Delta and Gamma of the barrier option via autograd.

    Delta = dV/dS = (1/K) * dv/dm * dm/dS = (1/K) * v_m * (1/S)
          = v_m / (K * S) * K = v_m / S

    Gamma = d2V/dS2

    Returns delta, gamma in the original (S, t) coordinates.
    """
    device = next(model.parameters()).device
    m_phys = torch.tensor([np.log(S / K)], dtype=torch.float32,
                          device=device, requires_grad=True)
    tau_t = torch.tensor([T], dtype=torch.float32, device=device)

    m_n, tau_n = normalizer.normalize(m_phys, tau_t)
    v = model(m_n, tau_n, m_phys)

    # dv/dm
    v_m = torch.autograd.grad(v, m_phys, torch.ones_like(v),
                              create_graph=True)[0]
    # d2v/dm2
    v_mm = torch.autograd.grad(v_m, m_phys, torch.ones_like(v_m),
                               create_graph=True)[0]

    # Convert to S-space:
    # V = K * v(m, tau),  m = ln(S/K),  dm/dS = 1/S
    # Delta = dV/dS = K * dv/dm * dm/dS = K * v_m / S = v_m * (K/S)
    # Gamma = d2V/dS2 = K * [v_mm * (1/S)^2 - v_m * (1/S^2)]
    #       = (K/S^2) * (v_mm - v_m)

    delta = float(v_m.item()) * K / S
    gamma = float((v_mm - v_m).item()) * K / (S * S)

    return delta, gamma


# ── Self-test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from phase4_barriers.barrier_fdm import sigma_constant, price_barrier_fdm
    from phase4_barriers.bs_barrier_closedform import down_and_out_call, bs_call

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    S, K, T, r, q, sigma_val = 100.0, 100.0, 1.0, 0.05, 0.015, 0.20
    B = 90.0
    barrier_m = np.log(B / K)
    barrier_type = 'down-out'

    sf = sigma_constant(sigma_val)
    normalizer = LogMoneynessNormalizer(m_scale=0.5, tau_max=T)

    # Train
    m_spot = np.log(S / K)
    model, history = train_barrier_pinn(
        barrier_m=barrier_m,
        barrier_type=barrier_type,
        sigma_func_frozen=sf,
        r=r, q=q,
        m_spot=m_spot,
        m_domain=(barrier_m, 1.0),
        tau_max=T,
        normalizer=normalizer,
        device=device,
        n_epochs=5000,
        lr=1e-3,
        n_pde=2000,
        n_ic=500,
    )

    # Compare
    pinn_price = price_barrier_pinn(model, normalizer, S, K, B, T, barrier_type)
    fdm_price = price_barrier_fdm(S, K, B, T, r, q, sf, barrier_type)
    bs_price = down_and_out_call(S, K, B, T, r, q, sigma_val)

    print(f"\n  --- Results for Down-and-Out Call (B={B}) ---")
    print(f"  BS Closed-Form: {bs_price:.4f}")
    print(f"  FDM:            {fdm_price:.4f}")
    print(f"  PINN:           {pinn_price:.4f}")

    # Greeks
    delta, gamma = compute_greeks(model, normalizer, S, K, T)
    print(f"\n  PINN Greeks:")
    print(f"    Delta = {delta:.6f}")
    print(f"    Gamma = {gamma:.6f}")
