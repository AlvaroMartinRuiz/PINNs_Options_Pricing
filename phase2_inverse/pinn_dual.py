"""
Dual-output Physics-Informed Neural Network for Phase 2 (Inverse Problem).

Architecture (v2 -- separate networks):
    Price network: (m_norm, tau_norm) -> v_hat   [4 x 64 tanh, 1 linear]
    Vol network:   (m_norm, tau_norm) -> sigma_hat [3 x 64 tanh, softplus]

    The two networks are coupled ONLY through the PDE loss, not through
    shared parameters. This prevents the ill-conditioned vol gradients
    from contaminating the price network's learning.

PDE (in log-moneyness coordinates):
    dv/dtau = (1/2)*sigma^2 * d2v/dm2
              + (r - q - sigma^2/2) * dv/dm
              - r * v
"""

import torch
import torch.nn as nn
import math


class PINN_Dual(nn.Module):
    """
    Dual-output PINN with SEPARATE networks for price and volatility.

    Parameters (architecture)
    ----------
    price_layers : list[int] -- price network hidden sizes (default: [64, 64, 64, 64])
    vol_layers   : list[int] -- vol network hidden sizes (default: [64, 64, 64])
    sigma_min    : float     -- minimum volatility floor (default: 0.01)
    """

    def __init__(self, price_layers=None, vol_layers=None, sigma_min=0.01):
        super().__init__()

        if price_layers is None:
            price_layers = [64, 64, 64, 64]
        if vol_layers is None:
            vol_layers = [64, 64, 64]

        self.sigma_min = sigma_min

        # --- Price network (independent) ---
        price_mods = []
        in_dim = 2  # (m_norm, tau_norm)
        for h in price_layers:
            price_mods.append(nn.Linear(in_dim, h))
            in_dim = h
        price_mods.append(nn.Linear(in_dim, 1))  # output layer
        self.price_net = nn.ModuleList(price_mods)

        # --- Vol network (independent) ---
        vol_mods = []
        in_dim = 2  # (m_norm, tau_norm)
        for h in vol_layers:
            vol_mods.append(nn.Linear(in_dim, h))
            in_dim = h
        vol_mods.append(nn.Linear(in_dim, 1))  # output layer
        self.vol_net = nn.ModuleList(vol_mods)

        self.act = torch.tanh

        # Xavier initialization: avoid vanishing/exploding gradients.
        self._init_weights()

    def _init_weights(self):
        for module_list in [self.price_net, self.vol_net]:
            for layer in module_list:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, m_norm, tau_norm):
        """
        Forward pass through both independent networks.

        Parameters
        ----------
        m_norm, tau_norm : tensors of shape (N,) or (N, 1)

        Returns
        -------
        v_hat     : (N, 1) -- predicted normalized price V/K
        sigma_hat : (N, 1) -- predicted local volatility (always > sigma_min)
        """
        if m_norm.dim() == 1:
            m_norm = m_norm.unsqueeze(-1)
        if tau_norm.dim() == 1:
            tau_norm = tau_norm.unsqueeze(-1)

        inp = torch.cat([m_norm, tau_norm], dim=-1)  # (N, 2)

        # Price network
        v = inp
        for layer in self.price_net[:-1]:
            v = self.act(layer(v))
        v_hat = self.price_net[-1](v)  # linear output (price can be any value)

        # Vol network
        s = inp
        for layer in self.vol_net[:-1]:
            s = self.act(layer(s))
        sigma_raw = self.vol_net[-1](s)
        # softplus ensures sigma > 0, then add floor
        sigma_hat = nn.functional.softplus(sigma_raw) + self.sigma_min

        return v_hat, sigma_hat


def compute_pde_residual_phase2(model, m, tau, normalizer, r, q):
    """
    Compute the BS PDE residual in log-moneyness coordinates.

    PDE:
        dv/dtau - (1/2)*sigma^2 * d2v/dm2
                - (r - q - sigma^2/2) * dv/dm
                + r * v = 0

    The residual should be ~0 where the PDE is satisfied.

    Parameters
    ----------
    model      : PINN_Dual instance
    m, tau     : physical-coordinate tensors with requires_grad=True
    normalizer : LogMoneynessNormalizer instance
    r, q       : risk-free rate and dividend yield

    Returns
    -------
    residual : (N, 1) tensor
    """
    # Normalize inputs (keeps computational graph alive for autograd)
    m_norm, tau_norm = normalizer.normalize(m, tau)

    # Forward pass: get both price and vol predictions
    v_hat, sigma_hat = model(m_norm, tau_norm)

    # Compute derivatives of v w.r.t. physical coordinates (m, tau)
    # autograd handles the chain rule through the normalizer automatically
    v_tau = torch.autograd.grad(v_hat, tau, grad_outputs=torch.ones_like(v_hat),
                                create_graph=True)[0]
    v_m = torch.autograd.grad(v_hat, m, grad_outputs=torch.ones_like(v_hat),
                              create_graph=True)[0]
    v_mm = torch.autograd.grad(v_m, m, grad_outputs=torch.ones_like(v_m),
                               create_graph=True)[0]

    # PDE residual
    sigma2 = sigma_hat ** 2
    residual = (v_tau
                - 0.5 * sigma2 * v_mm
                - (r - q - 0.5 * sigma2) * v_m
                + r * v_hat)

    return residual


def compute_smoothness_loss(model, m, tau, normalizer):
    """
    Tikhonov regularization on the volatility surface: ||grad(sigma)||^2.

    This prevents wild oscillations in the recovered volatility.
    """
    m_norm, tau_norm = normalizer.normalize(m, tau)
    _, sigma_hat = model(m_norm, tau_norm)

    sigma_m = torch.autograd.grad(sigma_hat, m,
                                  grad_outputs=torch.ones_like(sigma_hat),
                                  create_graph=True)[0]
    sigma_tau = torch.autograd.grad(sigma_hat, tau,
                                    grad_outputs=torch.ones_like(sigma_hat),
                                    create_graph=True)[0]

    return torch.mean(sigma_m**2 + sigma_tau**2)


def terminal_condition_phase2(m):
    """
    Normalized call payoff: v(m, 0) = max(exp(m) - 1, 0).
    """
    return torch.clamp(torch.exp(m) - 1.0, min=0.0)
