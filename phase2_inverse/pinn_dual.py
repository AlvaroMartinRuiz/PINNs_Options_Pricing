"""
Dual-output Physics-Informed Neural Network for Phase 2 (Inverse Problem).

Architecture:
    Input:  (m_norm, tau_norm)  -- normalized log-moneyness and time-to-maturity
    Shared trunk: 3 hidden layers x 64 neurons, tanh
    Price head:   1 hidden layer x 64 neurons -> 1 output (v_hat, normalized price V/K)
    Vol head:     1 hidden layer x 32 neurons -> 1 output (sigma_hat, local volatility)
                  Uses softplus to guarantee sigma > 0

PDE (in log-moneyness coordinates):
    dv/dtau = (1/2)*sigma^2 * d2v/dm2
              + (r - q - sigma^2/2) * dv/dm
              - r * v

The PINN simultaneously learns:
    1. The normalized price surface v(m, tau)
    2. The local volatility surface sigma(m, tau)
"""

import torch
import torch.nn as nn
import math


class PINN_Dual(nn.Module):
    """
    Dual-output PINN for local volatility calibration.

    Parameters (architecture)
    ----------
    trunk_layers : list[int] -- shared trunk hidden sizes (default: [64, 64, 64])
    price_layers : list[int] -- price head hidden sizes (default: [64])
    vol_layers   : list[int] -- vol head hidden sizes (default: [32])
    sigma_min    : float     -- minimum volatility floor (default: 0.01)
    """

    def __init__(self, trunk_layers=None, price_layers=None, vol_layers=None,
                 sigma_min=0.01):
        super().__init__()

        if trunk_layers is None:
            trunk_layers = [64, 64, 64]
        if price_layers is None:
            price_layers = [64]
        if vol_layers is None:
            vol_layers = [32]

        self.sigma_min = sigma_min

        # --- Shared trunk ---
        trunk = []
        in_dim = 2  # (m_norm, tau_norm)
        for h in trunk_layers:
            trunk.append(nn.Linear(in_dim, h))
            in_dim = h
        self.trunk = nn.ModuleList(trunk)
        trunk_out_dim = trunk_layers[-1]

        # --- Price head ---
        price = []
        in_dim = trunk_out_dim
        for h in price_layers:
            price.append(nn.Linear(in_dim, h))
            in_dim = h
        price.append(nn.Linear(in_dim, 1))
        self.price_head = nn.ModuleList(price)

        # --- Vol head ---
        vol = []
        in_dim = trunk_out_dim
        for h in vol_layers:
            vol.append(nn.Linear(in_dim, h))
            in_dim = h
        vol.append(nn.Linear(in_dim, 1))
        self.vol_head = nn.ModuleList(vol)

        self.act = torch.tanh

        # Xavier initialization: We use it to avoid vanishing/exploding gradients.
        self._init_weights()

    def _init_weights(self):
        for module_list in [self.trunk, self.price_head, self.vol_head]:
            for layer in module_list:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, m_norm, tau_norm):
        """
        Forward pass.

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

        # Shared trunk
        x = torch.cat([m_norm, tau_norm], dim=-1)  # (N, 2)
        for layer in self.trunk:
            x = self.act(layer(x))

        # Price head
        v = x
        for layer in self.price_head[:-1]:
            v = self.act(layer(v))
        v_hat = self.price_head[-1](v)  # linear output

        # Vol head
        s = x
        for layer in self.vol_head[:-1]:
            s = self.act(layer(s))
        sigma_raw = self.vol_head[-1](s)
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

    Parameters
    ----------
    model      : PINN_Dual instance
    m, tau     : physical-coordinate tensors with requires_grad=True
    normalizer : LogMoneynessNormalizer

    Returns
    -------
    smooth_loss : scalar tensor -- mean of (dsigma/dm)^2 + (dsigma/dtau)^2
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


def compute_dupire_consistency_loss(model, m, tau, normalizer, r, q):
    """
    Dupire consistency loss: sigma_hat must equal the vol implied by the
    PINN's own price derivatives.

    From the PDE, solving for sigma^2:
        sigma^2 = 2 * (dv/dtau + r*v) / (d2v/dm2 - dv/dm)
               (ignoring the drift term for the Dupire-like inversion)

    More precisely, from:
        dv/dtau = 0.5*sig^2*(d2v/dm2 - dv/dm) + (r-q)*dv/dm - r*v
    we get:
        sig^2 = 2*(dv/dtau - (r-q)*dv/dm + r*v) / (d2v/dm2 - dv/dm)

    This loss penalizes: ||sigma_hat^2 - sigma_dupire^2||^2

    Only applied where d2v/dm2 - dv/dm is sufficiently large (away from
    degenerate regions where the denominator is near zero).
    """
    m_norm, tau_norm = normalizer.normalize(m, tau)
    v_hat, sigma_hat = model(m_norm, tau_norm)

    # Price derivatives
    v_tau = torch.autograd.grad(v_hat, tau, grad_outputs=torch.ones_like(v_hat),
                                create_graph=True)[0]
    v_m = torch.autograd.grad(v_hat, m, grad_outputs=torch.ones_like(v_hat),
                              create_graph=True)[0]
    v_mm = torch.autograd.grad(v_m, m, grad_outputs=torch.ones_like(v_m),
                               create_graph=True)[0]

    # Dupire-implied sigma^2
    numerator = 2.0 * (v_tau - (r - q) * v_m + r * v_hat)
    denominator = v_mm - v_m

    # Only use points where denominator is safely away from zero
    denom_abs = torch.abs(denominator)
    valid_mask = denom_abs > 0.01  # threshold to avoid division instability

    # Compute sigma^2_dupire only at valid points
    sigma2_dupire = numerator / (denominator + 1e-8)  # small eps for safety
    sigma2_hat = sigma_hat ** 2

    # Compute loss only at valid points
    if valid_mask.any():
        diff = (sigma2_hat[valid_mask] - sigma2_dupire[valid_mask]) ** 2
        return torch.mean(diff)
    else:
        return torch.tensor(0.0, device=m.device)


def terminal_condition_phase2(m):
    """
    Normalized call payoff: v(m, 0) = max(exp(m) - 1, 0).
    """
    return torch.clamp(torch.exp(m) - 1.0, min=0.0)
