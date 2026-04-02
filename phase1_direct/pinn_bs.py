"""
Physics-Informed Neural Network for the Black-Scholes PDE (Phase 1).

Solves the BS PDE with constant volatility σ, risk-free rate r, and
continuous dividend yield q for a European call option:

    ∂V/∂t + (r - q)·S·∂V/∂S + ½·σ²·S²·∂²V/∂S² − r·V = 0

All computations happen in *physical* (S, t) space; the normalizer maps
inputs to [0, 1]² for the network, and the PDE residual is computed via
chain-rule adjustments to the autograd derivatives.

Architecture
------------
    Input:  (S_norm, t_norm) ∈ [0,1]²
    Hidden: 4 layers × 64 neurons, tanh activation
    Output: V̂ (unnormalized option price)
"""

import torch
import torch.nn as nn
import math


class PINN_BS(nn.Module):
    """
    Feed-forward PINN for Black-Scholes with constant σ.

    Parameters (architecture)
    ----------
    layers       : list[int]  – layer sizes including input & output
                                default: [2, 64, 64, 64, 64, 1]
    activation   : str        – 'tanh' or 'relu' (tanh recommended for PDE)
    """

    def __init__(self, layers=None, activation='tanh'):
        super().__init__()
        if layers is None:
            layers = [2, 64, 64, 64, 64, 1]

        # Build layers
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))

        # Activation
        if activation == 'tanh':
            self.act = torch.tanh
        elif activation == 'relu':
            self.act = torch.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Xavier initialization (important for tanh networks): We use it to avoid
        # vanishing/exploding gradients.
        self._init_weights()

    def _init_weights(self):
        for lin in self.linears:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, S_norm: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        S_norm, t_norm : tensors of shape (N,) or (N, 1), values in [0, 1]

        Returns
        -------
        V : tensor of shape (N, 1) – predicted option price
        """
        # Ensure (N, 1) shape
        if S_norm.dim() == 1:
            S_norm = S_norm.unsqueeze(-1)
        if t_norm.dim() == 1:
            t_norm = t_norm.unsqueeze(-1)

        x = torch.cat([S_norm, t_norm], dim=-1)  # (N, 2) Inputs are S_norm and t_norm

        for lin in self.linears[:-1]: # Hidden layers
            x = self.act(lin(x))
        x = self.linears[-1](x)  # Output layer (no activation)
        return x


def compute_pde_residual(model: PINN_BS, S: torch.Tensor, t: torch.Tensor,
                         normalizer, sigma: float, r: float, q: float):
    """
    Compute the BS PDE residual at collocation points (S, t).

    The PDE (in physical space):
        ∂V/∂t + (r-q)·S·∂V/∂S + ½·σ²·S²·∂²V/∂S² − r·V = 0

    We feed normalized inputs to the network but compute derivatives
    w.r.t. the *physical* variables (not normalized) via autograd (which handles the
    chain rule automatically because S_norm = f(S)).

    Parameters
    ----------
    model      : PINN_BS instance
    S, t       : physical-space tensors with requires_grad=True
    normalizer : Phase1Normalizer instance
    sigma, r, q: BS parameters

    Returns
    -------
    residual : tensor of shape (N, 1) – should be ≈ 0 if PDE is satisfied
    """
    # Normalize (keeps the computational graph alive)
    S_norm, t_norm = normalizer.normalize(S, t)

    # Forward pass
    V = model(S_norm, t_norm)

    # First derivatives via autograd
    V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V),
                              create_graph=True)[0] # create_graph=True allows us to compute higher-order derivatives, that we need for the PDE.
    V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V),
                              create_graph=True)[0]

    # Second derivative
    V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S),
                               create_graph=True)[0]

    # PDE residual
    residual = V_t + (r - q) * S * V_S + 0.5 * sigma**2 * S**2 * V_SS - r * V

    return residual

# Boundary and initial conditions: They are necessary to solve the PDE.
# Otherwise, there would be infinite solutions.
def terminal_condition(S: torch.Tensor, K: float):
    """
    European call payoff at expiry (t=T): max(S - K, 0).
    """
    return torch.clamp(S - K, min=0.0)


def boundary_condition_lower(t: torch.Tensor, K: float, r: float, q: float, T: float):
    """
    V(S=0, t) = 0  for a call option.
    """
    return torch.zeros_like(t)


def boundary_condition_upper(S_max: float, t: torch.Tensor,
                             K: float, r: float, q: float, T: float):
    """
    Deep in-the-money approximation:
    V(S_max, t) ≈ S_max·e^{-q(T-t)} − K·e^{-r(T-t)}  for a deep ITM call.
    """
    tau = T - t  # time to maturity
    return S_max * torch.exp(-q * tau) - K * torch.exp(-r * tau)
