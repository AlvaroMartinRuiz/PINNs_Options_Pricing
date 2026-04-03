"""
Input normalization utilities for PINN training.

Phase 1:  Min-max scaling (maps [low, high] -> [0, 1])  (S, t) -> [0, 1]^2 
Phase 2+: Log-moneyness    (S, K) -> m = ln(S/K), tau = T - t
"""

import torch


class MinMaxScaler:
    """
    Linearly maps [lo, hi] -> [0, 1] and back.
    Works with PyTorch tensors (keeps gradients alive).
    """

    def __init__(self, lo: float, hi: float):
        assert hi > lo, f"hi ({hi}) must be > lo ({lo})"
        self.lo = lo
        self.hi = hi

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map x from [lo, hi] to [0, 1]."""
        return (x - self.lo) / (self.hi - self.lo)

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Map x_norm from [0, 1] back to [lo, hi]."""
        return x_norm * (self.hi - self.lo) + self.lo

    def __repr__(self):
        return f"MinMaxScaler(lo={self.lo}, hi={self.hi})"


class Phase1Normalizer:
    """
    Normalizes (S, t) to [0, 1]^2 for Phase 1.

    Parameters
    ----------
    S_min, S_max : float – spot price domain bounds
    t_min, t_max : float – time domain bounds
    """

    def __init__(self, S_min: float = 0.0, S_max: float = 100.0,
                 t_min: float = 0.0, t_max: float = 0.5):
        self.S_scaler = MinMaxScaler(S_min, S_max) # Two independent scalers: one for price S 
        self.t_scaler = MinMaxScaler(t_min, t_max) # one for time t

    def normalize(self, S: torch.Tensor, t: torch.Tensor):
        """Return (S_norm, t_norm) in [0,1]^2."""
        return self.S_scaler.normalize(S), self.t_scaler.normalize(t)

    def denormalize(self, S_norm: torch.Tensor, t_norm: torch.Tensor):
        """Return (S, t) from [0,1]^2 back to original domain."""
        return self.S_scaler.denormalize(S_norm), self.t_scaler.denormalize(t_norm)

    @property
    def S_min(self):
        return self.S_scaler.lo

    @property
    def S_max(self):
        return self.S_scaler.hi

    @property
    def t_min(self):
        return self.t_scaler.lo

    @property
    def t_max(self):
        return self.t_scaler.hi


class LogMoneynessNormalizer:
    """
    Phase 2+ normalizer: works in log-moneyness coordinates.

    Transforms:
        m    = ln(S/K)   -- log-moneyness (centered at 0 = ATM)
        tau  = T - t     -- time to maturity

    Normalization:
        m_norm   = m / m_scale      (simple scaling, keeps m=0 at center)
        tau_norm = tau / tau_max     (scales to [0, 1])

    Parameters
    ----------
    m_scale : float  -- characteristic moneyness scale (e.g. 0.5)
    tau_max : float  -- maximum maturity in the domain
    """

    def __init__(self, m_scale: float = 0.5, tau_max: float = 1.0):
        self.m_scale = m_scale
        self.tau_max = tau_max

    def to_log_moneyness(self, S: torch.Tensor, K) -> torch.Tensor:
        """Compute m = ln(S/K)."""
        return torch.log(S / K)

    def normalize(self, m: torch.Tensor, tau: torch.Tensor):
        """Normalize (m, tau) for network input."""
        return m / self.m_scale, tau / self.tau_max

    def denormalize(self, m_norm: torch.Tensor, tau_norm: torch.Tensor):
        """Recover physical (m, tau) from normalized values."""
        return m_norm * self.m_scale, tau_norm * self.tau_max
