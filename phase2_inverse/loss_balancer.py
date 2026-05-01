import torch
import numpy as np


class LossBalancer:
    """
    Dynamically balances the weights of different loss components in a PINN.

    Strategy (gradient-norm balancing):
        After each backward pass, the L2-norm of the accumulated gradients is
        computed for every loss component.  All non-anchor weights are then
        updated (via EMA) so that each component's weighted gradient norm
        matches the anchor's weighted gradient norm.

    Key safeguards against the weight-explosion pathology seen in v1:
        - Per-step clip: a single update may not move a weight by more than
          `step_clip` (x/÷) relative to its current value.
        - Global cap: weights are hard-clamped to [weight_min, weight_max].
        - EMA smoothing: sudden changes are dampened by `alpha`.
    """

    def __init__(self, model_params, init_weights, target_ratios=None,
                 alpha=0.9, update_freq=100,
                 step_clip=3.0, weight_max=1000.0, weight_min=0.01):
        """
        Parameters
        ----------
        model_params : iterable
            All model parameters (used to read .grad after backward).
        init_weights : dict[str, float]
            Initial scalar weight for each loss component.
        target_ratios : dict[str, float]
            Desired ratio of gradient magnitudes for each component.
            If None, defaults to 1:1 ratio for all components.
        alpha : float
            EMA decay factor (0 = no memory, 1 = never update).
        update_freq : int
            How often (in steps) to recompute weights.
        step_clip : float
            Maximum multiplicative change per update (e.g. 3.0 means cap at x3 or /3).
        weight_max : float
            Hard upper bound on any weight.
        weight_min : float
            Hard lower bound on any weight.
        """
        self.params = list(model_params)
        self.weights = {k: float(v) for k, v in init_weights.items()}
        self.anchor = 'data'   # this weight stays fixed; others are moved to match it
        
        if target_ratios is None:
            self.target_ratios = {k: 1.0 for k in init_weights.keys()}
        else:
            self.target_ratios = target_ratios
            
        self.alpha = alpha
        self.update_freq = update_freq
        self.step_clip = step_clip
        self.weight_max = weight_max
        self.weight_min = weight_min
        self.step_count = 0

    def _grad_norm_from_params(self):
        """
        Read the L2 norm of currently accumulated .grad tensors.
        Must be called AFTER loss.backward() and BEFORE optimizer.zero_grad().
        """
        norm_sq = 0.0
        for p in self.params:
            if p.grad is not None:
                norm_sq += p.grad.data.norm(2).item() ** 2
        return norm_sq ** 0.5

    def compute_per_component_grad_norms(self, losses_tensors):
        """
        Compute per-component gradient norms by doing individual backward
        passes on each unweighted loss tensor.

        IMPORTANT: Call this after optimizer.zero_grad() but before the
        main backward pass.

        Parameters
        ----------
        losses_tensors : dict[str, Tensor]
            Unweighted individual loss tensors (requires_grad=True).

        Returns
        -------
        grad_norms : dict[str, float]
        """
        grad_norms = {}
        for name, loss_t in losses_tensors.items():
            # Temporarily zero existing grads
            for p in self.params:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            loss_t.backward(retain_graph=True)
            grad_norms[name] = self._grad_norm_from_params()
        # Zero again so the caller's main backward starts clean
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        return grad_norms

    def update(self, grad_norms: dict):
        """
        Update weights given a pre-computed dict of {component_name: grad_norm}.

        Parameters
        ----------
        grad_norms : dict[str, float]
        """
        self.step_count += 1
        if self.step_count % self.update_freq != 0:
            return self.weights

        anchor_norm = grad_norms.get(self.anchor, 0.0)
        if anchor_norm <= 1e-8:
            return self.weights

        anchor_ratio = self.target_ratios.get(self.anchor, 1.0)
        
        # Base target constant C = (lambda_anchor * G_anchor) / R_anchor
        base_target = (self.weights[self.anchor] * anchor_norm) / anchor_ratio

        for name in self.weights:
            if name == self.anchor:
                continue
            g = grad_norms.get(name, 0.0)
            if g <= 1e-8:
                continue

            # Ideal weight to achieve target ratio
            r_i = self.target_ratios.get(name, 1.0)
            target_norm = base_target * r_i
            hat_w = target_norm / g

            # Per-step clip (relative) to avoid sudden jumps
            hat_w = float(np.clip(hat_w,
                                  self.weights[name] / self.step_clip,
                                  self.weights[name] * self.step_clip))

            # EMA update for smooth transitions
            new_w = self.alpha * self.weights[name] + (1.0 - self.alpha) * hat_w

            # Hard global clamp — prevents the compounding explosion seen in v1
            new_w = float(np.clip(new_w, self.weight_min, self.weight_max))
            self.weights[name] = new_w

        return self.weights

    def get_weights(self):
        return self.weights
