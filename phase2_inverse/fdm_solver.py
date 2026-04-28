"""
Crank-Nicolson Finite Difference solver for the BS PDE in log-moneyness coords.

Solves the PDE for the normalized price v = V/K as a function of (m, tau):

    dv/dtau = (1/2) * sigma^2 * d2v/dm2
              + (r - q - sigma^2/2) * dv/dm
              - r * v

where:
    m   = ln(S/K)   -- log-moneyness
    tau = T - t      -- time to maturity (forward marching direction)
    sigma = sigma_loc(m, tau) -- local volatility

Initial condition (tau = 0):
    v(m, 0) = max(exp(m) - 1, 0)   -- normalized call payoff

Boundary conditions:
    v(m_min, tau) = 0                             -- deep OTM call
    v(m_max, tau) = exp(m_max)*exp(-q*tau) - exp(-r*tau)  -- deep ITM call

The solver marches forward in tau from 0 to tau_max using Crank-Nicolson.
"""

import numpy as np
from scipy.linalg import solve_banded


def crank_nicolson_lv(sigma_func, r=0.05, q=0.015,
                      m_min=-1.5, m_max=1.5, N_m=400,
                      tau_max=1.0, N_tau=400):
    """
    Solve the BS PDE with local volatility using Crank-Nicolson.

    Parameters
    ----------
    sigma_func : callable(m_array, tau_scalar) -> sigma_array
                 Local volatility function. Takes a 1D array of m values
                 and a scalar tau, returns a 1D array of volatilities.
    r          : float -- risk-free rate
    q          : float -- continuous dividend yield
    m_min, m_max : float -- log-moneyness domain bounds
    N_m        : int -- number of spatial grid points (interior + boundary)
    tau_max    : float -- maximum time-to-maturity
    N_tau      : int -- number of time steps

    Returns
    -------
    m_grid   : 1D array of shape (N_m,) -- spatial grid
    tau_grid : 1D array of shape (N_tau+1,) -- temporal grid (including tau=0)
    V        : 2D array of shape (N_m, N_tau+1) -- normalized price v(m, tau)
    """
    # Grids
    dm = (m_max - m_min) / (N_m - 1) # m = ln(S/K)
    dtau = tau_max / N_tau # tau = T - t
    m_grid = np.linspace(m_min, m_max, N_m)
    tau_grid = np.linspace(0, tau_max, N_tau + 1)

    # Solution array: V[i, n] = v(m_i, tau_n)
    V = np.zeros((N_m, N_tau + 1)) # V is the normalized price v(m, tau). We are getting the V for each m and tau (3D surface)

    # Initial condition (tau = 0): v(m, 0) = max(exp(m) - 1, 0). Condition exactly at t=T. 
    # For all m at t=T. (payoff of a call option at maturity)
    # This is not a boundary condition, but the initial condition for the PDE solver.
    V[:, 0] = np.maximum(np.exp(m_grid) - 1.0, 0.0) 
    # This is the call price at time t=T (maturity). At maturity => V(S,T)=max(S−K,0). 
    # We are using normalized price v = V/K => v(m,0) = max(S/K - 1, 0) = max(exp(m)-1, 0)

    # Interior indices (exclude boundaries i=0 and i=N_m-1)
    N_int = N_m - 2  # number of interior points

    # Time-stepping: march forward in tau. Starting at t=T (tau=0) and going backwards to t=0 (tau=T).
    for n in range(N_tau):
        tau_n = tau_grid[n] # current time level (known)
        tau_np1 = tau_grid[n + 1] # next time level (unknown)

        # Evaluate local volatility at both time levels (for Crank-Nicolson averaging) 
        sigma_n = sigma_func(m_grid[1:-1], tau_n)      # interior points at tau_n
        sigma_np1 = sigma_func(m_grid[1:-1], tau_np1)  # interior points at tau_{n+1}

        # --- Build coefficients at tau_n (for explicit/RHS part) ---
        a_n = 0.5 * sigma_n**2 / dm**2                          # diffusion
        b_n = (r - q - 0.5 * sigma_n**2) / (2.0 * dm)          # drift

        # Sub, main, super diagonal coefficients for L_n
        sub_n = a_n - b_n           # coefficient of v_{i-1}
        main_n = -2.0 * a_n - r    # coefficient of v_i
        sup_n = a_n + b_n           # coefficient of v_{i+1}

        # --- Build coefficients at tau_{n+1} (for implicit/LHS part) ---
        a_np1 = 0.5 * sigma_np1**2 / dm**2
        b_np1 = (r - q - 0.5 * sigma_np1**2) / (2.0 * dm)

        sub_np1 = a_np1 - b_np1
        main_np1 = -2.0 * a_np1 - r
        sup_np1 = a_np1 + b_np1

        # --- RHS: (I + dtau/2 * L_n) * v^n ---
        v_int = V[1:-1, n]  # interior values at current time

        rhs = np.zeros(N_int)
        for i in range(N_int):
            rhs[i] = v_int[i] + 0.5 * dtau * main_n[i] * v_int[i] # Known terms at time n
            if i > 0:
                rhs[i] += 0.5 * dtau * sub_n[i] * v_int[i - 1] 
            else:
                # v_{i-1} = V[0, n] (left boundary)
                rhs[i] += 0.5 * dtau * sub_n[i] * V[0, n] # This is because v_int is for interior points, so we need to add the boundary points manually
            if i < N_int - 1:
                rhs[i] += 0.5 * dtau * sup_n[i] * v_int[i + 1]
            else:
                # v_{i+1} = V[N_m-1, n] (right boundary)
                rhs[i] += 0.5 * dtau * sup_n[i] * V[-1, n] # This is because v_int is for interior points, so we need to add the boundary points manually

        # --- Apply boundary conditions at tau_{n+1} ---
        V[0, n + 1] = 0.0  # deep OTM (As S->0, V=0)
        V[-1, n + 1] = np.exp(m_max) * np.exp(-q * tau_np1) - np.exp(-r * tau_np1)  # deep ITM (As S->inf, V=S*exp(-qT)-K*exp(-rT))

        # Add boundary contributions to RHS from the implicit side
        # (I - dtau/2 * L_{n+1}) v^{n+1} = rhs
        # Move known boundary terms to RHS
        rhs[0] -= (-0.5 * dtau * sub_np1[0]) * V[0, n + 1]
        rhs[-1] -= (-0.5 * dtau * sup_np1[-1]) * V[-1, n + 1]

        # --- LHS: (I - dtau/2 * L_{n+1}) ---
        # Build tridiagonal system in banded form for solve_banded
        # Band storage: ab[0,:] = super-diagonal, ab[1,:] = diagonal, ab[2,:] = sub-diagonal
        ab = np.zeros((3, N_int))
        ab[1, :] = 1.0 - 0.5 * dtau * main_np1      # diagonal
        ab[0, 1:] = -0.5 * dtau * sup_np1[:-1]       # super-diagonal (shifted)
        ab[2, :-1] = -0.5 * dtau * sub_np1[1:]        # sub-diagonal (shifted)

        # Solve the tridiagonal system
        V[1:-1, n + 1] = solve_banded((1, 1), ab, rhs)

    return m_grid, tau_grid, V


def extract_prices_at_observations(m_grid, tau_grid, V_surface,
                                   m_obs, tau_obs):
    """
    Interpolate FDM prices at arbitrary observation points.

    Uses 2D bilinear interpolation from the FDM grid to the observation points.

    Parameters
    ----------
    m_grid, tau_grid : 1D arrays from FDM solver
    V_surface        : 2D array from FDM solver (N_m x N_tau+1)
    m_obs, tau_obs   : 1D arrays of observation coordinates

    Returns
    -------
    v_obs : 1D array of interpolated normalized prices
    """
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (m_grid, tau_grid), V_surface,
        method='linear', bounds_error=False, fill_value=None
    )
    points = np.column_stack([m_obs, tau_obs])
    return interp(points)
