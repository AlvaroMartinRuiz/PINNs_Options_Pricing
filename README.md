# Local Volatility Surface Calibration and Barrier Option Pricing via Physics-Informed Deep Learning

Bachelor thesis code. Physics-Informed Neural Networks (PINNs) applied to quantitative
finance: recovering local volatility surfaces from market option prices by solving the
inverse Dupire equation, then pricing barrier options under the calibrated surface.

**Full thesis:** [`TFG_Physics_Informed_Neural_Networks.pdf`](TFG_Physics_Informed_Neural_Networks.pdf) (78 pp.)

Álvaro Martín Ruiz · Supervisor: Emilio Parrado Hernández · Universidad Carlos III de Madrid, 2026
Dual Bachelor in Data Science and Engineering / Telecommunication Technologies Engineering

> The LaTeX sources of the thesis are not kept in this repository — only the compiled PDF.

---

## The problem

The Dupire equation recovers a local volatility surface `σ(K, T)` from observed option
prices, but inverting it numerically is ill-posed: the second derivative in strike sits in
the denominator, so sparse or noisy market quotes produce negative variance and undefined
cells. Classical remedies (finite-difference inversion, SVI smoothing) do not fix this.

A PINN takes a different route. Two networks — one for the price surface, one for
volatility — are coupled only through the PDE residual, so the volatility estimate is
constrained by the physics of the pricing equation rather than by direct differentiation of
noisy data. Arbitrage conditions and smoothness enter as additional loss terms.

## Results

| | |
|---|---|
| Price reconstruction on 1,956 SPY options | `R² = 0.9991` |
| Recovered volatility range | `σ ∈ [0.08, 0.36]` |
| Invalid cells — **PINN** | **0 %** |
| Invalid cells — raw FDM Dupire inversion | 8.5 % (negative variance, max `σ = 8.36`) |
| Invalid cells — SVI-smoothed Dupire | 65.7 % (undefined, calendar-arbitrage violations) |

Being the only method tested that produces a fully valid, arbitrage-free surface is the
central result. On the pricing side, the FDM and Monte Carlo engines agree with each other
within one Monte Carlo standard error on all six barrier configurations, confirming the
surface is usable downstream.

The barrier PINN itself reproduces those benchmarks **within 2.2 % for down-and-out calls**,
but **does not price up-and-out calls reliably** — deviations run from −3.5 % to +256.5 %,
and repeated runs of identical code give materially different answers. A constant-volatility
control experiment locates the cause: the up-and-out networks converge to a comparable PDE
residual while pricing far worse, and their loss stays an order of magnitude above the
down-and-out ones even with a trivial constant `σ`. This is reported as an open limitation
in the thesis, not papered over.

Once trained, inference is effectively free: 1,000 barrier prices in a single batched
forward pass (19.7 µs per option) against 352 ms per option for a Crank–Nicolson solve.

## Phases

The four phases build on one another; each validates something the next one assumes.

**Phase 1 — Forward problem.** Single-output PINN solving Black–Scholes under constant
volatility. Input `(S, t)`, 4×64 tanh layers. Exists to confirm the architecture is sound
against a known closed form (`RMSE = 0.0023`); the only residual error sits at the payoff
kink, where a smooth approximator cannot follow.

**Phase 2 — Inverse problem, synthetic data.** The dual-output network appears here, trained
against prices generated from a known parametric surface so accuracy is measurable: ~87.5 %
of grid cells within 5 % relative error, 99.5 % within 10 %.

**Phase 3 — Inverse problem, real market data.** 1,956 OTM SPY options from 20 September
2021. Adds vega-weighted data loss, calendar-spread and butterfly arbitrage penalties, and
anisotropic smoothness (loose across the smile, tight along the term structure).

**Phase 4 — Barrier options.** The Phase 3 surface is frozen and used as the PDE coefficient.
Down-and-out and up-and-out calls priced three ways — Crank–Nicolson FDM, Monte Carlo with
the Broadie–Glasserman–Kou continuity correction, and a barrier-specific PINN using a hard
ansatz `g(m) = 1 − exp(−α(m_B − m))` to enforce the knock-out condition exactly.

## Running it

```bash
pip install -r requirements.txt
cd TFG_code   # run everything as modules from here
```

| Phase | Train | Validate |
|---|---|---|
| 1 | `python -m phase1_direct.train_phase1` | `python -m phase1_direct.validate_phase1` |
| 2 | `python -m phase2_inverse.train_phase2` | `python -m phase2_inverse.validate_phase2` |
| 3 | `python -m phase3_market.train_phase3` | `python -m phase3_market.validate_phase3` |
| 4 | — (uses the Phase 3 surface) | `python -m phase4_barriers.validate_phase4` |

The classical baselines the PINN is compared against:
`python -m phase3_market.benchmark_dupire`

Each engine also self-tests against the Reiner–Rubinstein closed form under constant
volatility when run directly, e.g. `python -m phase4_barriers.barrier_fdm`.

### Training cost

Measured on the reference hardware (Intel Core i3-1115G4, **CPU only**):

| Phase | Time |
|---|---|
| 1 — Black–Scholes PINN | ~45 min (8,000 Adam + 2,000 L-BFGS) |
| 2 — Synthetic inverse | ~106 min (15,000 Adam + 5,000 L-BFGS) |
| 3 — SPY calibration | ~165 min (pre-training + 15,000 Adam + 5,000 L-BFGS) |
| 4 — Barrier PINN | 43–85 min **per barrier level** (5,000 Adam) |

Phase 4's `validate_phase4.py` trains one network per barrier configuration, so the full
price comparison takes several hours. Pass `train_pinn=False` to `run_price_comparison` to
get the FDM and Monte Carlo columns in seconds without it.

## Layout

```
phase1_direct/     Forward Black-Scholes PINN
phase2_inverse/    Dual-output PINN + FDM data generator + parametric LV surface
phase3_market/     SPY preprocessing, calibration, classical Dupire benchmarks
phase4_barriers/   Barrier PINN, Crank-Nicolson FDM, Monte Carlo, Reiner-Rubinstein
utils/             Black-Scholes formulas, normalisation, plotting
results/           Trained weights and generated figures, by phase
scratch/           Working notes, not part of the thesis
```

Market data: `phase3_market/spy_otm_2021-09-20.csv` — 1,956 preprocessed SPY OTM options,
moneyness `[0.80, 1.20]`, 7–365 days to expiry. The raw Kaggle dataset it derives from is
not tracked (see `.gitignore`).

## Requirements

Python 3.9+, `torch >= 2.0`, `numpy`, `scipy`, `matplotlib`. CPU is sufficient throughout —
no run in the thesis used a GPU.

## Licence

The thesis PDF is licensed under Creative Commons Attribution–NonCommercial–NoDerivatives.
