"""
Phase 3 - Validation Plots for the preprocessed SPY OTM dataset.
Generates diagnostic plots to verify data quality before PINN training.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE = r"C:\_Alvaro\TFG\TFG_code"
DATA_PATH = os.path.join(BASE, "phase3_market", "spy_otm_2021-09-20.csv")
PLOT_DIR = os.path.join(BASE, "phase3_market", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} options from {DATA_PATH}")

# --- Plot 1: IV Smile by DTE bucket ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Implied Volatility Smile by DTE Bucket (2021-09-20)', fontsize=14, fontweight='bold')

dte_bins = [(7, 14, '7-14 DTE'), (15, 30, '15-30 DTE'), (31, 60, '31-60 DTE'),
            (61, 90, '61-90 DTE'), (91, 180, '91-180 DTE'), (181, 365, '181-365 DTE')]

for ax, (lo, hi, label) in zip(axes.flatten(), dte_bins):
    mask = (df['DTE'] >= lo) & (df['DTE'] <= hi)
    subset = df[mask]
    
    calls = subset[subset['option_type'] == 'call']
    puts = subset[subset['option_type'] == 'put']
    
    ax.scatter(calls['moneyness'], calls['sigma_market'], alpha=0.5, s=15, c='royalblue', label=f'Calls ({len(calls)})')
    ax.scatter(puts['moneyness'], puts['sigma_market'], alpha=0.5, s=15, c='crimson', label=f'Puts ({len(puts)})')
    ax.set_title(f'{label} (n={len(subset)})')
    ax.set_xlabel('Moneyness (K/S)')
    ax.set_ylabel('Implied Volatility')
    ax.legend(fontsize=8)
    ax.set_xlim(0.78, 1.22)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'iv_smile_by_dte.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: iv_smile_by_dte.png")

# --- Plot 2: IV Surface Heatmap ---
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
sc = ax.scatter(df['moneyness'], df['tau'], c=df['sigma_market'], 
                cmap='RdYlBu_r', s=8, alpha=0.7, vmin=0.10, vmax=0.50)
cbar = plt.colorbar(sc, ax=ax, label='Implied Volatility')
ax.set_xlabel('Moneyness (K/S)', fontsize=12)
ax.set_ylabel('Time to Expiry (years)', fontsize=12)
ax.set_title('Market Implied Volatility Surface - SPY 2021-09-20', fontsize=14, fontweight='bold')
ax.axvline(x=1.0, color='white', linestyle='--', alpha=0.7, linewidth=1)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'iv_surface_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: iv_surface_scatter.png")

# --- Plot 3: Moneyness and DTE distributions ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Moneyness histogram
axes[0].hist(df['moneyness'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(x=1.0, color='red', linestyle='--', linewidth=1.5, label='ATM')
axes[0].set_xlabel('Moneyness (K/S)')
axes[0].set_ylabel('Count')
axes[0].set_title('Moneyness Distribution')
axes[0].legend()

# DTE histogram
axes[1].hist(df['DTE'], bins=40, color='seagreen', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Days to Expiry')
axes[1].set_ylabel('Count')
axes[1].set_title('DTE Distribution')

# Volume (log scale)
axes[2].hist(np.log10(df['volume'].clip(lower=1)), bins=40, color='coral', edgecolor='white', alpha=0.8)
axes[2].set_xlabel('log10(Volume)')
axes[2].set_ylabel('Count')
axes[2].set_title('Volume Distribution (log10)')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.suptitle('Data Distributions - SPY OTM Options 2021-09-20', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: distributions.png")

# --- Plot 4: Risk-Free Rate Term Structure ---
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# Sort by DTE for line plot
rate_data = df[['DTE', 'r']].drop_duplicates().sort_values('DTE')
ax.plot(rate_data['DTE'], rate_data['r'] * 100, 'o-', color='navy', markersize=3, alpha=0.7)
ax.set_xlabel('Days to Expiry')
ax.set_ylabel('Risk-Free Rate (%)')
ax.set_title('Interpolated Treasury Rate by DTE - 2021-09-20', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add Treasury tenor markers
tenor_days = [30, 61, 91, 182, 365]
tenor_rates = [0.06, 0.05, 0.04, 0.05, 0.07]
ax.scatter(tenor_days, tenor_rates, color='red', s=80, zorder=5, label='Treasury Tenors', marker='D')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'rate_term_structure.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: rate_term_structure.png")

# --- Plot 5: Bid-Ask spread quality ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df['spread'] = df['ask'] - df['bid']
df['spread_pct'] = df['spread'] / ((df['bid'] + df['ask']) / 2)

axes[0].scatter(df['moneyness'], df['spread_pct'] * 100, alpha=0.3, s=5, c='purple')
axes[0].set_xlabel('Moneyness (K/S)')
axes[0].set_ylabel('Bid-Ask Spread (%)')
axes[0].set_title('Spread % vs Moneyness')
axes[0].set_ylim(0, 35)
axes[0].axhline(y=30, color='red', linestyle='--', alpha=0.5, label='30% cutoff')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(df['DTE'], df['spread_pct'] * 100, alpha=0.3, s=5, c='teal')
axes[1].set_xlabel('Days to Expiry')
axes[1].set_ylabel('Bid-Ask Spread (%)')
axes[1].set_title('Spread % vs DTE')
axes[1].set_ylim(0, 35)
axes[1].axhline(y=30, color='red', linestyle='--', alpha=0.5, label='30% cutoff')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Bid-Ask Spread Quality - SPY OTM 2021-09-20', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'spread_quality.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: spread_quality.png")

print(f"\nAll plots saved to {PLOT_DIR}")
print("Done!")
