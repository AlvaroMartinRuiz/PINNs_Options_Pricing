"""
Phase 3 - Full Preprocessing Pipeline
Extracts 2021-09-20 SPY options, cleans, filters OTM, matches risk-free rate
from Treasury yield curve, matches dividend yield, and saves PINN-ready dataset.

Decisions:
  - Single day: 2021-09-20 (SPY=$434.01, 1956 OTM options)
  - Moneyness: K/S in [0.80, 1.20]
  - DTE: [7, 365]
  - OTM only (calls K>S, puts K<S)
  - Strict liquidity: bid>0, spread/mid<0.30, volume>=10
"""
import pandas as pd
import numpy as np
import os
import warnings
import time
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE = r"C:\_Alvaro\TFG\TFG_code"
SPY_PATH = os.path.join(BASE, "Kaggle_data", "spy_2020_2022.csv", "spy_2020_2022.csv")
YC_PATH = os.path.join(BASE, "rates", "par-yield-curve-rates-2020-2023.csv")
OUT_DIR = os.path.join(BASE, "phase3_market")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_DATE = "2021-09-20"

# =============================================================================
# 1. EXTRACT SINGLE-DAY DATA
# =============================================================================
print("=" * 70)
print(f"Phase 3 Preprocessing - Target Date: {TARGET_DATE}")
print("=" * 70)

print("\n[1/6] Extracting target date from SPY options file...")
start = time.time()

chunks = []
CHUNK_SIZE = 200_000
for chunk in pd.read_csv(SPY_PATH, chunksize=CHUNK_SIZE):
    chunk.columns = [c.strip().strip('[]') for c in chunk.columns]
    # Quick string filter before heavy processing
    if chunk['QUOTE_DATE'].dtype == object:
        mask = chunk['QUOTE_DATE'].str.strip() == TARGET_DATE
    else:
        mask = chunk['QUOTE_DATE'] == TARGET_DATE
    day_chunk = chunk[mask]
    if len(day_chunk) > 0:
        chunks.append(day_chunk.copy())

if not chunks:
    raise ValueError(f"No data found for {TARGET_DATE}!")

df = pd.concat(chunks, ignore_index=True)
elapsed = time.time() - start
print(f"  Extracted {len(df)} rows in {elapsed:.0f}s")

# =============================================================================
# 2. CLEAN & CONVERT TYPES
# =============================================================================
print("\n[2/6] Cleaning data types...")

# Strip string columns
for col in ['QUOTE_DATE', 'EXPIRE_DATE', 'QUOTE_READTIME']:
    if df[col].dtype == object:
        df[col] = df[col].str.strip()

# Convert all numeric columns
num_cols = ['C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 'C_RHO', 'C_IV',
            'C_VOLUME', 'C_LAST', 'C_BID', 'C_ASK',
            'P_BID', 'P_ASK', 'P_LAST', 'P_DELTA', 'P_GAMMA', 'P_VEGA',
            'P_THETA', 'P_RHO', 'P_IV', 'P_VOLUME']
for col in num_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')

print(f"  Raw rows for {TARGET_DATE}: {len(df)}")
print(f"  SPY underlying: ${df['UNDERLYING_LAST'].iloc[0]:.2f}")
print(f"  Unique expiries: {df['EXPIRE_DATE'].nunique()}")
print(f"  DTE range: [{df['DTE'].min():.0f}, {df['DTE'].max():.0f}]")
print(f"  Strike range: [{df['STRIKE'].min():.0f}, {df['STRIKE'].max():.0f}]")

# =============================================================================
# 3. FILTER: DTE, MONEYNESS, REMOVE DTE=0
# =============================================================================
print("\n[3/6] Applying filters...")

n_before = len(df)

# DTE filter
df = df[(df['DTE'] >= 7) & (df['DTE'] <= 365)]
print(f"  After DTE in [7, 365]: {len(df)} ({n_before - len(df)} removed)")

# Moneyness filter
df['moneyness'] = df['STRIKE'] / df['UNDERLYING_LAST']
n_before = len(df)
df = df[(df['moneyness'] >= 0.80) & (df['moneyness'] <= 1.20)]
print(f"  After K/S in [0.80, 1.20]: {len(df)} ({n_before - len(df)} removed)")

# Remove rows where both IVs are NaN
n_before = len(df)
df = df[df['C_IV'].notna() | df['P_IV'].notna()]
print(f"  After removing all-NaN IV: {len(df)} ({n_before - len(df)} removed)")

# =============================================================================
# 4. SELECT OTM OPTIONS + STRICT LIQUIDITY
# =============================================================================
print("\n[4/6] Selecting OTM options with strict liquidity filter...")

# --- OTM Calls: K > S (moneyness > 1.0) ---
otm_calls = df[df['moneyness'] > 1.0].copy()
otm_calls['mid'] = (otm_calls['C_BID'] + otm_calls['C_ASK']) / 2
otm_calls['spread_pct'] = (otm_calls['C_ASK'] - otm_calls['C_BID']) / otm_calls['mid']
otm_calls = otm_calls[
    (otm_calls['C_BID'] > 0) &
    (otm_calls['mid'] > 0) &
    (otm_calls['spread_pct'] < 0.30) &
    (otm_calls['C_VOLUME'] >= 10) &
    (otm_calls['C_IV'].notna()) & (otm_calls['C_IV'] > 0) & (otm_calls['C_IV'] < 3.0)
]
# Extract relevant columns for calls
otm_calls_out = otm_calls[['QUOTE_DATE', 'EXPIRE_DATE', 'UNDERLYING_LAST', 'STRIKE',
                            'DTE', 'moneyness', 'C_IV', 'C_BID', 'C_ASK', 'C_DELTA',
                            'C_VOLUME']].copy()
otm_calls_out = otm_calls_out.rename(columns={
    'C_IV': 'sigma_market', 'C_BID': 'bid', 'C_ASK': 'ask',
    'C_DELTA': 'delta', 'C_VOLUME': 'volume'
})
otm_calls_out['option_type'] = 'call'
otm_calls_out['mid'] = (otm_calls_out['bid'] + otm_calls_out['ask']) / 2

# --- OTM Puts: K < S (moneyness < 1.0) ---
otm_puts = df[df['moneyness'] < 1.0].copy()
otm_puts['mid'] = (otm_puts['P_BID'] + otm_puts['P_ASK']) / 2
otm_puts['spread_pct'] = (otm_puts['P_ASK'] - otm_puts['P_BID']) / otm_puts['mid']
otm_puts = otm_puts[
    (otm_puts['P_BID'] > 0) &
    (otm_puts['mid'] > 0) &
    (otm_puts['spread_pct'] < 0.30) &
    (otm_puts['P_VOLUME'] >= 10) &
    (otm_puts['P_IV'].notna()) & (otm_puts['P_IV'] > 0) & (otm_puts['P_IV'] < 3.0)
]
otm_puts_out = otm_puts[['QUOTE_DATE', 'EXPIRE_DATE', 'UNDERLYING_LAST', 'STRIKE',
                          'DTE', 'moneyness', 'P_IV', 'P_BID', 'P_ASK', 'P_DELTA',
                          'P_VOLUME']].copy()
otm_puts_out = otm_puts_out.rename(columns={
    'P_IV': 'sigma_market', 'P_BID': 'bid', 'P_ASK': 'ask',
    'P_DELTA': 'delta', 'P_VOLUME': 'volume'
})
otm_puts_out['option_type'] = 'put'
otm_puts_out['mid'] = (otm_puts_out['bid'] + otm_puts_out['ask']) / 2

# Combine
otm = pd.concat([otm_calls_out, otm_puts_out], ignore_index=True)
otm = otm.sort_values(['DTE', 'STRIKE']).reset_index(drop=True)

print(f"  OTM Calls after filtering: {len(otm_calls_out)}")
print(f"  OTM Puts  after filtering: {len(otm_puts_out)}")
print(f"  Total OTM options: {len(otm)}")

# =============================================================================
# 5. MATCH RISK-FREE RATE FROM TREASURY YIELD CURVE
# =============================================================================
print("\n[5/6] Matching risk-free rates from Treasury par yield curve...")

yc = pd.read_csv(YC_PATH)
yc.columns = [c.strip().lower() for c in yc.columns]
yc['date'] = pd.to_datetime(yc['date'], format='%m/%d/%Y')

# Find the yield curve date closest to (and <=) our target date
target_dt = pd.to_datetime(TARGET_DATE)
yc_available = yc[yc['date'] <= target_dt]
yc_row = yc_available.iloc[-1]  # Most recent date <= target
yc_date = yc_row['date']
print(f"  Yield curve date used: {yc_date.strftime('%Y-%m-%d')}")

# Build tenor-to-days mapping and extract rates
# Tenors: 1mo, 2mo, 3mo, 4mo, 6mo, 1yr, 2yr, 3yr, 5yr, 7yr, 10yr, 20yr, 30yr
tenor_days = {
    '1 mo': 30, '2 mo': 61, '3 mo': 91, '4 mo': 122,
    '6 mo': 182, '1 yr': 365, '2 yr': 730, '3 yr': 1095,
    '5 yr': 1825, '7 yr': 2555, '10 yr': 3650, '20 yr': 7300, '30 yr': 10950
}

# Extract available tenors (skip NaN)
tenor_rates = []
for tenor_name, days in tenor_days.items():
    if tenor_name in yc_row.index:
        rate = yc_row[tenor_name]
        if pd.notna(rate):
            tenor_rates.append((days, rate / 100.0))  # Convert % to decimal

tenor_rates.sort(key=lambda x: x[0])
tenor_days_arr = np.array([t[0] for t in tenor_rates])
tenor_rates_arr = np.array([t[1] for t in tenor_rates])

print(f"  Available tenors: {len(tenor_rates)}")
print(f"  Tenor days: {tenor_days_arr.tolist()}")
print(f"  Rates (%): {(tenor_rates_arr * 100).round(4).tolist()}")

# Interpolate risk-free rate for each option's DTE
def interpolate_rate(dte, tenor_days, tenor_rates):
    """Linearly interpolate Treasury rate for a given DTE."""
    if dte <= tenor_days[0]:
        return tenor_rates[0]  # Use shortest tenor
    if dte >= tenor_days[-1]:
        return tenor_rates[-1]  # Use longest tenor
    # Find surrounding tenors
    idx = np.searchsorted(tenor_days, dte)
    d0, d1 = tenor_days[idx-1], tenor_days[idx]
    r0, r1 = tenor_rates[idx-1], tenor_rates[idx]
    # Linear interpolation
    weight = (dte - d0) / (d1 - d0)
    return r0 + weight * (r1 - r0)

otm['r'] = otm['DTE'].apply(lambda d: interpolate_rate(d, tenor_days_arr, tenor_rates_arr))

# Verify interpolation
print(f"\n  Rate interpolation check:")
sample_dtes = [7, 30, 60, 90, 180, 365]
for dte in sample_dtes:
    r = interpolate_rate(dte, tenor_days_arr, tenor_rates_arr)
    print(f"    DTE={dte:>4d} -> r = {r*100:.4f}%")

# =============================================================================
# 6. MATCH DIVIDEND YIELD
# =============================================================================
print("\n[6/6] Matching S&P 500 dividend yield...")

# Monthly S&P 500 dividend yields from multpl.com (trailing 12-month)
# Source: https://www.multpl.com/s-p-500-dividend-yield/table/by-month
div_yields = {
    '2021-01': 1.49, '2021-02': 1.45, '2021-03': 1.40,
    '2021-04': 1.38, '2021-05': 1.36, '2021-06': 1.34,
    '2021-07': 1.29, '2021-08': 1.31, '2021-09': 1.32,
    '2021-10': 1.29, '2021-11': 1.25, '2021-12': 1.27,
    '2022-01': 1.33, '2022-02': 1.35, '2022-03': 1.34,
    '2022-04': 1.39, '2022-05': 1.53, '2022-06': 1.61,
    '2022-07': 1.60, '2022-08': 1.55, '2022-09': 1.74,
    '2022-10': 1.77, '2022-11': 1.68, '2022-12': 1.68,
}

# For 2021-09-20, the dividend yield is approximately 1.32%
target_month = TARGET_DATE[:7]
q = div_yields.get(target_month, 1.32) / 100.0  # Convert to decimal
otm['q'] = q
print(f"  Dividend yield for {target_month}: {q*100:.2f}%")

# =============================================================================
# 7. BUILD FINAL PINN-READY DATASET
# =============================================================================
print("\n" + "=" * 70)
print("Building final PINN-ready dataset...")
print("=" * 70)

# Rename and select final columns
otm['S'] = otm['UNDERLYING_LAST']
otm['K'] = otm['STRIKE']
otm['tau'] = otm['DTE'] / 365.0  # Time to expiry in years
otm['V_market'] = otm['mid']     # Mid-price as market price

final = otm[['S', 'K', 'tau', 'r', 'q', 'sigma_market', 'V_market',
             'moneyness', 'option_type', 'delta', 'volume',
             'bid', 'ask', 'DTE', 'EXPIRE_DATE']].copy()

# Sort by moneyness then tau
final = final.sort_values(['tau', 'moneyness']).reset_index(drop=True)

# Save
out_path = os.path.join(OUT_DIR, f"spy_otm_{TARGET_DATE}.csv")
final.to_csv(out_path, index=False)

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*70}")
print(f"DATASET SUMMARY - {TARGET_DATE}")
print(f"{'='*70}")
print(f"  Total options:  {len(final)}")
print(f"  OTM Calls:      {(final['option_type']=='call').sum()}")
print(f"  OTM Puts:       {(final['option_type']=='put').sum()}")
print(f"  Underlying (S): ${final['S'].iloc[0]:.2f}")
print(f"  Strike range:   ${final['K'].min():.0f} - ${final['K'].max():.0f}")
print(f"  Moneyness:      [{final['moneyness'].min():.4f}, {final['moneyness'].max():.4f}]")
print(f"  DTE range:      [{final['DTE'].min():.0f}, {final['DTE'].max():.0f}]")
print(f"  tau range:      [{final['tau'].min():.4f}, {final['tau'].max():.4f}]")
print(f"  r range:        [{final['r'].min()*100:.4f}%, {final['r'].max()*100:.4f}%]")
print(f"  q (div yield):  {q*100:.2f}%")
print(f"  sigma range:    [{final['sigma_market'].min():.4f}, {final['sigma_market'].max():.4f}]")
print(f"  V_market range: [${final['V_market'].min():.2f}, ${final['V_market'].max():.2f}]")
print(f"\n  Unique expiries: {final['EXPIRE_DATE'].nunique()}")
print(f"  Expiry dates:   {sorted(final['EXPIRE_DATE'].unique())}")

# DTE distribution
print(f"\n  Options per DTE bucket:")
dte_bins = [0, 14, 30, 60, 90, 180, 365, 1000]
dte_labels = ['7-14d', '15-30d', '31-60d', '61-90d', '91-180d', '181-365d']
final['dte_bin'] = pd.cut(final['DTE'], bins=dte_bins, labels=dte_labels + ['365+'])
print(final['dte_bin'].value_counts().sort_index().to_string())

# Volume distribution
print(f"\n  Volume stats:")
print(final['volume'].describe())

print(f"\n  Saved to: {out_path}")
print(f"  File size: {os.path.getsize(out_path) / 1024:.1f} KB")
print("\nDone!")
