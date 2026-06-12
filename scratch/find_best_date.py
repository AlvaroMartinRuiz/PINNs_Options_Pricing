"""
Phase 3 - Step 1: Find the best candidate trading date for single-day calibration.
Scans the full SPY dataset in chunks, applying filters, and ranks dates by OTM option count.

Criteria:
  - Quote date >= 2021-01-01 (calmer markets, post-COVID)
  - DTE in [7, 365]
  - Moneyness K/S in [0.80, 1.20]
  - OTM only (calls: K > S, puts: K < S)
  - Strict liquidity: bid > 0, spread/mid < 0.30, volume >= 10
"""
import pandas as pd
import numpy as np
import time

SPY_PATH = r"C:\_Alvaro\TFG\TFG_code\Kaggle_data\spy_2020_2022.csv\spy_2020_2022.csv"
CHUNK_SIZE = 200_000

print("Scanning SPY options file in chunks...")
start = time.time()

date_stats = {}  # date -> {otm_calls, otm_puts, total_otm, underlying}

for i, chunk in enumerate(pd.read_csv(SPY_PATH, chunksize=CHUNK_SIZE)):
    # Clean column names
    chunk.columns = [c.strip().strip('[]') for c in chunk.columns]
    
    # Parse date, filter to 2021+
    chunk['QUOTE_DATE'] = chunk['QUOTE_DATE'].str.strip()
    chunk = chunk[chunk['QUOTE_DATE'] >= '2021-01-01']
    if len(chunk) == 0:
        if (i + 1) % 5 == 0:
            print(f"  Chunk {i+1} processed ({(i+1)*CHUNK_SIZE/1e6:.1f}M rows), {len(date_stats)} dates found so far...")
        continue
    
    # Convert numeric columns robustly (handle both string and numeric dtypes)
    num_cols = ['C_DELTA', 'C_IV', 'C_VOLUME', 'C_BID', 'C_ASK',
                'P_DELTA', 'P_IV', 'P_VOLUME', 'P_BID', 'P_ASK']
    for col in num_cols:
        try:
            chunk[col] = pd.to_numeric(chunk[col].astype(str).str.strip(), errors='coerce')
        except Exception:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    
    # Basic filters
    chunk = chunk[(chunk['DTE'] >= 7) & (chunk['DTE'] <= 365)]
    chunk['moneyness'] = chunk['STRIKE'] / chunk['UNDERLYING_LAST']
    chunk = chunk[(chunk['moneyness'] >= 0.80) & (chunk['moneyness'] <= 1.20)]
    
    # --- OTM Calls: K > S (moneyness > 1) ---
    otm_calls = chunk[chunk['moneyness'] > 1.0].copy()
    otm_calls['mid'] = (otm_calls['C_BID'] + otm_calls['C_ASK']) / 2
    otm_calls = otm_calls[
        (otm_calls['C_BID'] > 0) &
        (otm_calls['mid'] > 0) &
        ((otm_calls['C_ASK'] - otm_calls['C_BID']) / otm_calls['mid'] < 0.30) &
        (otm_calls['C_VOLUME'] >= 10) &
        (otm_calls['C_IV'].notna()) & (otm_calls['C_IV'] > 0) & (otm_calls['C_IV'] < 3.0)
    ]
    
    # --- OTM Puts: K < S (moneyness < 1) ---
    otm_puts = chunk[chunk['moneyness'] < 1.0].copy()
    otm_puts['mid'] = (otm_puts['P_BID'] + otm_puts['P_ASK']) / 2
    otm_puts = otm_puts[
        (otm_puts['P_BID'] > 0) &
        (otm_puts['mid'] > 0) &
        ((otm_puts['P_ASK'] - otm_puts['P_BID']) / otm_puts['mid'] < 0.30) &
        (otm_puts['P_VOLUME'] >= 10) &
        (otm_puts['P_IV'].notna()) & (otm_puts['P_IV'] > 0) & (otm_puts['P_IV'] < 3.0)
    ]
    
    # Aggregate by date
    for date in set(otm_calls['QUOTE_DATE'].unique()) | set(otm_puts['QUOTE_DATE'].unique()):
        nc = (otm_calls['QUOTE_DATE'] == date).sum()
        np_ = (otm_puts['QUOTE_DATE'] == date).sum()
        underlying = chunk.loc[chunk['QUOTE_DATE'] == date, 'UNDERLYING_LAST'].iloc[0] if date in chunk['QUOTE_DATE'].values else None
        
        if date not in date_stats:
            date_stats[date] = {'otm_calls': 0, 'otm_puts': 0, 'underlying': underlying}
        date_stats[date]['otm_calls'] += nc
        date_stats[date]['otm_puts'] += np_
        if underlying is not None:
            date_stats[date]['underlying'] = underlying
    
    if (i + 1) % 5 == 0:
        elapsed = time.time() - start
        print(f"  Chunk {i+1} processed ({(i+1)*CHUNK_SIZE/1e6:.1f}M rows), {len(date_stats)} dates found, {elapsed:.0f}s elapsed")

elapsed = time.time() - start
print(f"\nScan complete in {elapsed:.0f}s. Found {len(date_stats)} candidate dates.\n")

# Build results dataframe
results = pd.DataFrame([
    {'date': d, 'otm_calls': v['otm_calls'], 'otm_puts': v['otm_puts'],
     'total_otm': v['otm_calls'] + v['otm_puts'], 'underlying': v['underlying']}
    for d, v in date_stats.items()
])
results = results.sort_values('total_otm', ascending=False)

print("=" * 70)
print("TOP 20 CANDIDATE DATES (by total OTM options after strict filtering)")
print("=" * 70)
print(results.head(20).to_string(index=False))

print("\n--- Distribution of total OTM count ---")
print(results['total_otm'].describe())

# Save for later use
results.to_csv(r"C:\_Alvaro\TFG\TFG_code\scratch\candidate_dates.csv", index=False)
print(f"\nFull results saved to scratch/candidate_dates.csv")
