"""
Phase 3 Data Exploration Script
Explores SPY options data, yield curve data, Fed Funds, SOFR, and dividend yield data.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. SPY OPTIONS DATA
# ============================================================
print("=" * 70)
print("1. SPY OPTIONS DATA ANALYSIS")
print("=" * 70)

spy_path = r"C:\_Alvaro\TFG\TFG_code\Kaggle_data\spy_2020_2022.csv\spy_2020_2022.csv"

# Read first 200k rows for quick exploration
df = pd.read_csv(spy_path, nrows=200000)

# Clean column names (they have leading spaces and brackets)
df.columns = [c.strip().strip('[]') for c in df.columns]

# Identify which numeric columns are read as object (strings)
expected_numeric = ['C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 'C_RHO', 'C_IV',
                    'C_VOLUME', 'C_LAST', 'C_BID', 'C_ASK',
                    'P_BID', 'P_ASK', 'P_LAST', 'P_DELTA', 'P_GAMMA', 'P_VEGA',
                    'P_THETA', 'P_RHO', 'P_IV', 'P_VOLUME']

print("\n--- Columns read as object that should be numeric ---")
for col in expected_numeric:
    if df[col].dtype == object:
        # Check sample values
        sample_vals = df[col].str.strip().unique()[:10]
        empty_count = (df[col].str.strip() == '').sum()
        print(f"  {col}: {empty_count} empty strings out of {len(df)}")
        # Show some non-empty samples
        non_empty = df[col][df[col].str.strip() != ''].head(3).tolist()
        print(f"    Sample values: {non_empty}")

# Convert numeric columns
for col in expected_numeric:
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col].str.strip(), errors='coerce')

print(f"\nShape (first 200k rows): {df.shape}")
print(f"Total rows in file: ~3,589,080")

# Date handling
df['QUOTE_DATE'] = df['QUOTE_DATE'].str.strip()
df['EXPIRE_DATE'] = df['EXPIRE_DATE'].str.strip()

print(f"\n--- Date Range ---")
print(f"Quote dates: {df['QUOTE_DATE'].min()} to {df['QUOTE_DATE'].max()}")
print(f"Expire dates: {df['EXPIRE_DATE'].min()} to {df['EXPIRE_DATE'].max()}")

# Unique quote dates
unique_dates = df['QUOTE_DATE'].nunique()
print(f"Unique quote dates (in 200k sample): {unique_dates}")

print(f"\n--- DTE (Days to Expiration) ---")
print(df['DTE'].describe())
print(f"\nDTE = 0 count: {(df['DTE'] == 0).sum()}")

print(f"\n--- Underlying Price (SPY) ---")
print(df['UNDERLYING_LAST'].describe())

print(f"\n--- Strike ---")
print(df['STRIKE'].describe())

print(f"\n--- Moneyness Analysis (K/S) ---")
df['moneyness'] = df['STRIKE'] / df['UNDERLYING_LAST']
print(df['moneyness'].describe())
print(f"\nMoneyness distribution:")
bins = [0, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 100]
labels = ['<0.7', '0.7-0.8', '0.8-0.9', '0.9-0.95', '0.95-1.0', '1.0-1.05', '1.05-1.1', '1.1-1.2', '1.2-1.3', '>1.3']
df['m_bin'] = pd.cut(df['moneyness'], bins=bins, labels=labels)
print(df['m_bin'].value_counts().sort_index().to_string())

print(f"\n--- Call Implied Volatility ---")
c_iv = df['C_IV']
print(f"Non-null: {c_iv.notna().sum()}, NaN: {c_iv.isna().sum()}")
print(f"Zero: {(c_iv == 0).sum()}")
c_iv_valid = c_iv[(c_iv > 0) & (c_iv.notna())]
print(f"Valid (>0): {len(c_iv_valid)}")
if len(c_iv_valid) > 0:
    print(c_iv_valid.describe())

print(f"\n--- Put Implied Volatility ---")
p_iv = df['P_IV']
print(f"Non-null: {p_iv.notna().sum()}, NaN: {p_iv.isna().sum()}")
print(f"Zero: {(p_iv == 0).sum()}")
p_iv_valid = p_iv[(p_iv > 0) & (p_iv.notna())]
print(f"Valid (>0): {len(p_iv_valid)}")
if len(p_iv_valid) > 0:
    print(p_iv_valid.describe())

# IV sanity: check for extreme values
print(f"\n--- IV Extreme Values ---")
print(f"C_IV > 5 (500%): {(c_iv > 5).sum()}")
print(f"C_IV > 10: {(c_iv > 10).sum()}")
print(f"P_IV > 5 (500%): {(p_iv > 5).sum()}")
print(f"P_IV > 10: {(p_iv > 10).sum()}")

print(f"\n--- Volume Analysis ---")
print("Call Volume:")
print(df['C_VOLUME'].describe())
print(f"\nPut Volume:")
print(df['P_VOLUME'].describe())
print(f"\nC_VOLUME NaN: {df['C_VOLUME'].isna().sum()}")
print(f"P_VOLUME NaN: {df['P_VOLUME'].isna().sum()}")
print(f"C_VOLUME = 0: {(df['C_VOLUME'] == 0).sum()}")
print(f"P_VOLUME = 0: {(df['P_VOLUME'] == 0).sum()}")

print(f"\n--- Bid/Ask Analysis ---")
df['c_spread'] = df['C_ASK'] - df['C_BID']
df['p_spread'] = df['P_ASK'] - df['P_BID']
print(f"Call spread stats:")
print(df['c_spread'].describe())
print(f"\nPut spread stats:")
print(df['p_spread'].describe())

print(f"\nNegative call spreads: {(df['c_spread'] < 0).sum()}")
print(f"Negative put spreads: {(df['p_spread'] < 0).sum()}")
print(f"Zero call bid: {(df['C_BID'] == 0).sum()}")
print(f"Zero put bid: {(df['P_BID'] == 0).sum()}")

# Mid prices
df['c_mid'] = (df['C_BID'] + df['C_ASK']) / 2
df['p_mid'] = (df['P_BID'] + df['P_ASK']) / 2
print(f"\nCall mid=0 count: {(df['c_mid'] == 0).sum()}")
print(f"Put mid=0 count: {(df['p_mid'] == 0).sum()}")

print(f"\n--- Greeks Range Checks ---")
print(f"C_DELTA: [{df['C_DELTA'].min():.4f}, {df['C_DELTA'].max():.4f}]")
print(f"P_DELTA: [{df['P_DELTA'].min():.4f}, {df['P_DELTA'].max():.4f}]")
print(f"C_GAMMA: [{df['C_GAMMA'].min():.6f}, {df['C_GAMMA'].max():.6f}]")
print(f"C_THETA: [{df['C_THETA'].min():.4f}, {df['C_THETA'].max():.4f}]")

# OTM detection
otm_calls = df[(df['C_DELTA'] > 0) & (df['C_DELTA'] < 0.5)]
otm_puts = df[(df['P_DELTA'] < 0) & (df['P_DELTA'] > -0.5)]
print(f"\nOTM Calls (0 < delta < 0.5): {len(otm_calls)}")
print(f"OTM Puts (-0.5 < delta < 0): {len(otm_puts)}")

# Check for DTE = 0 rows
print(f"\n--- DTE=0 analysis ---")
dte0 = df[df['DTE'] == 0]
print(f"DTE=0 rows: {len(dte0)}")
if len(dte0) > 0:
    print(f"  C_IV mean when DTE=0: {dte0['C_IV'].mean():.4f}")
    print(f"  Many have 0 IV? {(dte0['C_IV'] == 0).sum()}")

# Check QUOTE_TIME_HOURS
print(f"\n--- QUOTE_TIME_HOURS ---")
print(df['QUOTE_TIME_HOURS'].value_counts().head(10))

# ============================================================
# 2. YIELD CURVE DATA
# ============================================================
print("\n" + "=" * 70)
print("2. YIELD CURVE DATA ANALYSIS")
print("=" * 70)

yc_2020 = pd.read_csv(r"C:\_Alvaro\TFG\TFG_code\rates\par-yield-curve-rates-2020-2023.csv")
yc_2010 = pd.read_csv(r"C:\_Alvaro\TFG\TFG_code\rates\par-yield-curve-rates-2010-2019.csv")

print(f"\n--- 2020-2023 Yield Curve ---")
print(f"Shape: {yc_2020.shape}")
print(f"Columns: {list(yc_2020.columns)}")
print(f"Date range: {yc_2020['date'].iloc[0]} to {yc_2020['date'].iloc[-1]}")

print(f"\n--- Missing values in 2020-2023 ---")
missing_yc = yc_2020.isnull().sum()
print(missing_yc.to_string())

# Check the 4-month tenor column  
print(f"\n--- 4-month tenor availability ---")
four_mo = yc_2020['4 mo']
non_null_4mo = four_mo.notna().sum()
total = len(four_mo)
first_valid_idx = four_mo.first_valid_index()
print(f"Non-null: {non_null_4mo}/{total}")
if first_valid_idx is not None:
    print(f"First non-null at row {first_valid_idx}: date = {yc_2020.loc[first_valid_idx, 'date']}")

# Check the 2010-2019 file
print(f"\n--- 2010-2019 Yield Curve ---")
print(f"Shape: {yc_2010.shape}")
print(f"Columns: {list(yc_2010.columns)}")
print(f"Date range: {yc_2010['date'].iloc[0]} to {yc_2010['date'].iloc[-1]}")
missing_yc10 = yc_2010.isnull().sum()
missing_yc10 = missing_yc10[missing_yc10 > 0]
print(f"Missing:\n{missing_yc10.to_string()}")

# Check date format consistency
print(f"\n--- Date format samples ---")
print(f"2020 file: {yc_2020['date'].head(3).tolist()}")
print(f"2010 file: {yc_2010['date'].head(3).tolist()}")

# ============================================================
# 3. FED FUNDS RATE
# ============================================================
print("\n" + "=" * 70)
print("3. FED FUNDS RATE ANALYSIS")
print("=" * 70)

ff = pd.read_csv(r"C:\_Alvaro\TFG\TFG_code\rates\FEDFUNDS.csv")
ff['observation_date'] = pd.to_datetime(ff['observation_date'])
print(f"Shape: {ff.shape}")
print(f"Date range: {ff['observation_date'].min()} to {ff['observation_date'].max()}")
ff_relevant = ff[(ff['observation_date'] >= '2020-01-01') & (ff['observation_date'] <= '2022-12-31')]
print(f"Rows in 2020-2022: {len(ff_relevant)}")
print(f"Rate range (2020-2022): {ff_relevant['FEDFUNDS'].min()} to {ff_relevant['FEDFUNDS'].max()}")
print(f"Monthly frequency: {ff_relevant['observation_date'].diff().mode().values}")

# ============================================================
# 4. SOFR DATA
# ============================================================
print("\n" + "=" * 70)
print("4. SOFR DATA ANALYSIS")
print("=" * 70)

sofr = pd.read_excel(r"C:\_Alvaro\TFG\TFG_code\rates\sofr.xlsx")
print(f"Shape: {sofr.shape}")
print(f"Key columns: Effective Date, Rate Type, Rate (%)")
print(f"Date range: {sofr['Effective Date'].iloc[-1]} to {sofr['Effective Date'].iloc[0]}")
sofr['date_parsed'] = pd.to_datetime(sofr['Effective Date'], format='mixed')
sofr_relevant = sofr[(sofr['date_parsed'] >= '2020-01-01') & (sofr['date_parsed'] <= '2022-12-31')]
print(f"Rows in 2020-2022: {len(sofr_relevant)}")
print(f"Rate range (2020-2022): {sofr_relevant['Rate (%)'].min()} to {sofr_relevant['Rate (%)'].max()}")
print(f"Daily frequency? Business days in sample: {len(sofr_relevant)}")

# ============================================================
# 5. CRITICAL FINDINGS SUMMARY  
# ============================================================
print("\n" + "=" * 70)
print("5. CRITICAL FINDINGS FOR PREPROCESSING")
print("=" * 70)

print("""
ISSUE 1: Many numeric columns read as 'object' (strings)
  - C_DELTA, C_GAMMA, C_VEGA, C_THETA, C_RHO, C_IV, C_VOLUME, C_LAST
  - C_BID, C_ASK, P_BID, P_ASK, P_LAST, P_DELTA, etc.
  - Contains empty strings (C_VOLUME, C_LAST, P_VOLUME, P_LAST have empties)
  - Need: pd.to_numeric(..., errors='coerce') after stripping whitespace

ISSUE 2: Yield curve '4 mo' tenor is missing in early rows
  - Need to handle interpolation without this tenor in early dates

ISSUE 3: DTE=0 rows (expiration day options)
  - Should be filtered out for PINN training

ISSUE 4: Extreme IV values and zero IVs
  - Need to filter C_IV/P_IV = 0 (no market/no computation)
  - Need to cap extreme IVs (> 500% etc.)

ISSUE 5: File is ~1.2 GB with ~3.6M rows
  - Need efficient loading strategy (chunked or filtered read)

ISSUE 6: Dividend yield data needs manual extraction or web scraping
  - multpl.com table not parsed by simple HTTP fetch
  - Data already scraped from browser: 36 monthly data points 2020-2022
""")

print("\n--- Done ---")
