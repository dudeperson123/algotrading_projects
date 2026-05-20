import os, glob, time, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


DATA_DIR = "PatternMatchingResults"
N_WORKERS = max(2, int(os.cpu_count() // 2))
CHUNKSIZE = 64
MIN_KEEP = 100
BASE_TOP_FRAC = 0.05        # base top fraction
TOP_FRAC_MIN = 0.03
TOP_FRAC_MAX = 0.12

BASE_CAP = 0.18
TARGET_VOL = 0.02         
ROLL_VOL = 50           
VOL_FLOOR = 1e-4           # prevents division by zero / near-zero vol
MOM_LOOKBACK = 20
MOM_UP = 1.25
MOM_DOWN = 0.80
DD_REDUCE_THRESHOLD = 0.25
DD_REDUCE_FACTOR = 0.65

COEFFS = {
    "ExpectedReturn": 0.12,
    "MedianReturn": 0.23,
    "WinRate": 0.20,
    "MatchQuality": 0.09,
    "Consistency": 0.21,
    "Liquidity": 0.06,
    "C_times_Median": 0.03,
    "WtimesE": 0.06
}

# Safety
RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)


def safe_annualize(total_return, periods, periods_per_year=252.0):
    if periods <= 0 or total_return is None or np.isnan(total_return):
        return np.nan
    try:
        return (1.0 + total_return) ** (periods_per_year / max(1.0, periods)) - 1.0
    except Exception:
        return np.nan

def compute_sharpe_from_equity(eq_series, periods_per_year=252.0):
    if len(eq_series) < 2:
        return np.nan
    rets = eq_series.pct_change().dropna().values
    if len(rets) < 2:
        return np.nan
    mu = np.nanmean(rets) * periods_per_year
    vol = np.nanstd(rets) * math.sqrt(periods_per_year)
    return mu / vol if vol > 1e-12 else np.nan

def compute_sortino_from_equity(eq_series, periods_per_year=252.0):
    if len(eq_series) < 2:
        return np.nan
    rets = eq_series.pct_change().dropna().values
    mu = np.nanmean(rets) * periods_per_year
    neg = [r for r in rets if r < 0]
    if len(neg) == 0:
        return np.nan
    downside = np.nanstd(neg) * math.sqrt(periods_per_year)
    return mu / downside if downside > 1e-12 else np.nan

t0 = time.time()
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print(f"Found {len(files):,} CSV windows in {DATA_DIR}")

def load_one(path):
    try:
        df = pd.read_csv(path)
        df["_DateKey"] = os.path.basename(path).split(".")[0]
        return df
    except Exception:
        return pd.DataFrame()

with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    dfs = list(tqdm(ex.map(load_one, files, chunksize=CHUNKSIZE),
                    total=len(files), desc="Loading windows"))
df_all = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
print(f"Loaded {len(df_all):,} rows in {time.time()-t0:.1f}s\n")

def process_group(df_group):
    if "FilteredReturn" not in df_group.columns:
        return None
    # interactions
    if "Consistency" in df_group.columns and "MedianReturn" in df_group.columns:
        df_group["C_times_Median"] = df_group["Consistency"] * df_group["MedianReturn"]
    if "WinRate" in df_group.columns and "ExpectedReturn" in df_group.columns:
        df_group["WtimesE"] = df_group["WinRate"] * df_group["ExpectedReturn"]

    # pick present coeffs
    present = [k for k in COEFFS if k in df_group.columns]
    if not present:
        return None

    # min-max normalize present features (window-local)
    for k in present:
        arr = df_group[k].astype(float).values
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        rng = (hi - lo) if (hi - lo) > 1e-12 else 1.0
        df_group[k] = (arr - lo) / rng

    # stability adj: prefer consistent/winrate
    if "Consistency" in df_group.columns and "WinRate" in df_group.columns:
        df_group["StabilityAdj"] = (df_group["Consistency"].rank(pct=True) * 0.4 +
                                    df_group["WinRate"].rank(pct=True) * 0.6)
    else:
        df_group["StabilityAdj"] = 1.0

    comp = np.zeros(len(df_group), dtype=float)
    for k in present:
        comp += df_group[k].astype(float).fillna(0.0).values * float(COEFFS.get(k, 0.0))
    df_group["CompositeScore"] = comp * df_group["StabilityAdj"]

    # adaptive top frac: scale by dispersion of CompositeScore
    score_std = float(np.nanstd(df_group["CompositeScore"].values))
    score_mean = float(np.nanmean(df_group["CompositeScore"].values))
    # if mean close to 0, avoid division by zero
    dispersion = score_std / (abs(score_mean) + 1e-9)
    adj_frac = float(np.clip(BASE_TOP_FRAC * (1.0 + dispersion), TOP_FRAC_MIN, TOP_FRAC_MAX))
    n_keep = max(MIN_KEEP, int(len(df_group) * adj_frac))

    df_sel = df_group.sort_values("CompositeScore", ascending=False).head(n_keep).copy()
    total = df_sel["CompositeScore"].sum()
    if total <= 0 or np.isclose(total, 0.0):
        df_sel["SelWeight"] = 1.0 / len(df_sel)
    else:
        df_sel["SelWeight"] = df_sel["CompositeScore"] / total

    filt = df_sel["FilteredReturn"].astype(float).fillna(0.0).values
    w = df_sel["SelWeight"].astype(float).values
    port_ret = float(np.dot(filt, w) / 100.0)  # fraction form
    return {"Date": df_group["_DateKey"].iloc[0], "Realized": port_ret,
            "NumSelected": len(df_sel), "UniverseSize": len(df_group),
            "CompositeStd": score_std, "CompositeMean": score_mean}

# group and process in parallel
groups = list(df_all.groupby("_DateKey"))
results = []
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = {ex.submit(process_group, g[1]): g[0] for g in groups}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring windows"):
        r = fut.result()
        if r:
            results.append(r)

df_perf = pd.DataFrame(results).sort_values("Date").reset_index(drop=True)
print(f"Processed {len(df_perf):,} windows in {time.time()-t0:.1f}s\n")

df_perf["RealizedBase"] = df_perf["Realized"] * BASE_CAP
caps = []
realized_adj = []
equity = [1.0]
recent_realized = []

for i in range(len(df_perf)):
    # rolling vol from previous realized base returns
    if i == 0:
        prev = np.array([])
    else:
        start = max(0, i - ROLL_VOL)
        prev = df_perf.loc[start:i-1, "RealizedBase"].to_numpy()
    if prev.size < 2:
        vol = VOL_FLOOR
    else:
        vol = float(np.nanstd(prev) * math.sqrt(252.0))
        if np.isnan(vol) or vol < VOL_FLOOR:
            vol = VOL_FLOOR

    scale = float(TARGET_VOL / vol)
    cap = float(np.clip(BASE_CAP * scale, 0.04, 0.60))  # allow bigger peak to catch upside

    # drawdown-based reduction
    current_dd = (equity[-1] / max(equity)) - 1.0
    if current_dd <= -DD_REDUCE_THRESHOLD:
        cap *= DD_REDUCE_FACTOR

    # momentum/recent-win boost: check recent realized outcomes
    if len(recent_realized) >= MOM_LOOKBACK:
        win_frac = np.mean([1 if r > 0 else 0 for r in recent_realized[-MOM_LOOKBACK:]])
        if win_frac > 0.55:
            cap *= MOM_UP
        elif win_frac < 0.45:
            cap *= MOM_DOWN

    cap = float(np.clip(cap, 0.04, 0.60))
    caps.append(cap)
    r = float(df_perf.at[i, "Realized"] * cap)
    realized_adj.append(r)
    recent_realized.append(df_perf.at[i, "Realized"])
    equity.append(equity[-1] * (1 + r))

df_perf["CapUsed"] = caps
df_perf["RealizedAdj"] = realized_adj
df_perf["_Equity"] = pd.Series(equity[1:]).values

eq = df_perf["_Equity"].astype(float)
# pct-change returns
if len(eq) > 1:
    rets = pd.Series(eq).pct_change().dropna().values
else:
    rets = np.array([])

total_return = (eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 0 else np.nan
ann_ret = safe_annualize(total_return, len(df_perf), periods_per_year=252.0)
sharpe = compute_sharpe_from_equity(pd.Series(eq), periods_per_year=252.0)
sortino = compute_sortino_from_equity(pd.Series(eq), periods_per_year=252.0)
max_dd = ((pd.Series(eq) / pd.Series(eq).cummax()) - 1.0).min() if len(eq)>0 else np.nan
calmar = (-ann_ret / max_dd) if (not np.isnan(max_dd) and max_dd < 0) else np.nan
win_rate = float((pd.Series(df_perf["RealizedAdj"]) > 0).mean() * 100) if len(df_perf)>0 else np.nan
avg_win = float(pd.Series(df_perf["RealizedAdj"]).mean() * 100) if len(df_perf)>0 else np.nan
std_win = float(pd.Series(df_perf["RealizedAdj"]).std() * 100) if len(df_perf)>0 else np.nan

# rolling Sharpe (last 1 year approx)
rolling_sharpe = np.nan
if len(rets) >= 252:
    roll = pd.Series(rets).rolling(252).apply(lambda x: (np.nanmean(x)*252)/(np.nanstd(x)*math.sqrt(252)) if np.nanstd(x)>1e-12 else np.nan)
    rolling_sharpe = float(np.nanmedian(roll.dropna())) if not roll.dropna().empty else np.nan
  
# For pruning we need per-window feature aggregates: recompute lightweight
feature_means = {}
possible_feats = ["ExpectedReturn","MedianReturn","WinRate","MatchQuality","Consistency","Liquidity","C_times_Median","WtimesE"]
for feat in possible_feats:
    vals = []
    for gkey, group in df_all.groupby("_DateKey"):
        if feat in group.columns:
            arr = group[feat].astype(float).values
            if arr.size:
                vals.append(np.nanmean(arr))
            else:
                vals.append(np.nan)
        else:
            vals.append(np.nan)
    feature_means[feat] = vals

feat_df = pd.DataFrame(feature_means)
feat_df["Date"] = [g[0] for g in groups]
# align lengths
if len(feat_df) == len(df_perf):
    combined = pd.concat([df_perf.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
else:
    # attempt to align by Date if possible
    try:
        feat_df["Date"] = pd.to_datetime(feat_df["Date"], errors="coerce")
        df_temp = df_perf.copy()
        df_temp["Date"] = pd.to_datetime(df_temp["Date"], errors="coerce")
        combined = pd.merge(df_temp, feat_df, on="Date", how="left")
    except Exception:
        combined = df_perf.copy()

# compute correlations (safely)
corr_suggestions = {}
if "RealizedAdj" in combined.columns:
    for feat in possible_feats:
        if feat in combined.columns:
            c = combined[feat].corr(combined["RealizedAdj"])
            if pd.notna(c):
                corr_suggestions[feat] = c

print("=== ADAPTIVE BACKTEST REPORT ===")
print(f"Windows used: {len(df_perf):,}")
print(f"Avg selected/window: {df_perf['NumSelected'].mean():.1f} | Avg universe: {df_perf['UniverseSize'].mean():.1f}")
print(f"Ann Return: {np.nan_to_num(ann_ret)*100:.2f}% | Sharpe: {np.nan_to_num(sharpe):.3f} | Sortino: {np.nan_to_num(sortino):.3f}")
print(f"Max Drawdown: {np.nan_to_num(max_dd)*100:.2f}% | Calmar: {np.nan_to_num(calmar):.3f}")
print(f"Win rate: {win_rate:.2f}% | Avg window ret: {avg_win:.4f}% | Std window ret: {std_win:.4f}%")
if not np.isnan(rolling_sharpe):
    print(f"Rolling 1y Sharpe (median): {rolling_sharpe:.3f}")

print("\n--- Top 6 windows by RealizedAdj (Date, RealizedAdj, CapUsed, NumSelected) ---")
print(df_perf.nlargest(6, "RealizedAdj")[["Date","RealizedAdj","CapUsed","NumSelected"]].to_string(index=False))
print("\n--- Bottom 6 windows by RealizedAdj ---")
print(df_perf.nsmallest(6, "RealizedAdj")[["Date","RealizedAdj","CapUsed","NumSelected"]].to_string(index=False))

print("\n--- Metric ↔ Profitability correlation (per-window) ---")
# print top correlations from corr_suggestions
sorted_corrs = sorted(corr_suggestions.items(), key=lambda x: x[1] if x[1] is not None else -999, reverse=True)
for k, v in sorted_corrs:
    print(f"{k:18s}: {v:+.4f}")

# auto-pruning suggestions: negative corr features are candidates to down-weight
neg_feats = [(k,v) for k,v in sorted_corrs if v is not None and v < 0]
if neg_feats:
    print("\n--- Suggested auto-pruning (negatively correlated) ---")
    for k, v in neg_feats:
        print(f"• {k}: corr={v:+.4f}  → consider reducing weight or removing")

# decile snapshot for RealizedAdj by CapUsed
try:
    df_perf["dec_Cap"] = pd.qcut(df_perf["CapUsed"].rank(method="first"), 10, labels=False, duplicates="drop")
    dec_cap = df_perf.groupby("dec_Cap")["RealizedAdj"].mean().dropna()
    print("\nCapUsed decile means (bottom3 ... top3) RealizedAdj%:", ", ".join(f"{x*100:.3f}" for x in list(dec_cap.head(3)) + list(dec_cap.tail(3))))
except Exception:
    pass

print(f"\nTotal runtime: {time.time()-t0:.1f}s")
print("=========================================\n")
