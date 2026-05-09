import os
import numpy as np
import pandas as pd
import itertools
import concurrent.futures
import multiprocessing as mp
from tqdm import tqdm

# Try optional accelerators
USE_NUMBA = False
try:
    from numba import njit
    USE_NUMBA = True
except Exception:
    USE_NUMBA = False

# Try to use a faster multiprocessing context on macOS for lower overhead (copy-on-write).
try:
    MP_CONTEXT = mp.get_context("fork")
except (AttributeError, ValueError):
    MP_CONTEXT = mp.get_context("spawn")

# PARAMETERS (tune these if desired)

DONCHIAN_WINDOWS = [2, 3] 
ATR_WINDOWS = [2, 3, 5, 8, 17, 24] 
ATR_MULTIPLIERS = [0.01] 
TREND_SMA_WINDOWS = [2, 6, 8, 10, 30] 
COOLDOWN_BARS_OPTIONS = [3] #

RISK_PER_TRADE = 0.02
INITIAL_BALANCE = 10000.0

IN_SAMPLE_DAYS = 365  
OUT_SAMPLE_DAYS = 91  
STEP_DAYS = OUT_SAMPLE_DAYS

FEE_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.0002
COST_PER_SIDE = FEE_PER_SIDE + SLIPPAGE_PER_SIDE

def load_csv(filename):
    # Try pyarrow engine if available; it is faster and lower memory than the default on M1.
    try:
        df = pd.read_csv(
            filename,
            engine="pyarrow",
            parse_dates=["datetime"]
        )
        return df
    except Exception:
        # Fall back to C engine
        return pd.read_csv(
            filename,
            parse_dates=["datetime"],
            engine="c",
            low_memory=False
        )

df = load_csv("/Users/SaiSanjayD/Documents/PythonPrograms/BTCUSDT_BINANCEUS_1H/BTCUSDT_1H.csv")  # 5 years of 1H data
df.sort_values("datetime", inplace=True)
df.reset_index(drop=True, inplace=True)


G_CLOSE = None
G_HIGH = None
G_LOW = None
G_DATETIMES = None

G_DC_HIGH_MAP = None 
G_DC_LOW_MAP = None  
G_ATR_MAP = None      
G_SMA_MAP = None       

def _set_globals(close, high, low, datetimes, dc_high_map, dc_low_map, atr_map, sma_map):
    global G_CLOSE, G_HIGH, G_LOW, G_DATETIMES, G_DC_HIGH_MAP, G_DC_LOW_MAP, G_ATR_MAP, G_SMA_MAP
    G_CLOSE = close
    G_HIGH = high
    G_LOW = low
    G_DATETIMES = datetimes
    G_DC_HIGH_MAP = dc_high_map
    G_DC_LOW_MAP = dc_low_map
    G_ATR_MAP = atr_map
    G_SMA_MAP = sma_map

def _worker_initializer(close, high, low, datetimes, dc_high_map, dc_low_map, atr_map, sma_map):
    """
    Initializer for ProcessPool workers: sets global references once.
    With 'fork', this is quick; with 'spawn', this avoids sending full data on every task.
    """
    _set_globals(close, high, low, datetimes, dc_high_map, dc_low_map, atr_map, sma_map)

# ------------- Rolling helpers (precompute once per window) -------------

def _rolling_max(a: np.ndarray, window: int) -> np.ndarray:
    # pandas is already optimized in C; used only ~6 times over full series.
    return pd.Series(a).rolling(window, min_periods=window).max().to_numpy()

def _rolling_min(a: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(a).rolling(window, min_periods=window).min().to_numpy()

def _rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(a).rolling(window, min_periods=window).mean().to_numpy()

def precompute_all_windows(df: pd.DataFrame,
                           dc_windows,
                           atr_windows,
                           sma_windows):
    """
    Precompute all indicator arrays for every window across the FULL dataset.
    This avoids recomputing indicators for each parameter combo and fold.
    """
    high = df["high"].to_numpy(dtype=np.float64, copy=False)
    low = df["low"].to_numpy(dtype=np.float64, copy=False)
    close = df["close"].to_numpy(dtype=np.float64, copy=False)
    datetimes = df["datetime"].to_numpy(copy=False)

    dc_high_map = {}
    dc_low_map = {}
    for w in dc_windows:
        dc_high_map[w] = _rolling_max(high, w)
        dc_low_map[w] = _rolling_min(low, w)

    # True range components (vectorized once)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum.reduce([tr1, tr2, tr3])

    atr_map = {}
    for w in atr_windows:
        atr_map[w] = _rolling_mean(tr, w)

    sma_map = {}
    for w in sma_windows:
        raw = _rolling_mean(close, w)
        sma_map[w] = np.roll(raw, 1)
        sma_map[w][0] = np.nan

    return close, high, low, datetimes, dc_high_map, dc_low_map, atr_map, sma_map

# Metrics

def annualize_return(ret, periods):
    years = periods / (365 * 24)
    return (1 + ret) ** (1 / years) - 1 if years > 0 else np.nan

def annualize_volatility(returns, periods):
    return np.std(returns) * np.sqrt(24 * 365)

def sharpe_ratio(returns):
    ann_return = np.mean(returns) * 24 * 365
    ann_vol = np.std(returns) * np.sqrt(24 * 365)
    return ann_return / ann_vol if ann_vol > 0 else np.nan

def sortino_ratio(returns):
    neg = returns[returns < 0]
    if neg.size == 0:
        return np.nan
    neg_vol = np.std(neg) * np.sqrt(24 * 365)
    ann_return = np.mean(returns) * 24 * 365
    return ann_return / neg_vol if neg_vol > 0 else np.nan

def max_drawdown(equity):
    roll_max = np.maximum.accumulate(equity)
    drawdowns = (equity - roll_max) / roll_max
    return drawdowns.min() * 100

def calmar_ratio(ann_ret, max_dd):
    return ann_ret / abs(max_dd / 100) if max_dd < 0 else np.nan

def profit_factor(trades):
    pnl = np.array([t["pnl"] for t in trades if "pnl" in t], dtype=np.float64)
    profits = pnl[pnl > 0.0]
    losses = -pnl[pnl < 0.0]
    return np.sum(profits) / np.sum(losses) if np.sum(losses) > 0 else np.nan

def win_rate(trades):
    pnl = np.array([t["pnl"] for t in trades if "pnl" in t], dtype=np.float64)
    wins = np.sum(pnl > 0.0)
    return 100.0 * wins / len(pnl) if len(pnl) > 0 else np.nan

def avg_trade_return(trades, initial_balance):
    pnl = np.array([t["pnl"] for t in trades if "pnl" in t], dtype=np.float64)
    if pnl.size == 0:
        return np.nan
    returns = pnl / initial_balance
    return np.mean(returns) * 100.0

def trade_frequency(trades, duration_yrs):
    return len(trades) / duration_yrs if duration_yrs > 0 else np.nan

def alpha_beta(strategy_returns, bh_returns):
    min_len = min(len(strategy_returns), len(bh_returns))
    strategy_returns = strategy_returns[-min_len:]
    bh_returns = bh_returns[-min_len:]
    if np.var(bh_returns) == 0:
        return np.nan, np.nan
    beta = np.cov(strategy_returns, bh_returns)[0, 1] / np.var(bh_returns)
    alpha = strategy_returns.mean() - beta * bh_returns.mean()
    return alpha * 24 * 365, beta

def ulcer_index(equity):
    roll_max = np.maximum.accumulate(equity)
    drawdowns = (equity - roll_max) / roll_max * 100.0
    return np.sqrt(np.mean(drawdowns ** 2))

def longest_drawdown_duration(equity, datetimes):
    roll_max = np.maximum.accumulate(equity)
    drawdowns = (equity - roll_max) / roll_max
    in_drawdown = drawdowns < 0
    if not np.any(in_drawdown):
        return 0.0
    max_duration = 0.0
    cur_start = None
    for idx, dd in enumerate(in_drawdown):
        if dd:
            if cur_start is None:
                cur_start = datetimes[idx]
        else:
            if cur_start is not None:
                end_dt = datetimes[idx - 1]
                duration = (end_dt - cur_start) / np.timedelta64(1, "D")
                max_duration = max(max_duration, float(duration))
                cur_start = None
    if cur_start is not None:
        end_dt = datetimes[-1]
        duration = (end_dt - cur_start) / np.timedelta64(1, "D")
        max_duration = max(max_duration, float(duration))
    return max_duration

# Fast training backtest, long only

def _backtest_equity_only_py(close, dc_high, dc_low, atr, sma,
                             balance_start, dc_window, atr_window, atr_mult, tsma_window,
                             risk_per_trade, cooldown_bars, cost_per_side):
    n = close.shape[0]
    balance = balance_start
    eq_curve = np.empty(n, dtype=np.float64)

    position = 0  # 0 flat, 1 long
    entry_price = 0.0
    stop_price = 0.0
    position_size = 0.0
    last_trade_bar = -cooldown_bars

    current_equity = balance

    min_w = max(dc_window, atr_window, tsma_window)
    for i in range(n):
        if i < min_w:
            eq_curve[i] = current_equity
            continue

        # Enforce cooldown
        if (i - last_trade_bar) < cooldown_bars:
            # mark-to-market if in position
            if position == 1:
                eq_curve[i] = balance + (close[i] - entry_price) * position_size
            else:
                eq_curve[i] = balance
            current_equity = eq_curve[i]
            continue

        # Signals use previous DC values
        long_signal = (close[i] > dc_high[i - 1]) and (close[i] > sma[i])

        if position == 0:
            if long_signal:
                position = 1
                entry_price = close[i] * (1.0 + cost_per_side)
                stop_price = entry_price - atr_mult * atr[i]
                risk_amt = balance * risk_per_trade
                stop_dist = entry_price - stop_price
                position_size = min(risk_amt / stop_dist if stop_dist > 0 else 0.0,
                                    balance / entry_price)
                last_trade_bar = i
            eq_curve[i] = balance
            current_equity = balance
            continue

        if position == 1:
            # trailing stop
            ts = close[i] - atr_mult * atr[i]
            if ts > stop_price:
                stop_price = ts
            if close[i] <= stop_price:
                exit_price = stop_price * (1.0 - cost_per_side)
                pnl = (exit_price - entry_price) * position_size
                balance += pnl
                position = 0
                last_trade_bar = i
                eq_curve[i] = balance
                current_equity = balance
                continue
            else:
                eq_curve[i] = balance + (close[i] - entry_price) * position_size
                current_equity = eq_curve[i]

    return eq_curve

if USE_NUMBA:
    @njit(cache=True, fastmath=True)
    def _backtest_equity_only_numba(close, dc_high, dc_low, atr, sma,
                                    balance_start, dc_window, atr_window, atr_mult, tsma_window,
                                    risk_per_trade, cooldown_bars, cost_per_side):
        n = close.shape[0]
        balance = balance_start
        eq_curve = np.empty(n, dtype=np.float64)

        position = 0
        entry_price = 0.0
        stop_price = 0.0
        position_size = 0.0
        last_trade_bar = -cooldown_bars

        current_equity = balance
        min_w = dc_window
        if atr_window > min_w:
            min_w = atr_window
        if tsma_window > min_w:
            min_w = tsma_window

        for i in range(n):
            if i < min_w:
                eq_curve[i] = current_equity
                continue

            if (i - last_trade_bar) < cooldown_bars:
                if position == 1:
                    eq_curve[i] = balance + (close[i] - entry_price) * position_size
                else:
                    eq_curve[i] = balance
                current_equity = eq_curve[i]
                continue

            long_signal = (close[i] > dc_high[i - 1]) and (close[i] > sma[i])

            if position == 0:
                if long_signal:
                    position = 1
                    entry_price = close[i] * (1.0 + cost_per_side)
                    stop_price = entry_price - atr_mult * atr[i]
                    risk_amt = balance * risk_per_trade
                    stop_dist = entry_price - stop_price
                    if stop_dist > 0.0:
                        tmp = risk_amt / stop_dist
                    else:
                        tmp = 0.0
                    cap = balance / entry_price
                    position_size = tmp if tmp < cap else cap
                    last_trade_bar = i
                eq_curve[i] = balance
                current_equity = balance
                continue

            if position == 1:
                ts = close[i] - atr_mult * atr[i]
                if ts > stop_price:
                    stop_price = ts
                if close[i] <= stop_price:
                    exit_price = stop_price * (1.0 - cost_per_side)
                    pnl = (exit_price - entry_price) * position_size
                    balance += pnl
                    position = 0
                    last_trade_bar = i
                    eq_curve[i] = balance
                    current_equity = balance
                    continue
                else:
                    eq_curve[i] = balance + (close[i] - entry_price) * position_size
                    current_equity = eq_curve[i]
        return eq_curve

def backtest_equity_only(close, dc_high, dc_low, atr, sma,
                         balance_start, dc_window, atr_window, atr_mult, tsma_window,
                         risk_per_trade, cooldown_bars, cost_per_side):
    if USE_NUMBA:
        return _backtest_equity_only_numba(close, dc_high, dc_low, atr, sma,
                                           balance_start, dc_window, atr_window, atr_mult, tsma_window,
                                           risk_per_trade, cooldown_bars, cost_per_side)
    else:
        return _backtest_equity_only_py(close, dc_high, dc_low, atr, sma,
                                        balance_start, dc_window, atr_window, atr_mult, tsma_window,
                                        risk_per_trade, cooldown_bars, cost_per_side)

# Detailed backtest for OOS trades

def backtest_segment(test_df, balance_start, dc_window, atr_window, atr_mult, trend_sma_window, cooldown_bars):
    n = len(test_df)
    balance = balance_start
    eq_curve = np.zeros(n, dtype=np.float64)
    trades = []
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    position_size = 0.0
    last_trade_bar = -cooldown_bars

    close = test_df["close"].to_numpy(dtype=np.float64, copy=False)
    DC_high = test_df["DC_high"].to_numpy(dtype=np.float64, copy=False)
    DC_low = test_df["DC_low"].to_numpy(dtype=np.float64, copy=False)
    ATR = test_df["ATR"].to_numpy(dtype=np.float64, copy=False)
    trend_sma = test_df["trend_sma"].to_numpy(dtype=np.float64, copy=False)
    datetimes = test_df["datetime"].to_numpy(copy=False)

    long_signal = (close > np.roll(DC_high, 1)) & (close > trend_sma)
    long_signal[:1] = False

    current_equity = balance
    min_w = max(dc_window, atr_window, trend_sma_window)

    for i in range(n):
        if i < min_w:
            eq_curve[i] = current_equity
            continue

        if (i - last_trade_bar) < cooldown_bars:
            if position == 1:
                eq_curve[i] = balance + (close[i] - entry_price) * position_size
                current_equity = eq_curve[i]
            else:
                eq_curve[i] = balance
                current_equity = eq_curve[i]
            continue

        if position == 0:
            if long_signal[i]:
                position = 1
                entry_price = close[i] * (1.0 + COST_PER_SIDE)
                stop_price = entry_price - atr_mult * ATR[i]
                risk_amt = balance * RISK_PER_TRADE
                stop_dist = entry_price - stop_price
                position_size = min(risk_amt / stop_dist if stop_dist > 0 else 0.0, balance / entry_price)
                trades.append({"side": "long", "entry_time": datetimes[i], "entry": entry_price, "size": position_size})
                last_trade_bar = i
            eq_curve[i] = balance
            current_equity = eq_curve[i]
            continue

        if position == 1:
            stop_price = max(stop_price, close[i] - atr_mult * ATR[i])
            if close[i] <= stop_price:
                exit_price = stop_price * (1.0 - COST_PER_SIDE)
                pnl = (exit_price - entry_price) * position_size
                balance += pnl
                trades[-1].update({"exit_time": datetimes[i], "exit": exit_price, "pnl": pnl})
                position = 0
                last_trade_bar = i
                eq_curve[i] = balance
                current_equity = eq_curve[i]
                continue
            else:
                eq_curve[i] = balance + (close[i] - entry_price) * position_size
                current_equity = eq_curve[i]

    return eq_curve, balance, trades

# Parallel parameter search (uses chunking + shared arrays)

def _eval_param_batch(batch_params, start_idx, end_idx, balance):
    """
    Worker function: evaluate a batch of (dcw, atrw, atrm, tsmaw, cdb) over [start_idx:end_idx)
    using precomputed global arrays. Returns list of (score, params).
    """
    results = []
    # Slices for this fold (views; no copy)
    c = G_CLOSE[start_idx:end_idx]
    for (dcw, atrw, atrm, tsmaw, cdb) in batch_params:
        dc_high = G_DC_HIGH_MAP[dcw][start_idx:end_idx]
        dc_low = G_DC_LOW_MAP[dcw][start_idx:end_idx]
        atr = G_ATR_MAP[atrw][start_idx:end_idx]
        sma = G_SMA_MAP[tsmaw][start_idx:end_idx]

        eq = backtest_equity_only(
            c, dc_high, dc_low, atr, sma,
            float(balance), int(dcw), int(atrw), float(atrm), int(tsmaw),
            float(RISK_PER_TRADE), int(cdb), float(COST_PER_SIDE)
        )
        if eq.shape[0] < 2 or np.all(eq == eq[0]):
            srt = -np.inf
        else:
            rets = np.diff(eq) / eq[:-1]
            # Compute Sortino quickly inline
            neg = rets[rets < 0.0]
            if neg.size == 0:
                srt = -np.inf
            else:
                neg_vol = np.std(neg) * np.sqrt(24.0 * 365.0)
                ann_ret = np.mean(rets) * 24.0 * 365.0
                srt = ann_ret / neg_vol if neg_vol > 0.0 else -np.inf
        results.append((srt, (dcw, atrw, atrm, tsmaw, cdb)))
    return results

def param_search(combos, in_start_idx, in_end_idx, balance, fold_idx=None, total_folds=None):
    max_workers = os.cpu_count() or 4
    # Chunk to amortize IPC overhead; tune as needed for your M1 (1000–4000 works well)
    chunk_size = 2000

    results = []
    desc = f"[Fold {fold_idx + 1}/{total_folds}] Param Search" if fold_idx is not None else "Param Search"
    with tqdm(total=len(combos), desc=desc, ncols=80, unit="param", leave=False) as inner_pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=MP_CONTEXT,
                                                    initializer=_worker_initializer,
                                                    initargs=(G_CLOSE, G_HIGH, G_LOW, G_DATETIMES,
                                                              G_DC_HIGH_MAP, G_DC_LOW_MAP, G_ATR_MAP, G_SMA_MAP)) as executor:
            futures = []
            for i in range(0, len(combos), chunk_size):
                batch = combos[i:i + chunk_size]
                futures.append(executor.submit(_eval_param_batch, batch, in_start_idx, in_end_idx, balance))
            for fut in concurrent.futures.as_completed(futures):
                batch_results = fut.result()
                results.extend(batch_results)
                inner_pbar.update(len(batch_results))
    return results

def slice_indices(datetimes: np.ndarray, start_dt: np.datetime64, end_dt: np.datetime64):
    """
    Return half-open [start_idx, end_idx) indices for datetimes sorted ascending.
    """
    # Ensure numpy datetime64 for searchsorted
    if not np.issubdtype(datetimes.dtype, np.datetime64):
        arr = datetimes.astype("datetime64[ns]")
    else:
        arr = datetimes
    start_idx = np.searchsorted(arr, start_dt, side="left")
    end_idx = np.searchsorted(arr, end_dt, side="left")
    return int(start_idx), int(end_idx)

if __name__ == "__main__":
    # Precompute all indicators once (huge speedup vs. per-combo/per-fold)
    close, high, low, datetimes, DC_HIGH_MAP, DC_LOW_MAP, ATR_MAP, SMA_MAP = precompute_all_windows(
        df, DONCHIAN_WINDOWS, ATR_WINDOWS, TREND_SMA_WINDOWS
    )
    _set_globals(close, high, low, datetimes, DC_HIGH_MAP, DC_LOW_MAP, ATR_MAP, SMA_MAP)

    start_date = df["datetime"].iloc[0]
    end_date = df["datetime"].iloc[-1]

    windows = []
    curr_in_start = start_date

    while True:
        curr_in_end = curr_in_start + pd.Timedelta(days=IN_SAMPLE_DAYS)
        curr_out_start = curr_in_end
        curr_out_end = curr_out_start + pd.Timedelta(days=OUT_SAMPLE_DAYS)
        if curr_out_end > end_date:
            break
        windows.append((curr_in_start, curr_in_end, curr_out_start, curr_out_end))
        curr_in_start = curr_in_start + pd.Timedelta(days=STEP_DAYS)

    print(f"Performing adaptive walk-forward backtest with {len(windows)} windows...")
    print("------------------------")

    walk_equity = []
    walk_dates = []
    walk_trades = []
    balance = float(INITIAL_BALANCE)
    best_params = []

    combos = list(itertools.product(DONCHIAN_WINDOWS, ATR_WINDOWS, ATR_MULTIPLIERS, TREND_SMA_WINDOWS, COOLDOWN_BARS_OPTIONS))

    with tqdm(total=len(windows), desc="Walk-forward folds", ncols=80, unit="fold", leave=True) as outer_pbar:
        for idx, (in_start, in_end, out_start, out_end) in enumerate(windows):
            # Compute index slices once (faster than boolean masks)
            in_s, in_e = slice_indices(G_DATETIMES, np.datetime64(in_start.to_datetime64()),
                                       np.datetime64(in_end.to_datetime64()))
            out_s, out_e = slice_indices(G_DATETIMES, np.datetime64(out_start.to_datetime64()),
                                         np.datetime64(out_end.to_datetime64()))

            # Skip if no OOS data
            if out_e <= out_s:
                outer_pbar.update(1)
                continue

            # Parallel parameter search over IN-SAMPLE using precomputed arrays
            results = param_search(combos, in_s, in_e, balance, fold_idx=idx, total_folds=len(windows))
            best_result = max(results, key=lambda x: x[0]) if results else (-np.inf, combos[0])
            _, best_setting = best_result
            best_params.append(best_setting)

            print(f"\nFOLD {idx + 1}:")
            dcw, atrw, atrm, tsmaw, cdb = best_setting

            # Build test features from precomputed arrays (no recompute)
            # Use views for speed; then wrap into a DataFrame expected by backtest_segment
            test_df = pd.DataFrame({
                "datetime": G_DATETIMES[out_s:out_e],
                "close": G_CLOSE[out_s:out_e],
                "DC_high": G_DC_HIGH_MAP[dcw][out_s:out_e],
                "DC_low": G_DC_LOW_MAP[dcw][out_s:out_e],
                "ATR": G_ATR_MAP[atrw][out_s:out_e],
                "trend_sma": G_SMA_MAP[tsmaw][out_s:out_e],
            })

            # OOS detailed backtest (keeps trades for final metrics)
            eq_curve, balance, trades = backtest_segment(test_df, balance, dcw, atrw, atrm, tsmaw, cdb)
            walk_equity.extend(eq_curve.tolist())
            walk_dates.extend(test_df["datetime"].tolist())
            walk_trades.extend(trades)

            eq_arr = np.asarray(eq_curve, dtype=np.float64)
            if eq_arr.size > 1 and not np.all(eq_arr == eq_arr[0]):
                total_return = (eq_arr[-1] / eq_arr[0]) - 1.0
                periods = eq_arr.size
                ann_ret = annualize_return(total_return, periods)
                mdd = max_drawdown(eq_arr)
                returns = np.diff(eq_arr) / eq_arr[:-1]
                shrp = sharpe_ratio(returns)
                srto = sortino_ratio(returns)
                calmar = calmar_ratio(ann_ret, mdd)
                print(f"Annualized Return: {ann_ret * 100:.2f}%")
                print(f"Total Return: {total_return:.2f}%")
                print(f"Max Drawdown: {mdd:.2f}%")
                print(f"Sharpe Ratio: {shrp:.2f}")
                print(f"Sortino Ratio: {srto:.2f}")
                print(f"Calmar Ratio: {calmar:.2f}")
                print("------------------------")
            else:
                print(f"(No trades were made)")
                print("------------------------")

            outer_pbar.update(1)

    # Buy & Hold baseline
    bh_init = df["close"].iloc[0]
    bh_curve = df["close"] / bh_init * INITIAL_BALANCE

    walk_equity = np.asarray(walk_equity, dtype=np.float64)
    walk_dates = pd.to_datetime(walk_dates)
    equity_df = pd.DataFrame({"datetime": walk_dates, "strategy": walk_equity})

    # Align B&H to strategy dates robustly using searchsorted (avoid datetime -> int casting and np.interp issues)
    # For each strategy datetime, pick the most recent buy&hold value at or before that timestamp.
    df_datetimes = df["datetime"].to_numpy(copy=False)
    bh_values = bh_curve.to_numpy(dtype=np.float64, copy=False)
    strat_datetimes = equity_df["datetime"].to_numpy(copy=False)

    # searchsorted returns index where to insert to keep order; we want the previous index (right side - 1)
    idxs = np.searchsorted(df_datetimes, strat_datetimes, side="right") - 1
    idxs = np.clip(idxs, 0, len(bh_values) - 1)
    bh_aligned = bh_values[idxs]

    equity_df["buy_and_hold"] = bh_aligned

    strategy_equity = equity_df["strategy"].to_numpy(dtype=np.float64, copy=False)
    bh_equity = equity_df["buy_and_hold"].to_numpy(dtype=np.float64, copy=False)
    datetimes = equity_df["datetime"].to_numpy(copy=False)

    strategy_returns = np.diff(strategy_equity) / strategy_equity[:-1]
    bh_returns = np.diff(bh_equity) / bh_equity[:-1]
    periods = strategy_equity.size

    strategy_total_return = (strategy_equity[-1] / strategy_equity[0]) - 1.0
    bh_total_return = (bh_equity[-1] / bh_equity[0]) - 1.0

    strategy_ann_return = annualize_return(strategy_total_return, periods)
    bh_ann_return = annualize_return(bh_total_return, periods)

    strategy_vol = annualize_volatility(strategy_returns, periods) * 100.0
    bh_vol = annualize_volatility(bh_returns, periods) * 100.0

    strategy_sharpe = sharpe_ratio(strategy_returns)
    bh_sharpe = sharpe_ratio(bh_returns)

    strategy_sortino = sortino_ratio(strategy_returns)
    bh_sortino = sortino_ratio(bh_returns)

    strategy_max_dd = max_drawdown(strategy_equity)
    bh_max_dd = max_drawdown(bh_equity)

    strategy_calmar = calmar_ratio(strategy_ann_return, strategy_max_dd)
    bh_calmar = calmar_ratio(bh_ann_return, bh_max_dd)

    strategy_ulcer = ulcer_index(strategy_equity)
    bh_ulcer = ulcer_index(bh_equity)

    strategy_longest_dd = longest_drawdown_duration(strategy_equity, datetimes)
    bh_longest_dd = longest_drawdown_duration(bh_equity, datetimes)

    strategy_profit_factor = profit_factor(walk_trades)
    strategy_win_rate = win_rate(walk_trades)
    strategy_avg_trade_return = avg_trade_return(walk_trades, INITIAL_BALANCE)

    years = (datetimes[-1] - datetimes[0]) / np.timedelta64(1, "D") / 365.25
    strategy_trade_freq = trade_frequency(walk_trades, float(years))

    strategy_alpha, strategy_beta = alpha_beta(strategy_returns, bh_returns)

    print("\nOptimal Parameter Summary (Walk-Forward):")
    print("-" * 80)
    print(f"{'Fold':7} {'DONCHIAN':>12} {'ATR_WIN':>12} {'ATR_MULT':>12} {'SMA_WIN':>12} {'COOLDOWN':>12}")
    print("-" * 80)
    for idx, best_setting in enumerate(best_params):
        print(f"{idx + 1:7} {best_setting[0]:12g} {best_setting[1]:12g} {best_setting[2]:12g} {best_setting[3]:12g} {best_setting[4]:12g}")
    print("-" * 80)

    print("\nFinancial Performance Metrics (Walk-Forward OOS):")
    print("-" * 70)
    print(f"{'Metric':35} {'Strategy':>12} {'Buy & Hold':>12}")
    print("-" * 70)
    print(f"{'Total Return (%)':35} {strategy_total_return * 100:12.2f} {bh_total_return * 100:12.2f}")
    print(f"{'Annualized Return (%)':35} {strategy_ann_return * 100:12.2f} {bh_ann_return * 100:12.2f}")
    print(f"{'Max Drawdown (%)':35} {strategy_max_dd:12.2f} {bh_max_dd:12.2f}")
    print(f"{'Annual Volatility (%)':35} {strategy_vol:12.2f} {bh_vol:12.2f}")
    print(f"{'Sharpe Ratio':35} {strategy_sharpe:12.2f} {bh_sharpe:12.2f}")
    print(f"{'Sortino Ratio':35} {strategy_sortino:12.2f} {bh_sortino:12.2f}")
    print(f"{'Calmar Ratio':35} {strategy_calmar:12.2f} {bh_calmar:12.2f}")
    print(f"{'Win Rate (%)':35} {strategy_win_rate:12.2f} {'N/A':>12}")
    print(f"{'Profit Factor':35} {strategy_profit_factor:12.2f} {'N/A':>12}")
    print(f"{'Trade Frequency (/yr)':35} {strategy_trade_freq:12.2f} {'N/A':>12}")
    print(f"{'Avg. Trade Return (%)':35} {strategy_avg_trade_return:12.2f} {'N/A':>12}")
    print(f"{'Longest Drawdown (days)':35} {strategy_longest_dd:12.2f} {bh_longest_dd:12.2f}")
    print(f"{'Ulcer Index':35} {strategy_ulcer:12.2f} {bh_ulcer:12.2f}")
    print(f"{'Alpha':35} {strategy_alpha:12.2f} {'0.00':>12}")
    print(f"{'Beta':35} {strategy_beta:12.2f} {'1.00':>12}")
    print("-" * 70)

    # Optional plotting
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(equity_df["datetime"], equity_df["strategy"], label="Strategy")
        plt.plot(equity_df["datetime"], equity_df["buy_and_hold"], label="Buy & Hold", alpha=0.8)
        plt.xlabel("Date")
        plt.ylabel("Equity (USDT)")
        plt.title("Equity Curve: Strategy vs. Buy & Hold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception:
        print("matplotlib not installed, skipping plots.")
