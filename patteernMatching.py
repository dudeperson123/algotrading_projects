import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
from datetime import datetime, timedelta
from bisect import bisect_left, bisect_right
import heapq
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

csv_folder = "Nasdaq_daily_data"
results_folder = "PatternMatchingResults"
os.makedirs(results_folder, exist_ok=True)

OHLCV_WEIGHTS = {
    "Open": 1.0,
    "High": 1.0,
    "Low": 1.0,
    "Close": 3.0,
    "Volume": 2.0
}

FEATURES = ["Open", "High", "Low", "Close", "Volume"]
_FEATURE_INDEX = {f: i for i, f in enumerate(FEATURES)}
_WEIGHTS_ARR = np.array([OHLCV_WEIGHTS.get(f, 1.0) for f in FEATURES], dtype=np.float64)

initial_start_date = datetime(2010, 3, 3)  # inclusive
initial_end_date = datetime(2010, 3, 9)    # inclusive

TOP_MATCHES = 5  # Change this number to set how many top matches you want

_EPS = 1e-8


_CACHE = {}

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False


def pct_change_array(arr):

    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if arr.size < 2:
        return out
    prev = arr[:-1]
    curr = arr[1:]
    valid = (prev != 0.0) & ~np.isnan(prev) & ~np.isnan(curr)
    pc = np.full(prev.shape, np.nan, dtype=np.float64)
    pc[valid] = (curr[valid] - prev[valid]) / prev[valid] * 100.0
    out[1:] = pc
    return out


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def fast_parse_date_only(date_str):
    if not date_str:
        return None
    try:
        # slice first 10 chars (YYYY-MM-DD)
        return datetime.fromisoformat(date_str[:10]).date()
    except Exception:
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        except Exception:
            return None


def get_trading_dates(csv_file_path):
    dates = []
    try:
        with open(csv_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if not header:
                return []
            try:
                date_idx = header.index('Date')
            except ValueError:
                return []
            for row in reader:
                if not row or date_idx >= len(row):
                    continue
                d = fast_parse_date_only(row[date_idx])
                if d is not None:
                    dates.append(d)
    except Exception:
        return []
    return sorted(set(dates))


def load_symbol_cached(filepath):
    if filepath in _CACHE:
        return _CACHE[filepath]

    date_objs = []
    date_strs = []
    cols = {f: [] for f in FEATURES}
    close_vals = []
    vol_vals = []
    try:
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if not header:
                raise ValueError("Empty CSV")
            # Build column index map once
            try:
                date_idx = header.index('Date')
            except ValueError:
                raise ValueError("No Date column")
            idx_map = {}
            for f in FEATURES:
                try:
                    idx_map[f] = header.index(f)
                except ValueError:
                    idx_map[f] = None  

            for row in reader:
                if not row:
                    continue
                if date_idx is None or date_idx >= len(row):
                    continue
                ds = row[date_idx]
                d = fast_parse_date_only(ds)
                if d is None:
                    continue
                date_objs.append(d)
                date_strs.append(ds)
                # Fill feature columns
                for f in FEATURES:
                    ci = idx_map[f]
                    val = safe_float(row[ci]) if (ci is not None and ci < len(row)) else np.nan
                    cols[f].append(val)
                close_vals.append(cols["Close"][-1])
                vol_vals.append(cols["Volume"][-1])
    except Exception:
        empty = {
            'dates': np.array([], dtype='datetime64[D]'),
            'date_objs': [],
            'date_strs': [],
            'values': {'Close': np.array([], dtype=np.float64), 'Volume': np.array([], dtype=np.float64)},
            'pcts': {f: np.array([], dtype=np.float64) for f in FEATURES},
            'pcts_stacked': np.empty((len(FEATURES), 0), dtype=np.float64),
        }
        _CACHE[filepath] = empty
        return empty

    # Convert to numpy arrays
    values_all = {f: np.array(cols[f], dtype=np.float64) for f in FEATURES}
    pcts = {f: pct_change_array(values_all[f]) for f in FEATURES}
    # Stack once per symbol for fast 2D slicing later (rows=features, cols=time)
    pcts_stacked = np.vstack([pcts[f] for f in FEATURES]).astype(np.float64, copy=False)

    data = {
        'dates': np.array(date_objs, dtype='datetime64[D]'),
        'date_objs': date_objs,
        'date_strs': date_strs,
        'values': {
            'Close': np.array(close_vals, dtype=np.float64),
            'Volume': np.array(vol_vals, dtype=np.float64),
        },
        'pcts': pcts,
        'pcts_stacked': pcts_stacked,
    }
    _CACHE[filepath] = data
    return data


def compute_weighted_diff_precomputed(ref_pcts, win_pcts, weights):
    total_error = 0.0
    total_norm = 0.0

    L = 0
    for f in FEATURES:
        if f in ref_pcts:
            L = len(ref_pcts[f])
            break

    if L == 0:
        for f in FEATURES:
            total_norm += weights.get(f, 1.0) * _EPS
        return 100.0, 0.0

    for f in FEATURES:
        rp = ref_pcts.get(f, np.array([], dtype=np.float64))
        wp = win_pcts.get(f, np.array([], dtype=np.float64))
        if rp.size != wp.size:
            minL = min(rp.size, wp.size)
            rp = rp[:minL]
            wp = wp[:minL]

        mask = ~np.isnan(rp) & ~np.isnan(wp)
        err = float(np.sum(np.abs(rp[mask] - wp[mask]))) if mask.any() else 0.0

        rmask = ~np.isnan(rp)
        norm = float(np.sum(np.abs(rp[rmask]))) if rmask.any() else 0.0

        w = weights.get(f, 1.0)
        total_error += w * err
        total_norm += w * (norm if norm != 0.0 else _EPS)

    if total_norm == 0.0 and total_error == 0.0:
        match_pct = 100.0
    else:
        match_pct = max(0.0, 100.0 - 100.0 * total_error / (total_norm if total_norm != 0.0 else _EPS))

    return match_pct, total_error


def compute_weighted_diff_precomputed_stacked(rp_stacked, wp_stacked, weights_arr):
    """
    rp_stacked, wp_stacked: 2D float64 arrays shaped (F, L)
    weights_arr: 1D float64 array of length F
    """
    F = rp_stacked.shape[0]
    L = rp_stacked.shape[1] if rp_stacked.ndim == 2 else 0

    if L == 0 or F == 0:
        total_norm = float(np.sum(weights_arr) * _EPS)
        return 100.0, 0.0

    total_error = 0.0
    total_norm = 0.0
    for fi in range(F):
        rp = rp_stacked[fi]
        wp = wp_stacked[fi]
        mask_both = ~np.isnan(rp) & ~np.isnan(wp)
        if mask_both.any():
            err = float(np.sum(np.abs(rp[mask_both] - wp[mask_both])))
        else:
            err = 0.0
        mask_r = ~np.isnan(rp)
        norm = float(np.sum(np.abs(rp[mask_r]))) if mask_r.any() else 0.0

        w = float(weights_arr[fi])
        total_error += w * err
        total_norm += w * (norm if norm != 0.0 else _EPS)

    if total_norm == 0.0 and total_error == 0.0:
        return 100.0, 0.0
    pct = 100.0 - 100.0 * total_error / (total_norm if total_norm != 0.0 else _EPS)
    if pct < 0.0:
        pct = 0.0
    return pct, total_error


reference_file_global = "INTC_1D.csv"  # or any symbol with the full date range
reference_path_global = os.path.join(csv_folder, reference_file_global)
_trading_dates_global = get_trading_dates(reference_path_global)
earliest_date = _trading_dates_global[0] if _trading_dates_global else None
latest_date = _trading_dates_global[-1] if _trading_dates_global else None
backtest_days = (initial_start_date.date() - earliest_date).days if earliest_date else 0

if _NUMBA_AVAILABLE:
    import math
    from numba import njit

    @njit(cache=True, fastmath=False)
    def _numba_total_error(stacked_rp, stacked_wp, weights_arr):
        # stacked_rp and stacked_wp are shaped (F, L) where F = len(FEATURES)
        F, L = stacked_rp.shape
        total_error = 0.0
        for fi in range(F):
            err = 0.0
            for j in range(L):
                r = stacked_rp[fi, j]
                if not np.isnan(r):
                    wv = stacked_wp[fi, j]
                    if not np.isnan(wv):
                        err += math.fabs(r - wv)
            total_error += weights_arr[fi] * err
        return total_error

    _USE_NUMBA = True
else:
    _USE_NUMBA = False
    _numba_total_error = None  # not used


def _precompute_total_norm(rp_stacked_c, weights_arr):
    """Compute the total_norm term once for a given reference window (matches legacy logic)."""
    total_norm = 0.0
    F = rp_stacked_c.shape[0]
    for fi in range(F):
        rp = rp_stacked_c[fi]
        mask_r = ~np.isnan(rp)
        if mask_r.any():
            norm = float(np.sum(np.abs(rp[mask_r])))
        else:
            norm = 0.0
        w = float(weights_arr[fi])
        total_norm += w * (norm if norm != 0.0 else _EPS)
    return total_norm


def _total_error_numpy(rp_stacked_c, wp_stacked, weights_arr):
    """Compute only the weighted total error between stacked reference and window (NumPy)."""
    total_error = 0.0
    F = rp_stacked_c.shape[0]
    for fi in range(F):
        rp = rp_stacked_c[fi]
        wp = wp_stacked[fi]
        mask_both = ~np.isnan(rp) & ~np.isnan(wp)
        if mask_both.any():
            total_error += float(weights_arr[fi] * np.sum(np.abs(rp[mask_both] - wp[mask_both])))
    return total_error


def process_symbol(args):
    """Worker function: compute metrics for a single symbol for the given reference window."""
    filepath, symbol, start_date, end_date, window_size, eval_period, _unused = args

    data = load_symbol_cached(filepath)
    dates_np = data['dates']
    date_strs = data['date_strs']
    values_close = data['values']['Close']
    values_vol = data['values']['Volume']
    pcts_stacked = data['pcts_stacked']

    n = dates_np.size
    if n == 0:
        return None

    # Convert once for fast comparisons
    start_date_np = np.datetime64(start_date)
    end_date_np = np.datetime64(end_date)

    ref_left = int(np.searchsorted(dates_np, start_date_np, side='left'))
    ref_right = int(np.searchsorted(dates_np, end_date_np, side='right')) - 1
    has_ref = (ref_left < n and ref_left <= ref_right)

    # Compute liquidity over the filtered range (Volume * Close)
    if has_ref:
        close_seg = values_close[ref_left:ref_right + 1]
        vol_seg = values_vol[ref_left:ref_right + 1]
        mask = ~np.isnan(close_seg) & ~np.isnan(vol_seg)
        if mask.any():
            avg_liquidity = float(np.mean(close_seg[mask] * vol_seg[mask]))
        else:
            avg_liquidity = np.nan
    else:
        avg_liquidity = np.nan

    # FilteredRangeStart/End as original strings
    filtered_range_start_str = date_strs[ref_left] if has_ref else ""
    filtered_range_end_str = date_strs[ref_right] if has_ref else ""

    filtered_return = None
    if has_ref and dates_np[ref_left] == start_date_np:
        filtered_start_idx = ref_left
        last_idx = filtered_start_idx + window_size - 1
        if last_idx < n:
            last_close = values_close[last_idx]
            if not np.isnan(last_close):
                next_start = last_idx + 1
                next_end_excl = min(n, next_start + eval_period)
                next_closes = values_close[next_start:next_end_excl]
                next_closes = next_closes[~np.isnan(next_closes)]
                if next_closes.size >= 1:
                    filtered_return = float((next_closes[-1] - last_close) / last_close * 100.0)

    # Build reference % change slice (stacked 2D view) once per symbol for this window
    if has_ref:
        L = ref_right - ref_left  # number of pct-change points in reference
        if L > 0:
            rp_stacked = pcts_stacked[:, ref_left + 1: ref_left + 1 + L]
            rp_stacked_c = np.ascontiguousarray(rp_stacked)
            total_norm_const = _precompute_total_norm(rp_stacked_c, _WEIGHTS_ARR)
        else:
            rp_stacked_c = np.empty((len(FEATURES), 0), dtype=np.float64)
            total_norm_const = float(np.sum(_WEIGHTS_ARR) * _EPS)
    else:
        L = 0
        rp_stacked_c = np.empty((len(FEATURES), 0), dtype=np.float64)
        total_norm_const = float(np.sum(_WEIGHTS_ARR) * _EPS)

    # Local bindings for speed
    dates_local = dates_np
    weights_arr = _WEIGHTS_ARR
    use_numba = _USE_NUMBA
    numba_err = _numba_total_error
    total_err_numpy = _total_error_numpy

    # Precompute inverse norm scalar for match % calculation
    denom = total_norm_const if total_norm_const != 0.0 else _EPS
    inv_norm_scale = 100.0 / denom

    # Loop candidate windows and pick top N by error (use max-heap of size TOP_MATCHES)
    heap = []  
    max_start = n - window_size
    if max_start < 0:
        top_matches = []
    else:
        for i in range(0, max_start + 1):
            # fast check: skip candidate windows that overlap the reference window (no leakage)
            if dates_local[i] <= end_date_np and dates_local[i + window_size - 1] >= start_date_np:
                continue

            if L > 0:
                wp_stacked = pcts_stacked[:, i + 1: i + 1 + L]  # 2D view
                if use_numba:
                    err = float(numba_err(rp_stacked_c, wp_stacked, weights_arr))
                else:
                    err = total_err_numpy(rp_stacked_c, wp_stacked, weights_arr)

                match_pct = 100.0 - (err * inv_norm_scale)
                if match_pct < 0.0:
                    match_pct = 0.0
            else:
                # Emulate original behavior for empty reference slice
                match_pct, err = 100.0, 0.0

            if len(heap) < TOP_MATCHES:
                heapq.heappush(heap, (-err, i, match_pct))
            else:
                if err < -heap[0][0]:
                    heapq.heapreplace(heap, (-err, i, match_pct))

        # order top matches by ascending error
        top_entries = sorted(heap, key=lambda t: (-t[0], t[1]))
        top_matches = [(start, match_pct, -neg_err) for (neg_err, start, match_pct) in top_entries]

    historical_outcomes = []
    match_percentages = []
    for (match_start, match_pct, _err) in top_matches:
        last_idx = match_start + window_size - 1
        if last_idx < n:
            match_last_close = values_close[last_idx]
        else:
            match_last_close = np.nan

        next_start = last_idx + 1
        next_end_excl = min(n, next_start + eval_period)
        next_closes = values_close[next_start:next_end_excl]
        next_closes = next_closes[~np.isnan(next_closes)]

        if np.isnan(match_last_close) or next_closes.size < 1:
            historical_outcomes.append(np.nan)
        else:
            ret = float((next_closes[-1] - match_last_close) / match_last_close * 100.0)
            historical_outcomes.append(ret)

        match_percentages.append(match_pct)

    # Aggregate statistics (same logic as before)
    valid_indices = [i for i, r in enumerate(historical_outcomes) if not np.isnan(r)]
    if valid_indices:
        valid_outcomes = [historical_outcomes[i] for i in valid_indices]
        valid_matches = [match_percentages[i] for i in valid_indices]
        weights = np.array(valid_matches, dtype=np.float64) / 100.0
        if np.sum(weights) > 0:
            expected_return = float(np.average(valid_outcomes, weights=weights))
        else:
            expected_return = float('nan')
        average_return = float(np.mean(valid_outcomes))
        median_return = float(np.median(valid_outcomes))
        avg_sign = np.sign(average_return)
        win_count = sum(np.sign(r) == avg_sign and avg_sign != 0 for r in valid_outcomes)
        win_rate = win_count / len(valid_outcomes) * 100.0
        match_quality = float(np.mean(valid_matches))
        stdev = float(np.std(valid_outcomes))
        consistency = max(0.0, 100.0 - (stdev / abs(average_return) * 100.0)) if average_return != 0 else 0.0
    else:
        expected_return = float('nan')
        average_return = float('nan')
        median_return = float('nan')
        win_rate = float('nan')
        match_quality = float('nan')
        consistency = float('nan')

    return {
        "Symbol": symbol,
        "FilteredRangeStart": filtered_range_start_str if has_ref else "",
        "FilteredRangeEnd": filtered_range_end_str if has_ref else "",
        "FilteredReturn": filtered_return if filtered_return is not None else "",
        "ExpectedReturn": expected_return,
        "AverageReturn": average_return,
        "MedianReturn": median_return,
        "WinRate": win_rate,
        "MatchQuality": match_quality,
        "Consistency": consistency,
        "Liquidity": avg_liquidity,
    }


def main():
    reference_file = reference_file_global

    if _trading_dates_global:
        trading_dates = _trading_dates_global
    else:
        trading_dates = get_trading_dates(os.path.join(csv_folder, reference_file))
    trading_dates = sorted(trading_dates)  # oldest → newest

    latest_end_date = initial_end_date.date()
    earliest_start_date = (initial_start_date - timedelta(days=backtest_days)).date() if backtest_days else trading_dates[0]

    reference_date_range = [d for d in trading_dates if earliest_start_date <= d <= latest_end_date]

    window_len = sum(1 for d in trading_dates if initial_start_date.date() <= d <= initial_end_date.date())

    series = []
    end_idx = len(reference_date_range) - 1
    while end_idx - (window_len - 1) >= 0:
        start_date = reference_date_range[end_idx - (window_len - 1)]
        end_date = reference_date_range[end_idx]
        series.append((start_date, end_date))
        end_idx -= 1  # step backward by one trading day

    symbols = []
    for f in os.listdir(csv_folder):
        if f.endswith(".csv"):
            fp = os.path.join(csv_folder, f)
            symbols.append((fp, f.split("_")[0]))

    env_workers = os.getenv("PM_NUM_WORKERS")
    if env_workers and env_workers.isdigit() and int(env_workers) > 0:
        n_workers = min(int(env_workers), len(symbols))
    else:
        n_workers = min(max(1, cpu_count() // 2), max(1, len(symbols)))

    chunksize = 16  # adjust if needed; larger = less IPC overhead

    with Pool(n_workers) as pool:
        for start_date, end_date in tqdm(series, desc="Windows"):
            output_file = os.path.join(
                results_folder,
                f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
            )

            # Use calendar-day-based length for consistency with evaluation period
            filtered_window_len = (end_date - start_date).days + 1
            window_size = filtered_window_len
            eval_period = filtered_window_len

            # Arguments per symbol
            args_list = [(fp, sym, start_date, end_date, window_size, eval_period, None)
                         for (fp, sym) in symbols]

            results = []
            for result in tqdm(pool.imap_unordered(process_symbol, args_list, chunksize=chunksize),
                               total=len(symbols), desc="Symbols", leave=False):
                if result is not None:
                    results.append(result)

            if results:
                # keep deterministic column order
                fieldnames = list(results[0].keys())
                with open(output_file, "w", newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)


if __name__ == "__main__":
    main()
