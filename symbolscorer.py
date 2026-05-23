import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import concurrent.futures
from datetime import date, timedelta
import matplotlib.pyplot as plt

folder = 'Nasdaq_daily_data'
results_folder = 'SoldBoughtResults'

score_thresh = 7         
liq_thresh = 75_000_000  
rsi_thresh = 20        
stoch_thresh = 10     
mfi_thresh = 10        
RSI_PERIOD = 14
STOCH_K = 14
STOCH_D = 3
SMOOTH = 5               
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MFI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.5             

starting_equity = 10000 
os.makedirs(results_folder, exist_ok=True)


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_stochastic(df, k_period=14, k_smooth=3, d_period=3):
    low_min = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    raw_k = 100 * ((df['Close'] - low_min) / denom)
    smooth_k = raw_k.rolling(k_smooth).mean()
    d = smooth_k.rolling(d_period).mean()
    return smooth_k, d


def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def calculate_mfi(df, period=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    pos_sum = positive_flow.rolling(period).sum()
    neg_sum = negative_flow.rolling(period).sum().replace(0, np.nan)
    mf_ratio = pos_sum / neg_sum
    return 100 - (100 / (1 + mf_ratio))


def calculate_bollinger_bands(series, period=20, std_dev=2):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ma + std_dev * std, ma - std_dev * std


def get_trading_days_from_symbol_data(symbol_data, sample_filename="INTC_1D.csv"):
    if symbol_data and sample_filename in symbol_data:
        dates = sorted(symbol_data[sample_filename]['next_close'].keys())
        # convert to pandas.tfor consistent comparisons
        return [pd.Timestamp(d).normalize() for d in dates]
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No CSV files found")
    sample_file = os.path.join(folder, sample_filename)
    if not os.path.exists(sample_file):
        sample_file = os.path.join(folder, files[0])
    df = pd.read_csv(sample_file, parse_dates=['Date'], usecols=['Date'])
    trading_days = sorted(df['Date'].dt.normalize().unique())
    return list(trading_days)


def preload_symbol(filename):
    file_path = os.path.join(folder, filename)
    try:
        df = pd.read_csv(
            file_path,
            usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
            encoding='utf-8',
        )

        df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %H:%M:%S%z", utc=True)
        df['Date'] = df['Date'].dt.tz_convert(None)
        df.set_index('Date', inplace=True)
    except UnicodeDecodeError:
        df = pd.read_csv(
            file_path,
            parse_dates=['Date'],
            usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
            encoding='latin1',
        )

    # Ensure required columns and types (minimize copies)
    if 'Volume' not in df.columns:
        df['Volume'] = 0

    for col in ('Close', 'High', 'Low', 'Open'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['RSI'] = calculate_rsi(df['Close'], RSI_PERIOD)
    k, d = calculate_stochastic(df, STOCH_K, STOCH_D, SMOOTH)
    df['%K'] = k
    df['%D'] = d
    macd, macd_signal, macd_hist = calculate_macd(df['Close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df['MACD'] = macd
    df['MACD_SIGNAL'] = macd_signal
    df['MACD_HIST'] = macd_hist
    # Fill NaN volumes with zeros
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    df['MFI'] = calculate_mfi(df, MFI_PERIOD)
    bb_up, bb_low = calculate_bollinger_bands(df['Close'], BB_PERIOD, BB_STD)
    df['BB_UPPER'] = bb_up
    df['BB_LOWER'] = bb_low

    # Use a new column for normalized date (date objects) for grouping
    df['DateOnly'] = df.index.date

    k_prev = df['%K'].shift(1)
    d_prev = df['%D'].shift(1)
    hist_prev = df['MACD_HIST'].shift(1)

    score_series = (
        (df['RSI'] < rsi_thresh).astype(int)
        + (df['%K'] < stoch_thresh).astype(int)
        + (((k_prev < stoch_thresh) & (k_prev < d_prev) & (df['%K'] > df['%D'])).astype(int) * 2)
        + (((hist_prev < 0) & (df['MACD_HIST'] > 0)).astype(int) * 2)
        + ((df['Close'] <= df['BB_LOWER']).astype(int))
        + ((df['MFI'] < mfi_thresh).astype(int))
        - ((df['RSI'] > (100 - rsi_thresh)).astype(int))
        - ((df['%K'] > (100 - stoch_thresh)).astype(int))
        - (((k_prev > (100 - stoch_thresh)) & (k_prev > d_prev) & (df['%K'] < df['%D'])).astype(int) * 2)
        - (((hist_prev > 0) & (df['MACD_HIST'] < 0)).astype(int) * 2)
        - ((df['Close'] >= df['BB_UPPER']).astype(int))
        - ((df['MFI'] > (100 - mfi_thresh)).astype(int))
    ).astype(int)

    score_by_date_map = score_series.groupby(df['DateOnly']).last().astype(int).to_dict()

    # daily_last will give last row of each DateOnly
    daily_last = df.groupby('DateOnly').last()
    next_close_map = daily_last['Close'].apply(lambda x: round(x, 2)).to_dict()
    next_open_map = daily_last['Open'].to_dict()
    volume_by_date_map = daily_last['Volume'].to_dict()

    return filename, {
        'score_by_date': score_by_date_map,
        'next_close': next_close_map,
        'next_open': next_open_map,
    }


def process_symbol(args, symbol_data):
    filename, score_date, pred_date = args
    data = symbol_data.get(filename)
    if data is None:
        return None

    score_key = score_date.date()
    pred_key = pred_date.date()

    score = data['score_by_date'].get(score_key, None)

    pred_open = data['next_open'].get(pred_key, np.nan)
    pred_close = data['next_close'].get(pred_key, np.nan)

    try:
        if not np.isnan(pred_open) and not np.isnan(pred_close) and pred_open != 0:
            pct_change = round((pred_close - pred_open) / pred_open * 100, 2)
        else:
            pct_change = np.nan
    except Exception:
        pct_change = np.nan

    # Clean symbol name
    symbol = filename.replace("_1D.csv", "")

    return symbol, {
        'Score': score,
        'Next_Open': pred_open,
        'Next_Close': pred_close,
        'Pct_Change': pct_change
    }


def process_date(pred_date, trading_days, all_files, symbol_data, results_folder):
    trades = []

    # Find the most recent trading day before pred_date
    prior_days = [d for d in trading_days if d < pred_date]
    if not prior_days:
        return {'date': pred_date, 'trades': []}

    score_date = max(prior_days)
    score_key = score_date.date()
    pred_key = pred_date.date()

    candidates = []
    total_abs_score = 0.0

    # Local bindings for speed
    sd = symbol_data
    s_thresh = score_thresh

    for filename in all_files:
        data = sd.get(filename)
        if data is None:
            continue

        score = data['score_by_date'].get(score_key)
        if score is None or abs(score) < s_thresh:
            continue

        nc = data['next_close'].get(pred_key, np.nan)
        po = data['next_open'].get(pred_key, np.nan)

        # Fetch previous day's volume
        prev_volume = None
        if 'volume_by_date' in data:
            prev_volume = data['volume_by_date'].get(score_key, np.nan)
        else:
            prev_volume = np.nan

        # Liquidity filter
        if pd.notna(prev_volume):
            prev_close = data['next_close'].get(score_key, np.nan)
            if pd.notna(prev_close) and pd.notna(prev_volume):
                dollar_volume = prev_close * prev_volume
            else:
                dollar_volume = np.nan
            if dollar_volume < liq_thresh:
                continue

        try:
            pct_change = round((nc - po) / po * 100, 2)
        except Exception:
            pct_change = np.nan

        if pd.isna(pct_change):
            continue

        abs_score = abs(int(score))
        total_abs_score += abs_score

        symbol = filename.replace("_1D.csv", "")
        pos = 'long' if score > 0 else 'short'
        ret_decimal = pct_change / 100.0
        if pos == 'short':
            ret_decimal = -ret_decimal

        candidates.append({
            'Symbol': symbol,
            'Score': int(score),
            'AbsScore': abs_score,
            'Position': pos,
            'Next_Open': po,
            'Next_Close': nc,
            'Pct_Change': pct_change,
            'Return_Decimal': ret_decimal,
        })

    if not candidates or total_abs_score == 0:
        return {'date': pred_date, 'trades': []}

    # Weight trades by score 
    for c in candidates:
        c['Weight'] = float(c['AbsScore'] / total_abs_score)
        trades.append(c)

    return {'date': pred_date, 'trades': trades}



def performance_summary(equity_series, trades_list):
    start_equity = float(equity_series.iloc[0])
    end_equity = float(equity_series.iloc[-1])
    total_return = (end_equity / start_equity) - 1.0

    # daily returns
    daily_returns = equity_series.pct_change().dropna()
    days = daily_returns.shape[0]
    annual_factor = 252.0
    if days > 0:
        cumulative_return = (end_equity / start_equity)
        annualized_return = cumulative_return ** (annual_factor / days) - 1
        annualized_vol = daily_returns.std() * np.sqrt(annual_factor)
        sharpe = annualized_return / annualized_vol if annualized_vol != 0 else np.nan
    else:
        annualized_return = np.nan
        annualized_vol = np.nan
        sharpe = np.nan

    # Sortino ratio
    if days > 0:
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std()
        downside_annual = downside_std * np.sqrt(annual_factor) if not np.isnan(downside_std) else np.nan
        sortino = annualized_return / downside_annual if (downside_annual is not np.nan and downside_annual != 0) else np.nan
    else:
        sortino = np.nan

    # max drawdown
    cum_max = equity_series.cummax()
    drawdown = (equity_series - cum_max) / cum_max
    max_dd = drawdown.min()  # negative or zero

    # Calmar ratio: annualized_return / abs(max_drawdown), use absolute of max drawdown
    if not np.isnan(annualized_return) and max_dd < 0:
        calmar = annualized_return / abs(max_dd) if abs(max_dd) != 0 else np.nan
    else:
        calmar = np.nan

    # Trades stats
    flat_trades = [t for day in trades_list for t in day]
    num_trades = len(flat_trades)
    wins = sum(1 for t in flat_trades if t['Return_Decimal'] > 0)
    win_rate = (wins / num_trades) if num_trades > 0 else np.nan
    avg_trade_return = np.mean([t['Return_Decimal'] for t in flat_trades]) if num_trades > 0 else np.nan

    # Trade frequency: percent of days with at least one trade, and average trades per day
    total_days = len(trades_list)
    trade_days = sum(1 for day in trades_list if len(day) > 0)
    trade_freq_pct = (trade_days / total_days * 100) if total_days > 0 else np.nan
    avg_trades_per_day = (num_trades / total_days) if total_days > 0 else np.nan

    summary = {
        'Start Equity': start_equity,
        'End Equity': end_equity,
        'Total Return %': total_return * 100,
        'Annualized Return %': (annualized_return * 100) if not np.isnan(annualized_return) else np.nan,
        'Annualized Vol %': (annualized_vol * 100) if not np.isnan(annualized_vol) else np.nan,
        'Sharpe (ann)': sharpe,
        'Sortino (ann)': sortino,
        'Calmar Ratio': calmar,
        'Max Drawdown %': max_dd * 100,
        'Number of Trades': num_trades,
        'Win Rate %': win_rate * 100 if not np.isnan(win_rate) else np.nan,
        'Avg Trade Return %': avg_trade_return * 100 if not np.isnan(avg_trade_return) else np.nan,
        'Trade Frequency %': trade_freq_pct,
        'Avg Trades per Day': avg_trades_per_day,
    }
    return summary


def main():
    all_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in folder {folder}")

    symbol_data = {}

    cpu_count = os.cpu_count() or 4
    # Use cpu_count but leave one core free to not overstress system
    max_workers = max(1, min(8, cpu_count - 1))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(preload_symbol, all_files), total=len(all_files), desc="Preprocessing"))
        for filename, data in results:
            symbol_data[filename] = data

    trading_days = get_trading_days_from_symbol_data(symbol_data)
    pred_dates = trading_days

    # Create a list of args for each pred_date
    max_workers = max(1, min(8, cpu_count - 1))
    date_args = [(pred_date, trading_days, all_files, symbol_data, results_folder) for pred_date in pred_dates]
    day_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_date, *args) for args in date_args]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Backtesting"):
            try:
                res = fut.result()
            except Exception:
                # keep going if a date fails
                res = {'date': None, 'trades': []}
            day_results.append(res)

    # Sort day_results by date to ensure correct order
    day_results = [d for d in day_results if d['date'] is not None]
    day_results.sort(key=lambda x: x['date'])

    equities = []
    equity = float(starting_equity)
    equities_dates = []
    trades_list_for_perf = []

    for day in day_results:
        dt = day['date']
        trades = day['trades']
        trades_list_for_perf.append(trades)
        equities_dates.append(dt)
        if not trades:
            # no exposure that day: equity doesn't change
            equities.append(equity)
            continue

        daily_portfolio_return = 0.0
        for t in trades:
            daily_portfolio_return += t['Weight'] * t['Return_Decimal']

        equity = equity * (1.0 + daily_portfolio_return)
        equities.append(equity)

    if not equities_dates:
        print("No trading days produced any trades. Exiting.")
        return

    equity_series = pd.Series(data=equities, index=pd.to_datetime(equities_dates))
    equity_series.index.name = 'Date'

    # Performance summary
    summary = performance_summary(equity_series, trades_list_for_perf)

    # Print summary 
    print("\n--- Backtest Performance Summary ---")
    for k, v in summary.items():
        if isinstance(v, float) and (not np.isnan(v)):
            if 'Rate' in k or 'Return' in k or 'Vol' in k or 'Drawdown' in k or 'Frequency' in k:
                print(f"{k}: {v:.2f}")
            else:
                print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print("------------------------------------\n")

    # Plot equity curve
    plt.figure(figsize=(10, 6))
    plt.plot(equity_series.index, equity_series.values, label='Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.title('Equity Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
