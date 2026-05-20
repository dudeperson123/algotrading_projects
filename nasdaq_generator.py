import yfinance as yf
import pandas as pd
import os
from tqdm import tqdm

file_path = "nasdaqlisted.txt"
output_dir = "Nasdaq_daily_data"

def fetch_symbol_history(symbol):
    df = yf.download(
        tickers=symbol,
        period="max",
        interval="1d",
        auto_adjust=False,
        actions=True
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # yfinance often returns tz-naive dates, so we assume UTC then convert
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    df = df.reset_index()

    df = df.rename(columns={df.columns[0]: "Date"})

    # Keep only needed columns
    df = df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Dividends",
            "Stock Splits"
        ]
    ]

    output_file = os.path.join(output_dir, f"{symbol}_1D.csv")
    df.to_csv(output_file, index=False)

    print(f"Saved {len(df)} rows to {output_file}")

with open(file_path, "r", encoding="utf-8") as f:
    # Read header
    header = f.readline().strip().split("|")
    total_lines = sum(1 for _ in f) - 1

    # rewind
    f.seek(0)
    f.readline()

    symbol_index = header.index("Symbol")

    for line in tqdm(f, desc="Processing", total=total_lines):
        if not line.strip():
            continue  # skip empty lines

        parts = line.strip().split("|")

        # Safety check in case of malformed rows
        if len(parts) <= symbol_index:
            continue

        symbol = parts[symbol_index]

        try:
            fetch_symbol_history(symbol)
        except KeyError:
            continue
