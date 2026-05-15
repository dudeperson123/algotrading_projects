#!/usr/bin/env python3
"""
Prerequisites:
  pip install python-binance pandas numpy colorama

Environment Variables:
  BINANCE_API_KEY
  BINANCE_API_SECRET

DISCLAIMER: This code is for educational purposes only. Trading involves risk. Use at your own risk.
LIVE_MODE CONTROLS WHETHER THIS PROGRAM USES REAL MONEY OR NOT.
"""

import os
import time
import json
import math
import hmac
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from binance.client import Client
from binance.enums import (
    SIDE_BUY, SIDE_SELL,
    ORDER_TYPE_MARKET, ORDER_TYPE_STOP_LOSS_LIMIT,
    TIME_IN_FORCE_GTC
)
from binance.exceptions import BinanceAPIException, BinanceOrderException

try:
    import colorama
    colorama.init()
    COLORAMA_INSTALLED = True
except ImportError:
    COLORAMA_INSTALLED = False


SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = Client.KLINE_INTERVAL_1HOUR

DONCHIAN_WINDOW = 2
ATR_WINDOW = 3
ATR_MULTIPLIER = 0.01
SMA_WINDOW = 2
COOLDOWN_BARS = 3

RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))  
STATE_FILE = os.getenv("STATE_FILE", os.path.join(os.getcwd(), "state.json"))
LIVE_MODE = os.getenv("LIVE_MODE", "true").lower() == "true"  

STOP_LIMIT_OFFSET = float(os.getenv("STOP_LIMIT_OFFSET", "0.002")) 

# Retry/backoff
HTTP_RETRY = 3
HTTP_RETRY_SLEEP = 2.0  

class Colors:
    if os.name == "nt" and COLORAMA_INSTALLED:
        from colorama import Fore, Style
        GREEN = Fore.GREEN
        RED = Fore.RED
        RESET = Style.RESET_ALL
    else:
        GREEN = "\033[92m"
        RED = "\033[91m"
        RESET = "\033[0m"


def load_state(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # Default state
    return {
        "position": "FLAT",
        "entry_price": 0.0,
        "qty": 0.0,
        "stop_order_id": None,
        "last_trade_close_time": None,  # ms
        "last_processed_close_time": None,  # ms
        "symbol": SYMBOL,
        "last_exit_price": None,
        "last_closed_pl": None
    }

def save_state(path: str, state: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, path)


class SymbolFilters:
    def __init__(self, client: Client, symbol: str):
        info = client.get_symbol_info(symbol)
        if not info:
            raise RuntimeError(f"Symbol {symbol} not found on Binance.US")

        self.baseAsset = info["baseAsset"]
        self.quoteAsset = info["quoteAsset"]
        self.status = info["status"]

        self.price_tick = None
        self.min_price = None
        self.max_price = None

        self.step_size = None
        self.min_qty = None
        self.max_qty = None

        self.min_notional = None

        for f in info["filters"]:
            ftype = f["filterType"]
            if ftype == "PRICE_FILTER":
                self.price_tick = float(f["tickSize"])
                self.min_price = float(f["minPrice"])
                self.max_price = float(f["maxPrice"])
            elif ftype == "LOT_SIZE":
                self.step_size = float(f["stepSize"])
                self.min_qty = float(f["minQty"])
                self.max_qty = float(f["maxQty"])
            elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                self.min_notional = float(f.get("minNotional") or f.get("notional", 0.0))

        if self.price_tick is None or self.step_size is None:
            raise RuntimeError(f"Missing filters for {symbol}: {info['filters']}")

    def round_price(self, p: float) -> float:
        if self.price_tick <= 0:
            return p
        return math.floor(p / self.price_tick) * self.price_tick

    def round_qty(self, q: float) -> float:
        if self.step_size <= 0:
            return q
        return math.floor(q / self.step_size) * self.step_size

    def valid_notional(self, price: float, qty: float) -> bool:
        if self.min_notional is None or self.min_notional == 0.0:
            return True
        return (price * qty) >= self.min_notional

    def qty_precision(self) -> int:
        s = f"{self.step_size:.20f}".rstrip('0')
        if '.' in s:
            return len(s.split('.')[1])
        return 0

    def price_precision(self) -> int:
        s = f"{self.price_tick:.20f}".rstrip('0')
        if '.' in s:
            return len(s.split('.')[1])
        return 0


def with_retries(fn, *args, **kwargs):
    last_exc = None
    for _ in range(HTTP_RETRY):
        try:
            return fn(*args, **kwargs)
        except (BinanceAPIException, BinanceOrderException) as e:
            last_exc = e
            time.sleep(HTTP_RETRY_SLEEP)
        except Exception as e:
            last_exc = e
            time.sleep(HTTP_RETRY_SLEEP)
    if last_exc:
        raise last_exc


def fetch_klines(client: Client, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    raw = with_retries(client.get_klines, symbol=symbol, interval=interval, limit=limit)
    cols = [
        "open_time","open","high","low","close","volume","close_time",
        "qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    for col in ["open","high","low","close","volume","qav","taker_base_vol","taker_quote_vol"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    dc_high = df["high"].rolling(DONCHIAN_WINDOW, min_periods=DONCHIAN_WINDOW).max()
    dc_low = df["low"].rolling(DONCHIAN_WINDOW, min_periods=DONCHIAN_WINDOW).min()
    sma = df["close"].rolling(SMA_WINDOW, min_periods=SMA_WINDOW).mean()

    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean()

    out = df.copy()
    out["DC_high"] = dc_high
    out["DC_low"] = dc_low
    out["SMA"] = sma
    out["ATR"] = atr
    return out

def last_closed_signal(ind_df: pd.DataFrame) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Evaluate signal on the last CLOSED bar: long if close[-1] > DC_high[-2] and close[-1] > SMA[-1].
    Returns: (should_long, close_last, atr_last)
    """
    if len(ind_df) < max(DONCHIAN_WINDOW, ATR_WINDOW, SMA_WINDOW) + 2:
        return False, None, None
    # Use the last fully closed candle (the second-to-last row; last row is still forming)
    i = len(ind_df) - 2
    close_last = float(ind_df["close"].iloc[i])
    sma_last = float(ind_df["SMA"].iloc[i])
    dc_high_prev = float(ind_df["DC_high"].iloc[i - 1]) if not pd.isna(ind_df["DC_high"].iloc[i - 1]) else None
    atr_last = float(ind_df["ATR"].iloc[i]) if not pd.isna(ind_df["ATR"].iloc[i]) else None

    if dc_high_prev is None or atr_last is None or pd.isna(sma_last):
        return False, None, None

    long_signal = (close_last > dc_high_prev) and (close_last > sma_last)
    return long_signal, close_last, atr_last

def bars_since_time(df: pd.DataFrame, last_time_ms: Optional[int]) -> int:
    if last_time_ms is None:
        return 1_000_000  # effectively no cooldown
    last_t = pd.to_datetime(last_time_ms, unit="ms", utc=True)
    # Number of bars strictly after last_trade_close_time
    return int((df["close_time"] > last_t).sum())


def format_quantity(qty: float, step_size: float) -> str:
    decimals = max(0, -int(math.log10(step_size)))
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(qty)

def format_price(price: float, price_tick: float) -> str:
    decimals = max(0, -int(math.log10(price_tick)))
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(price)

def get_balances(client: Client, base_asset: str, quote_asset: str) -> Tuple[float, float]:
    acc = with_retries(client.get_account)
    base_free = 0.0
    quote_free = 0.0
    for b in acc["balances"]:
        if b["asset"] == base_asset:
            base_free = float(b["free"])
        elif b["asset"] == quote_asset:
            quote_free = float(b["free"])
    return base_free, quote_free

def place_market_buy(client: Client, symbol: str, qty: float, filters: SymbolFilters) -> Dict[str, Any]:
    qty_str = format_quantity(qty, filters.step_size)
    if not LIVE_MODE:
        print(f"[DRY-RUN] MARKET BUY {symbol} qty={qty_str}")
        return {"status": "FILLED", "executedQty": qty_str, "fills": [], "orderId": -1}
    order = with_retries(
        client.order_market_buy,
        symbol=symbol,
        quantity=qty_str
    )
    return order

def place_stop_loss_limit_sell(client: Client, symbol: str, qty: float, stop_price: float, limit_price: float, filters: SymbolFilters) -> Dict[str, Any]:
    qty_str = format_quantity(qty, filters.step_size)
    stop_price_str = format_price(stop_price, filters.price_tick)
    limit_price_str = format_price(limit_price, filters.price_tick)
    if not LIVE_MODE:
        print(f"[DRY-RUN] STOP_LOSS_LIMIT SELL {symbol} qty={qty_str} stopPrice={stop_price_str} price={limit_price_str}")
        return {"orderId": -2, "status": "NEW", "origQty": qty_str}
    order = with_retries(
        client.create_order,
        symbol=symbol,
        side=SIDE_SELL,
        type=ORDER_TYPE_STOP_LOSS_LIMIT,
        quantity=qty_str,
        price=limit_price_str,
        stopPrice=stop_price_str,
        timeInForce=TIME_IN_FORCE_GTC
    )
    return order

def cancel_order_safe(client: Client, symbol: str, order_id: int) -> None:
    if order_id is None:
        return
    try:
        if LIVE_MODE:
            with_retries(client.cancel_order, symbol=symbol, orderId=order_id)
        else:
            print(f"[DRY-RUN] Cancel order {order_id}")
    except BinanceAPIException as e:
        # If already canceled or filled, it's fine
        if e.code in (-2011, -2013):
            return
        raise

def get_order_status(client: Client, symbol: str, order_id: int) -> Optional[str]:
    try:
        od = with_retries(client.get_order, symbol=symbol, orderId=order_id)
        return od.get("status")
    except BinanceAPIException as e:
        # if not found, assume done
        if e.code in (-2011, -2013):
            return None
        raise


def compute_entry_and_stop(close_last: float, atr_last: float) -> Tuple[float, float]:
    entry_price = close_last  # we use market; actual fill may vary slightly
    stop_price = entry_price - ATR_MULTIPLIER * atr_last
    return entry_price, stop_price

def compute_position_size(quote_balance: float, entry_price: float, stop_price: float, filters: SymbolFilters) -> float:
    if quote_balance <= 0:
        return 0.0
    risk_amt = quote_balance * RISK_PER_TRADE
    stop_dist = max(entry_price - stop_price, 0.0)
    if stop_dist <= 0.0:
        return 0.0
    qty_risk_based = risk_amt / stop_dist
    qty_balance_cap = quote_balance / entry_price
    qty = min(qty_risk_based, qty_balance_cap)
    qty = filters.round_qty(qty)
    # Enforce minQty and minNotional
    if qty < (filters.min_qty or 0.0):
        return 0.0
    if not filters.valid_notional(entry_price, qty):
        # Try bump to meet minNotional if possible
        target_qty = max(qty, (filters.min_notional or 0.0) / max(entry_price, 1e-9))
        qty = filters.round_qty(target_qty)
        if not filters.valid_notional(entry_price, qty):
            return 0.0
    # Ensure final qty does not exceed precision
    qty_str = format_quantity(qty, filters.step_size)
    try:
        float_qty = float(qty_str)
    except Exception:
        return 0.0
    return float_qty

def make_stop_prices(sell_stop_price_raw: float, filters: SymbolFilters) -> Tuple[float, float]:
    sp = max(sell_stop_price_raw, filters.min_price or 0.0)
    sp = filters.round_price(sp)
    lp = sp * (1.0 - STOP_LIMIT_OFFSET)
    lp = max(lp, filters.min_price or 0.0)
    lp = filters.round_price(lp)
    # Ensure limit price is not above stopPrice for SELL 
    if lp > sp:
        lp = sp
    return sp, lp


def print_bar_log(bar_time: pd.Timestamp, position: str, symbol: str, qty: float, entry_price: float, stop_loss: float, closed_pl: Optional[float], signal_type: str):
    t_str = bar_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    qty_str = f"{qty:.6f}" if qty else "-"
    entry_str = f"{entry_price:.2f}" if entry_price else "-"
    stop_str = f"{stop_loss:.2f}" if stop_loss else "-"
    pl_str = f"{closed_pl:.2f}" if closed_pl is not None else "-"
    if signal_type == "BUY":
        color = Colors.GREEN
    elif signal_type == "SELL":
        color = Colors.RED
    else:
        color = Colors.RESET
    print(f"{color}[{t_str}] | {signal_type} | {symbol} | {qty_str} | {entry_str} | {stop_str} | {pl_str}{Colors.RESET}")


def main():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")

    print(f"Starting live Donchian strategy on Binance.US - Symbol={SYMBOL}, Interval=1h, LiveMode={LIVE_MODE}")
    client = Client(api_key, api_secret, tld="us")  # Binance.US
    filters = SymbolFilters(client, SYMBOL)

    state = load_state(STATE_FILE)
    if state.get("symbol") != SYMBOL:
        state = {
            "position": "FLAT",
            "entry_price": 0.0,
            "qty": 0.0,
            "stop_order_id": None,
            "last_trade_close_time": None,
            "last_processed_close_time": None,
            "symbol": SYMBOL,
            "last_exit_price": None,
            "last_closed_pl": None
        }
        save_state(STATE_FILE, state)

    last_processed_close_time = state.get("last_processed_close_time")

    while True:
        try:
            df = fetch_klines(client, SYMBOL, INTERVAL, limit=300)
            ind = compute_indicators(df)
            if ind.empty or len(ind) < 2:
                time.sleep(10)
                continue

            # Only process the last CLOSED bar
            last_close_time = int(ind["close_time"].iloc[-2].value // 1_000_000)  # ms
            bar_time = ind["close_time"].iloc[-2]
            closed_pl = None
            signal_type = "FLAT"

            if last_close_time != last_processed_close_time:
                bars_since = bars_since_time(ind, state.get("last_trade_close_time"))

                position = state.get("position", "FLAT")
                stop_order_id = state.get("stop_order_id")
                entry_price = state.get("entry_price", 0.0)
                qty = state.get("qty", 0.0)
                stop_loss = None

                if position == "LONG" and stop_order_id:
                    st = get_order_status(client, SYMBOL, stop_order_id)
                    if st in (None, "FILLED", "PARTIALLY_FILLED", "CANCELED", "EXPIRED", "REJECTED"):
                        base_free, quote_free = get_balances(client, filters.baseAsset, filters.quoteAsset)
                        if base_free < (filters.min_qty or 0.0) * 0.5:
                            exit_price = float(ind["close"].iloc[-2])
                            closed_pl = (exit_price - entry_price) * qty if entry_price and qty else None
                            print_bar_log(bar_time, "FLAT", SYMBOL, 0.0, 0.0, 0.0, closed_pl, "SELL")
                            print(f"[{datetime.now(timezone.utc)}] Position exited (stop likely triggered).")
                            state["position"] = "FLAT"
                            state["entry_price"] = 0.0
                            state["qty"] = 0.0
                            state["stop_order_id"] = None
                            state["last_trade_close_time"] = last_processed_close_time
                            state["last_exit_price"] = exit_price
                            state["last_closed_pl"] = closed_pl
                            save_state(STATE_FILE, state)
                            position = "FLAT"
                            stop_order_id = None
                            entry_price = 0.0
                            qty = 0.0
                            stop_loss = 0.0
                            signal_type = "SELL"
                        else:
                            stop_loss = None
                    else:
                        stop_loss = None

                # If FLAT and not in cooldown, evaluate entry
                if state.get("position", "FLAT") == "FLAT" and bars_since >= COOLDOWN_BARS:
                    should_long, close_last, atr_last = last_closed_signal(ind)
                    if should_long and close_last is not None and atr_last is not None:
                        entry_price, raw_stop_price = compute_entry_and_stop(close_last, atr_last)
                        base_free, quote_free = get_balances(client, filters.baseAsset, filters.quoteAsset)
                        qty = compute_position_size(quote_free, entry_price, raw_stop_price, filters)
                        if qty <= 0.0 or math.isnan(qty) or math.isinf(qty):
                            print(f"[{datetime.now(timezone.utc)}] Skip entry: qty <= 0 (balance={quote_free:.2f} {filters.quoteAsset})")
                        else:
                            # Place market buy
                            order = place_market_buy(client, SYMBOL, qty, filters)
                            filled_qty = float(order.get("executedQty", qty))
                            fills = order.get("fills", [])
                            if fills:
                                total_quote = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                                total_qty = sum(float(f["qty"]) for f in fills)
                                if total_qty > 0:
                                    entry_price = total_quote / total_qty

                            sp, lp = make_stop_prices(raw_stop_price, filters)
                            sl_order = place_stop_loss_limit_sell(client, SYMBOL, filled_qty, sp, lp, filters)
                            stop_id = sl_order.get("orderId")

                            print(f"[{datetime.now(timezone.utc)}] ENTER LONG qty={filled_qty} entry≈{entry_price:.2f} stopPrice={sp:.2f} limitPrice={lp:.2f}")

                            state["position"] = "LONG"
                            state["entry_price"] = entry_price
                            state["qty"] = filled_qty
                            state["stop_order_id"] = stop_id
                            state["last_trade_close_time"] = last_processed_close_time
                            save_state(STATE_FILE, state)

                            stop_loss = sp
                            signal_type = "BUY"
                            print_bar_log(bar_time, "LONG", SYMBOL, filled_qty, entry_price, sp, None, "BUY")
                    else:
                        signal_type = "FLAT"
                        print_bar_log(bar_time, "FLAT", SYMBOL, 0.0, 0.0, 0.0, None, "FLAT")

                # If LONG and cooldown elapsed, trail stop upwards
                position = state.get("position", "FLAT")
                if position == "LONG":
                    bars_since = bars_since_time(ind, state.get("last_trade_close_time"))
                    if bars_since >= COOLDOWN_BARS:
                        close_last = float(ind["close"].iloc[-2])
                        atr_last = float(ind["ATR"].iloc[-2])
                        if not pd.isna(atr_last):
                            trail_raw = close_last - ATR_MULTIPLIER * atr_last
                            curr_stop_id = state.get("stop_order_id")
                            need_replace = True
                            curr_stop_price = None
                            if curr_stop_id:
                                try:
                                    od = with_retries(client.get_order, symbol=SYMBOL, orderId=curr_stop_id)
                                    st = od.get("status")
                                    if st == "NEW":
                                        curr_stop_price = float(od.get("stopPrice", 0.0) or 0.0)
                                    else:
                                        need_replace = False
                                except Exception:
                                    pass
                            if curr_stop_price is not None and trail_raw <= curr_stop_price:
                                need_replace = False

                            if need_replace:
                                sp, lp = make_stop_prices(trail_raw, filters)
                                if curr_stop_price is None or sp > curr_stop_price:
                                    cancel_order_safe(client, SYMBOL, curr_stop_id)
                                    sl_order = place_stop_loss_limit_sell(client, SYMBOL, state["qty"], sp, lp, filters)
                                    new_id = sl_order.get("orderId")
                                    print(f"[{datetime.now(timezone.utc)}] TRAIL STOP to stopPrice={sp:.2f} limitPrice={lp:.2f}")
                                    state["stop_order_id"] = new_id
                                    save_state(STATE_FILE, state)
                                    stop_loss = sp

                    entry_price = state.get("entry_price", 0.0)
                    qty = state.get("qty", 0.0)
                    curr_stop_id = state.get("stop_order_id")
                    curr_stop_price = None
                    if curr_stop_id:
                        try:
                            od = with_retries(client.get_order, symbol=SYMBOL, orderId=curr_stop_id)
                            if od.get("status") == "NEW":
                                curr_stop_price = float(od.get("stopPrice", 0.0) or 0.0)
                        except Exception:
                            curr_stop_price = None
                    stop_loss = curr_stop_price
                    print_bar_log(bar_time, "LONG", SYMBOL, qty, entry_price, stop_loss if stop_loss else 0.0, None, "BUY")

                last_processed_close_time = last_close_time
                state["last_processed_close_time"] = last_processed_close_time
                save_state(STATE_FILE, state)
            else:
                pass

        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] Error: {e}")

        time.sleep(30)

if __name__ == "__main__":
    main()
