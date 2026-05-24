import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._fields")

import math, os, uuid, time, ast, datetime, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import ccxt
from openai import OpenAI
from datetime import datetime, timezone, timedelta

global model, symbol, interval, limit, api_calls, confidence_threshold, flat_weight, stop_percent, retries, delay, leverage, stop_price, data_fetch_time, size_risk, buffer, total_equity_usd, e
model = "gpt-5-mini"
symbol = 'BTC/USD:USD'
interval = '1h'
limit = 200
api_calls = 10
confidence_threshold = 70
flat_weight = 0
stop_percent = 2
size_risk = 100
retries = 3
delay = 2
leverage = 1
buffer = 0.5
data_fetch_time = None
e = None
MIN_CONTRACT = 0.0001
stop_price = 0
client = OpenAI()

# Kraken setup
k, s = os.getenv("KRAKEN_API_KEY"), os.getenv("KRAKEN_API_SECRET")
kf_demo = ccxt.krakenfutures({'apiKey': k, 'secret': s, 'enableRateLimit': True})
kf_demo.set_sandbox_mode(True)
kf_live = ccxt.krakenfutures({'apiKey': k, 'secret': s, 'enableRateLimit': True})
kf_live.set_sandbox_mode(False)
    
# Initialize fake equity; not used because of kraken demo headaches, just here in case of errors
total_equity_usd = 10000

def open_kraken_position(side, confidence, latest_price):
    global stop_price, total_equity_usd
    for _ in range(retries):
        try:
            wallets = kf_demo.fetch_balance()
            break
        except ccxt.NetworkError:
            time.sleep(delay)   
        except Exception as e:
            raise e
    else:
        raise Exception("Failed to fetch balance after retries")
    

    total_equity_usd = wallets['total']['USD']
    dollar_size = (size_risk/100 * total_equity_usd) * (confidence/100) 
    max_dollar_size = (1 - buffer/100) * total_equity_usd
    dollar_size = min(dollar_size, max_dollar_size)
    size = dollar_size / latest_price
    size = round(size / MIN_CONTRACT) * MIN_CONTRACT
    size = max(size, MIN_CONTRACT)
    kf_demo.create_order(symbol=symbol, type='market', side=side.lower(), amount=size,
                         params={'cliOrdId': str(uuid.uuid4()), 'leverage': leverage})

    if dollar_size > (stop_percent / 100) * total_equity_usd:
        pct_change = (((stop_percent / 100) * total_equity_usd) / dollar_size) / leverage
        stop_price = latest_price - pct_change * latest_price if side.lower() == 'buy' else latest_price + pct_change * latest_price
        # Place stop-loss order
        stop_side = 'sell' if side.lower() == 'buy' else 'buy'
        kf_demo.create_order(
            symbol=symbol,
            type='stop',
            side=stop_side,
            amount=size,
            price=stop_price,
            params={'reduceOnly': True, 'triggerPrice': stop_price}
        )

def close_kraken_position(current_positions):  
    for p in current_positions:
        if p['symbol'].upper() != symbol:
            continue
        sz = float(p['contracts'])
        if sz < MIN_CONTRACT:
            continue
        cs = 'sell' if p['side'] == 'long' else 'buy'
        amount = round(sz / MIN_CONTRACT) * MIN_CONTRACT
        kf_demo.create_order(symbol=symbol, type='market', side=cs.lower(), amount=amount,
                             params={'reduce_only': True, 'cliOrdId': str(uuid.uuid4())})

def fetch_data():
    global data_fetch_time, latest_price, low_price, high_price
    
    for _ in range(retries):
        try:
            ohlcv = kf_live.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
            break
        except ccxt.NetworkError:
            time.sleep(delay)   
        except Exception as e:
            raise e
    else:
        print("Failed to fetch data after retries. Skipping this hour.")
        return None, None, None
    
    
    data_fetch_time = datetime.now(timezone.utc)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    last_bar_ts = df['timestamp'].iloc[-1]
    last_bar_dt = datetime.fromtimestamp(last_bar_ts / 1000, tz=timezone.utc)
    if last_bar_dt.hour == data_fetch_time.hour:
        df = df.iloc[:-1]

    df = df[['open','high','low','close','volume']].astype(float)
    df['volume'] = df['volume'].round(5)
    latest_price = df['close'].iloc[-1]
    return data_fetch_time, latest_price, df.to_csv(index=False)

def prompt_gpt(df_string, data_fetch_time):
    prompt = f"""
You are the BEST trading AI. Your primary goal is to MAXIMIZE profit and MINIMIZE risk.
Given Kraken Futures {symbol} {interval} data, return the OPTIMAL trading action. 

Input data:
{df_string}

Data fetched at: {data_fetch_time}

Return ONLY a valid python list:
["BUY" or "FLAT" or "SELL", "Rationale"]

ONLY output BUY or SELL if you are CERTAIN, or as CLOSE to CERTAIN as POSSIBLE, that opening a position in that direction will result in MAXIMUM profit and MINIMUM risk. In ANY other case, output FLAT.

Example Outputs:
["BUY", "The price recently bounced off a strong support level with high trading volume. Recent price action suggests a likely upward continuation. Entering a long position has a high probability of high profit."]
["FLAT", "Price has been consolidating in a narrow range with low volume, showing no clear directional bias. Market conditions are uncertain, so it is safer to stay out until a decisive move occurs."]
["SELL", "Selling pressure is increasing, and recent price action shows likely downward continuation. Entering a short position has a high probability of high profit."]
"""
    def gpt_request():
        response = client.responses.create(model=model, input=prompt)
        response_text = response.output_text.strip()
        try:
            response_list = ast.literal_eval(response_text)
        except:
            response_list = ["FLAT", "Invalid Response"]
        return response_list
    
    responses = []
    with ThreadPoolExecutor(max_workers=api_calls) as executor:
        futures = [executor.submit(gpt_request) for _ in range(api_calls)]
        for future in as_completed(futures):
            responses.append(future.result())
    return responses

def trade_and_print(responses):
    global stop_price, latest_price
    parsed_responses = [r[0] for r in responses]
    rationales = [r[1] for r in responses]
    num_buys = 0
    num_sells = 0
    num_flats = 0

    for r in parsed_responses:
        if r == "BUY":
            num_buys += 1
        if r == "SELL":
            num_sells += 1
        if r == "FLAT":
            num_flats += 1

    score = num_buys - num_sells
    score = score - flat_weight * num_flats if score > 0 else score + flat_weight * num_flats if score < 0 else score

    action = "BUY" if score > 0 else "SELL" if score < 0 else "FLAT"
    confidence = round(abs(score / api_calls * 100), 2) if action != "FLAT" else 0.0

    now = datetime.now()
    now_simplified = now.strftime("%Y-%m-%d %H:%M")
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # COMMENT OUT BECAUSE OF KRAKEN DEMO ISSUES
    # # Check for open positions
    # for _ in range(retries):
    #     try:
    #         current_positions = kf_demo.fetch_positions()
    #         break
    #     except ccxt.NetworkError:
    #         time.sleep(delay)   
    #     except Exception as e:
    #         raise e
    # else:
    #     raise Exception("Failed to fetch current positions after retries")
    
    # actual_position = next((p for p in current_positions if p['symbol'].upper() == symbol and float(p['contracts']) >= MIN_CONTRACT), None)
    # open_position = "BUY" if actual_position and actual_position["side"] == "long" else "SELL" if actual_position else None

    # if actual_position:
    #     if action != open_position:
    #         close_kraken_position(current_positions)
    #     actual_position = None

    # if not actual_position and confidence >= confidence_threshold:
    #     if action == "BUY":
    #         open_kraken_position("buy", confidence, latest_price)
    #     elif action == "SELL":
    #         open_kraken_position("sell", confidence, latest_price)

    reset = "\033[0m"
    if confidence >= confidence_threshold and action != "FLAT":
        color = "\033[32m" if action=="BUY" else "\033[31m"
        line = f"{color}{now_simplified:<16} | {symbol + ' ' + interval:<17}- {latest_price:>10} | Action: {action:<4} | Confidence: {confidence:>5}%{reset}"
    else:
        action_color = "\033[32m" if action=="BUY" else "\033[31m" if action=="SELL" else ""
        action_reset = "\033[0m" if action_color else ""
        line = f"{now_simplified:<16} | {symbol + ' ' + interval:<17}- {latest_price:>10} | {action_color}Action: {action:<4}{action_reset} | Confidence: {confidence:>5}%"
    print(line)
    return {"datetime": now_str, "symbol": symbol, "latest_price": latest_price, "action": action, "confidence": confidence, "buys": num_buys, "sells": num_sells, "flats": num_flats, "rationales": rationales}

def interval_to_seconds(interval):
    num, unit = int(interval[:-1]), interval[-1]
    if unit=='m': return num*60
    elif unit=='h': return num*3600
    elif unit=='d': return num*86400
    else: raise ValueError(f"Unsupported interval: {interval}")

def wait_until_next_interval(interval):
    now = datetime.now()
    interval_sec = interval_to_seconds(interval)
    seconds_to_next = interval_sec - (int(now.timestamp()) % interval_sec)
    time.sleep(seconds_to_next)

def main():
    global latest_price, low_price, high_price
    data_fetch_time, latest_price, df_string = fetch_data()
    if data_fetch_time:
        responses = prompt_gpt(df_string, data_fetch_time)
        new_row = trade_and_print(responses)
    else:
        new_row = {"datetime": "Fetch data timeout: Skipped"}
        pd.DataFrame([new_row]).to_csv(
            "trading_log.csv", mode='a', index=False, header=not os.path.exists("trading_log.csv"), float_format='%.2f'
        )

try:
    while True:
        wait_until_next_interval(interval)
        main()
except KeyboardInterrupt:
    print("\nDone.")
