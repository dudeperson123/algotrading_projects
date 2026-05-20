# algotrading_projects
Steps to use for each program:
1. DonchianWFV.py:
     1. Go to https://www.binance.us/institutions/market-history
     2. Click Download for the Candlestick
     3. Select symbol BTCUSDT(or any other you like), interval monthly, granularity 1H(or any other for testing).
     4. For date range, select the earliest month you can(usually Sep 2019), then the latest month you can(usually Aug 2020). Download the zip files.
     5. Repeat this process for dates Sep 2020 to Aug 2021, and so on until you've downloaded all th zip files available(or as much as you need).
     6. Put all the zip files in a folder named "BTCUSDT_BINANCEUS_1H"
     7. Run DonchianWFV_datacollecter.py (make sure filepath is correct). This program extracts, combines, and cleans all the donwloaded files.
     8. Run DonchianWFV.py (make sure filepath is correct). The output is the summary of the OOS backtest, along with a graph of the equity curve.

2. PatternMatching.py:
     1. Run nasdaq_generator.py (This will create a folder of all historically available 1D data for all nasdaq symbols from yfinance)
     2. Run patteernMatching.py (This program will scan the historical data, find price/volume patterns that are similar to a reference window, and then evaluate what happened after those similar patterns. This is not the backtester. It simply finds previous matches for every stock, derives metrics from those matches, and then outputs csvs with symbols, match metrics, and future return, to be used for backtesting.)
     3. Run patternMatchingResults.py (This is the backtester. It looks at all the data that the previous program generated and builds a simulated portfolio, dynamically adjusts risk, and incorporates position sizing. It does this primarily by a composite ranking score using the match metrics that were mentioned previously, dynamically chooses how many of the top stocks to pick, weights sizing by score, and builds a full equity curve. A major feature is the risk management, which includes volatility targeting, volatility floor, drawdown prediction, and a momentum boost. At the end, it computes backtest metrics and also diagnostics, which include feature usefullness by correlation and cap efficiency analysis by binning exposures.)
