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
