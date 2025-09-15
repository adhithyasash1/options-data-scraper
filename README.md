# options scraper

this project provides a command-line tool to fetch, enrich, and export equity options chains. it is designed for quant research, analytics, and educational use.

## features

- fetches full option chains from optionsprofitcalculator.com
- adds analytics-ready fields:
  - implied volatility (via black-scholes bisection)
  - greeks: delta, gamma, vega, theta (per-day), rho
  - moneyness, time to expiry, mid price, bid-ask spread
- configurable risk-free rate and dividend yield
- efficient data handling:
  - optional [requests-cache](https://requests-cache.readthedocs.io/) for reduced network calls
  - optional [polars](https://pola.rs/) for faster dataframe operations
  - parquet, csv, and gzip csv outputs (requires pyarrow or fastparquet for parquet)
- parallel fetching of multiple tickers
- logging with clear progress and error reporting

## requirements

- python 3.8+
- required libraries:
  - requests
  - yfinance
  - numpy
  - pandas
  - python-dateutil
- optional libraries:
  - requests-cache (enables response caching)
  - polars (faster dataframes)
  - pyarrow or fastparquet (parquet output)

install everything with:

```bash
pip install -r requirements.txt
````

## usage

basic usage:

```bash
python options_scraper.py SPY
```

fetch multiple tickers:

```bash
python options_scraper.py SPY,QQQ,TSLA
```

export to parquet:

```bash
python options_scraper.py AAPL --out aapl_options.parquet
```

export to compressed csv with timestamped filename:

```bash
python options_scraper.py SPY,QQQ --out options_$(date +%Y%m%d_%H%M%S).csv.gz
```

pipe to stdout:

```bash
python options_scraper.py TSLA > tsla.csv
```

## arguments

* `tickers`: comma-separated list of tickers (required)
* `--out`: output file path. supports `.csv`, `.csv.gz`, `.parquet`
* `--r`: risk-free rate (annual, decimal). default: `0.02`
* `--q`: dividend yield (annual, decimal). default: `0.0`
* `--max-workers`: number of parallel workers. default: `5`

## logging

logs are printed to stderr with timestamp and level. info logs show ticker fetch progress, while warnings and errors highlight issues.

example log:

```
2025-09-15 09:00:53,369 - INFO - fetching chain for SPY
2025-09-15 09:00:55,259 - INFO - fetched 10674 records for SPY
```

## output schema

columns (when available):

* timestamp
* ticker
* underlying\_price
* expiration
* time\_to\_expiry\_days
* time\_to\_expiry\_years
* strike
* type (`c` or `p`)
* bid, ask, last, mid
* bid\_ask\_spread, bid\_ask\_spread\_pct
* moneyness
* implied\_vol
* open\_interest
* volume
* delta, gamma, vega, theta, rho

## notes

* implied volatility uses bisection search with bounds \[1e-6, 5.0].
* greeks are computed with implied vol if available, otherwise a fallback vol of 0.5.
* theta is scaled to per-day.
* options failing sanity checks (e.g., ask < bid) are skipped.

## license

mit license. use at your own risk. not for production trading without validation.
