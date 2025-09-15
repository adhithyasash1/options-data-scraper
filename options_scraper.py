#!/usr/bin/env python3
"""
options_scraper.py

enhanced options scraper with:
- implied volatility (bs implied vol via bisection)
- black-scholes greeks (delta, gamma, theta, vega, rho)
- analytics-ready features: moneyness, time_to_expiry_days, mid, spread_pct, bid_ask_spread
- performance improvements: optional requests_cache, optional polars usage
- output to stdout or file (csv, csv.gz, parquet if pyarrow available)

usage:
    python options_scraper.py SPY,QQQ
    python options_scraper.py SPY --out spy_options.parquet
    python options_scraper.py SPY,QQQ --out options_$(date +%Y%m%d_%H%M%S).csv.gz
"""
from __future__ import annotations
import sys
import time
import math
import logging
import argparse
import requests
from typing import List, Dict, Optional
from datetime import datetime, timezone
from dateutil import parser as dateparser

# optional libs
try:
    import requests_cache
    REQUESTS_CACHE_AVAILABLE = True
except Exception:
    REQUESTS_CACHE_AVAILABLE = False

try:
    import polars as pl
    POLARS_AVAILABLE = True
except Exception:
    POLARS_AVAILABLE = False

try:
    import pyarrow  # noqa: F401
    PYARROW_AVAILABLE = True
except Exception:
    PYARROW_AVAILABLE = False

import yfinance as yf
import numpy as np
import pandas as pd

# try to import config; fallback to defaults if missing
try:
    import config
    TIMEOUT = getattr(config, "DEFAULT_TIMEOUT", 10)
    MAX_RETRIES = getattr(config, "MAX_RETRIES", 3)
    RATE_LIMIT_DELAY = getattr(config, "RATE_LIMIT_DELAY", 0.5)
    LOG_LEVEL = getattr(config, "LOG_LEVEL", "INFO")
    CHUNK_SIZE = getattr(config, "CHUNK_SIZE", 100)
    DEFAULT_OUTPUT_DIR = getattr(config, "DEFAULT_OUTPUT_DIR", "data/raw")
except Exception:
    TIMEOUT = 10
    MAX_RETRIES = 3
    RATE_LIMIT_DELAY = 0.5
    LOG_LEVEL = "INFO"
    CHUNK_SIZE = 100
    DEFAULT_OUTPUT_DIR = "data/raw"

# logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr)
    ],
)
logger = logging.getLogger("options_scraper")

# requests session with user-agent
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; OptionsScraper/2.0; +https://github.com/adhithyasash1)"
})

# enable requests_cache if available (sqlite backend), short expiry
if REQUESTS_CACHE_AVAILABLE:
    try:
        # cache for 300 seconds by default (can be tuned)
        requests_cache.install_cache("opc_cache", expire_after=300)
        logger.info("requests_cache enabled (sqlite, 300s)")
    except Exception as e:
        logger.warning("requests_cache install failed: %s", e)


# -----------------------
# black-scholes and helpers
# -----------------------
def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    # use math.erf for cdf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(s: float, k: float, r: float, q: float, sigma: float, t: float, option_type: str) -> float:
    """
    black-scholes price for european call/put
    s: spot
    k: strike
    r: risk-free rate (annual, decimal)
    q: dividend yield (annual, decimal)
    sigma: volatility (annual, decimal)
    t: time to expiry in years
    option_type: 'c' or 'p'
    """
    if t <= 0:
        # option has expired, intrinsic value
        if option_type == "c":
            return max(0.0, s - k)
        else:
            return max(0.0, k - s)

    if sigma <= 0:
        # degenerate vol -> intrinsic discounted
        if option_type == "c":
            return max(0.0, s * math.exp(-q * t) - k * math.exp(-r * t))
        else:
            return max(0.0, k * math.exp(-r * t) - s * math.exp(-q * t))

    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if option_type == "c":
        price = s * math.exp(-q * t) * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2)
    else:
        price = k * math.exp(-r * t) * _norm_cdf(-d2) - s * math.exp(-q * t) * _norm_cdf(-d1)
    return price


def bs_vega(s: float, k: float, r: float, q: float, sigma: float, t: float) -> float:
    if t <= 0 or sigma <= 0:
        return 0.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    return s * math.exp(-q * t) * _norm_pdf(d1) * sqrt_t


def bs_greeks(s: float, k: float, r: float, q: float, sigma: float, t: float, option_type: str) -> Dict[str, float]:
    """
    compute greeks: delta, gamma, vega, theta (per day), rho
    theta returned is per-day (not per-year)
    """
    # edge cases
    if t <= 0:
        # expired option
        delta = 1.0 if (option_type == "c" and s > k) else 0.0
        if option_type == "p" and s < k:
            delta = -1.0
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    exp_minus_qt = math.exp(-q * t)
    exp_minus_rt = math.exp(-r * t)

    if option_type == "c":
        delta = exp_minus_qt * _norm_cdf(d1)
        theta = (- (s * sigma * exp_minus_qt * _norm_pdf(d1)) / (2 * sqrt_t)
                 - r * k * exp_minus_rt * _norm_cdf(d2)
                 + q * s * exp_minus_qt * _norm_cdf(d1))
        rho = k * t * exp_minus_rt * _norm_cdf(d2)
    else:
        delta = exp_minus_qt * (_norm_cdf(d1) - 1)
        theta = (- (s * sigma * exp_minus_qt * _norm_pdf(d1)) / (2 * sqrt_t)
                 + r * k * exp_minus_rt * _norm_cdf(-d2)
                 - q * s * exp_minus_qt * _norm_cdf(-d1))
        rho = -k * t * exp_minus_rt * _norm_cdf(-d2)

    gamma = exp_minus_qt * _norm_pdf(d1) / (s * sigma * sqrt_t) if (s > 0 and sigma > 0) else 0.0
    vega = bs_vega(s, k, r, q, sigma, t)

    # theta as per-year; convert to per-day
    theta_per_day = theta / 365.0

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta_per_day, "rho": rho}


def implied_vol_bisection(mkt_price: float, s: float, k: float, r: float, q: float,
                           t: float, option_type: str, tol: float = 1e-6, maxiter: int = 80) -> Optional[float]:
    """
    bisection search for implied vol. returns None if not found or invalid market price.
    bounds: [1e-6, 5.0] (0.000001 to 500% annual vol)
    """
    if mkt_price <= 0 or t < 0:
        return None

    # lower and upper vols
    low, high = 1e-6, 5.0

    # prices at bounds
    price_low = bs_price(s, k, r, q, low, t, option_type)
    price_high = bs_price(s, k, r, q, high, t, option_type)

    # if market price outside bounds, bail
    # but price with high vol should be >= market price for calls/puts in usual ranges; check both sides
    if not (min(price_low, price_high) <= mkt_price <= max(price_low, price_high)):
        # try expand bounds (rare)
        for high_try in (10.0, 20.0):
            price_high_try = bs_price(s, k, r, q, high_try, t, option_type)
            if min(price_low, price_high_try) <= mkt_price <= max(price_low, price_high_try):
                high = high_try
                price_high = price_high_try
                break
        else:
            # unable to bracket
            return None

    for i in range(maxiter):
        mid = 0.5 * (low + high)
        price_mid = bs_price(s, k, r, q, mid, t, option_type)
        # difference between theoretical and market
        diff = price_mid - mkt_price
        if abs(diff) < tol:
            return mid
        # decide which side to keep
        # compute price at low to know sign
        price_low = bs_price(s, k, r, q, low, t, option_type)
        if (price_low - mkt_price) * (price_mid - mkt_price) <= 0:
            # root in [low, mid]
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


# -----------------------
# data fetching + parsing
# -----------------------
def get_underlying_price(ticker: str) -> Optional[float]:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            logger.warning("no history for %s", ticker)
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.error("yfinance error for %s: %s", ticker, e)
        return None


def validate_option(contract: Dict) -> bool:
    # sanity checks
    if contract["ask"] < contract["bid"]:
        return False
    if contract["volume"] < 0 or contract["open_interest"] < 0:
        return False
    if contract["strike"] <= 0:
        return False
    return True


def fetch_options_chain(ticker: str, risk_free_rate: float = 0.02, dividend_yield: float = 0.0) -> List[Dict]:
    """
    fetch from optionsprofitcalculator.com and enrich with analytics features & greeks
    risk_free_rate and dividend_yield are simple inputs; for more accuracy, fetch r from market data.
    """
    logger.info("fetching chain for %s", ticker)
    underlying_price = get_underlying_price(ticker)
    if underlying_price is None:
        logger.error("no underlying for %s, skipping", ticker)
        return []

    url = f"https://www.optionsprofitcalculator.com/ajax/getOptions?stock={ticker}&reqId=1"
    retries = 0
    resp_json = None

    while retries < MAX_RETRIES:
        try:
            resp = session.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            resp_json = resp.json()
            break
        except requests.exceptions.RequestException as e:
            logger.warning("network error fetching %s (%d/%d): %s", ticker, retries + 1, MAX_RETRIES, e)
            retries += 1
            time.sleep(RATE_LIMIT_DELAY * (2 ** retries))
        except ValueError as e:
            logger.error("json decode error for %s: %s", ticker, e)
            return []

    if not resp_json or "options" not in resp_json or not resp_json["options"]:
        logger.warning("no options data for %s", ticker)
        return []

    timestamp = int(time.time())
    records: List[Dict] = []

    # options structure: {'options': {expiration: {'c': {strike: {b,a,l,oi,v}}, 'p': {...}}}}
    for expiration_str, chain in resp_json["options"].items():
        # parse date safely
        try:
            exp_dt = dateparser.parse(expiration_str)
            # normalize to date only
            exp_dt = exp_dt.date()
        except Exception:
            # if parsing fails, skip this expiration
            logger.debug("failed to parse expiration %s for %s", expiration_str, ticker)
            continue

        # compute time to expiry in days and years relative to now (UTC)
        now = datetime.now(timezone.utc).date()
        t_days = (exp_dt - now).days
        t_years = max(t_days / 365.0, 0.0)

        for side, strikes in chain.items():
            for strike_str, prices in strikes.items():
                try:
                    strike = float(strike_str)
                except Exception:
                    logger.debug("invalid strike %s", strike_str)
                    continue

                bid = float(prices.get("b", 0.0) or 0.0)
                ask = float(prices.get("a", 0.0) or 0.0)
                last = float(prices.get("l", 0.0) or 0.0)
                oi = int(prices.get("oi", 0) or 0)
                vol = int(prices.get("v", 0) or 0)

                # basic contract
                contract = {
                    "timestamp": timestamp,
                    "ticker": ticker,
                    "underlying_price": float(underlying_price),
                    "expiration": exp_dt.isoformat(),
                    "time_to_expiry_days": t_days,
                    "time_to_expiry_years": t_years,
                    "strike": strike,
                    "type": side,  # 'c' or 'p'
                    "bid": bid,
                    "ask": ask,
                    "last": last,
                    "open_interest": oi,
                    "volume": vol,
                }

                if not validate_option(contract):
                    # skip invalid rows
                    logger.debug("skipping invalid contract: %s", contract)
                    continue

                # analytics features
                mid = (bid + ask) / 2.0 if (bid > 0 or ask > 0) else last
                spread = ask - bid
                spread_pct = (spread / mid) if (mid > 0) else None
                moneyness = strike / underlying_price if underlying_price > 0 else None

                contract.update({
                    "mid": mid,
                    "bid_ask_spread": spread,
                    "bid_ask_spread_pct": spread_pct,
                    "moneyness": moneyness,
                })

                # compute implied vol using mid (if available) else last then bid/ask fallback
                market_price_for_iv = None
                if mid and mid > 0:
                    market_price_for_iv = mid
                elif last and last > 0:
                    market_price_for_iv = last
                elif ask and ask > 0:
                    market_price_for_iv = ask
                elif bid and bid > 0:
                    market_price_for_iv = bid

                iv = None
                greeks = {"delta": None, "gamma": None, "vega": None, "theta": None, "rho": None}
                if market_price_for_iv and t_years > 0:
                    try:
                        iv = implied_vol_bisection(
                            mkt_price=market_price_for_iv,
                            s=underlying_price,
                            k=strike,
                            r=risk_free_rate,
                            q=dividend_yield,
                            t=t_years,
                            option_type=side
                        )
                    except Exception as e:
                        logger.debug("iv bisection failed for %s %s %s : %s", ticker, expiration_str, strike, e)
                        iv = None

                contract["implied_vol"] = iv if iv is not None else None

                # compute greeks using implied vol if available, else use a fallback vol (e.g., 0.5)
                vol_for_greeks = iv if (iv is not None and iv > 0) else 0.5
                try:
                    greeks = bs_greeks(
                        s=underlying_price,
                        k=strike,
                        r=risk_free_rate,
                        q=dividend_yield,
                        sigma=vol_for_greeks,
                        t=t_years,
                        option_type=side
                    )
                except Exception as e:
                    logger.debug("greeks calc failed: %s", e)
                    greeks = {"delta": None, "gamma": None, "vega": None, "theta": None, "rho": None}

                # attach greeks
                contract.update({
                    "delta": greeks.get("delta"),
                    "gamma": greeks.get("gamma"),
                    "vega": greeks.get("vega"),
                    "theta": greeks.get("theta"),
                    "rho": greeks.get("rho"),
                })

                records.append(contract)

    return records


# -----------------------
# output helpers
# -----------------------
def to_dataframe(records: List[Dict]):
    """
    convert records to a dataframe. use polars if available for speed.
    """
    if POLARS_AVAILABLE:
        try:
            df = pl.from_dicts(records)
            return df
        except Exception as e:
            logger.warning("polars conversion failed, falling back to pandas: %s", e)

    # fallback pandas
    return pd.DataFrame.from_records(records)


def write_output(df, out_path: Optional[str]):
    """
    write dataframe to out_path or to stdout as csv.
    supports:
      - parquet (requires pyarrow or fastparquet)
      - .csv.gz (gzip)
      - .csv
    if df is polars DataFrame, use polars writers if available.
    """
    if out_path:
        logger.info("writing output to %s", out_path)
        if out_path.endswith(".parquet"):
            if POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
                df.write_parquet(out_path)
            else:
                # pandas to_parquet; requires pyarrow or fastparquet
                try:
                    if isinstance(df, pl.DataFrame):
                        df = df.to_pandas()
                    df.to_parquet(out_path, index=False)
                except Exception as e:
                    logger.error("parquet write failed (pyarrow needed?): %s", e)
                    # fallback to csv.gz
                    out_gz = out_path + ".csv.gz"
                    logger.info("falling back to gzip csv: %s", out_gz)
                    if isinstance(df, pl.DataFrame):
                        df = df.to_pandas()
                    df.to_csv(out_gz, index=False, compression="gzip")
        elif out_path.endswith(".csv.gz"):
            if POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
                df.write_csv(out_path)
            else:
                if isinstance(df, pl.DataFrame):
                    df = df.to_pandas()
                df.to_csv(out_path, index=False, compression="gzip")
        else:
            # default to csv
            if POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
                df.write_csv(out_path)
            else:
                if isinstance(df, pl.DataFrame):
                    df = df.to_pandas()
                df.to_csv(out_path, index=False)
    else:
        # write to stdout as csv
        if POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
            # polars DataFrame → convert to string first
            sys.stdout.write(df.write_csv())
        else:
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()
            df.to_csv(sys.stdout, index=False)


def parse_args():
    p = argparse.ArgumentParser(description="options data scraper with greeks & iv")
    p.add_argument("tickers", type=str, help="comma separated tickers, e.g. SPY,QQQ")
    p.add_argument("--out", type=str, default=None, help="output file path (csv, csv.gz, parquet)")
    p.add_argument("--r", type=float, default=0.02, help="risk-free rate (annual decimal)")
    p.add_argument("--q", type=float, default=0.0, help="dividend yield (annual decimal)")
    p.add_argument("--max-workers", type=int, default=5, help="max parallel workers for tickers")
    return p.parse_args()


def main():
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    all_records: List[Dict] = []

    # simple thread pool per ticker (network-bound)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=args.max_workers) as exe:
        futures = {exe.submit(fetch_options_chain, t, args.r, args.q): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                recs = fut.result()
                logger.info("fetched %d records for %s", len(recs), t)
                all_records.extend(recs)
            except Exception as e:
                logger.error("error fetching %s: %s", t, e)

    if not all_records:
        logger.warning("no records fetched, exiting")
        return

    df = to_dataframe(all_records)

    # optional light cleanup: ensure column ordering and types when pandas
    if POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
        # re-order columns for readability
        cols_order = [
            "timestamp", "ticker", "underlying_price", "expiration",
            "time_to_expiry_days", "time_to_expiry_years", "strike", "type",
            "bid", "ask", "mid", "bid_ask_spread", "bid_ask_spread_pct",
            "moneyness", "implied_vol",
            "last", "open_interest", "volume",
            "delta", "gamma", "vega", "theta", "rho"
        ]
        # keep only available columns
        cols = [c for c in cols_order if c in df.columns]
        df = df.select(cols)
    else:
        # pandas
        if isinstance(df, pd.DataFrame):
            cols_order = [
                "timestamp", "ticker", "underlying_price", "expiration",
                "time_to_expiry_days", "time_to_expiry_years", "strike", "type",
                "bid", "ask", "mid", "bid_ask_spread", "bid_ask_spread_pct",
                "moneyness", "implied_vol",
                "last", "open_interest", "volume",
                "delta", "gamma", "vega", "theta", "rho"
            ]
            cols = [c for c in cols_order if c in df.columns]
            df = df[cols]

    write_output(df, args.out)


if __name__ == "__main__":
    main()