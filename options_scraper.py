import sys
import requests
import time
import csv
import io


# A session object can improve performance by reusing underlying TCP connections.
# We also set a common User-Agent header, as some services may block default script agents.
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})


import yfinance as yf

def get_underlying_price(ticker: str) -> float | None:
    """
    Fetches the latest closing price for a stock ticker using the yfinance library.
    This method is robust and handles Yahoo's authentication requirements.
    """
    try:
        stock = yf.Ticker(ticker)
        # Get the most recent trading day's data
        hist = stock.history(period="1d")
        if hist.empty:
            print(f"Error: No history data found for '{ticker}'. The ticker may be invalid.", file=sys.stderr)
            return None
        # Return the 'Close' price from the last available day
        return hist['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching price for '{ticker}' with yfinance: {e}", file=sys.stderr)
        return None

def get_options_chain(ticker: str) -> list[dict]:
    """
    Fetches the entire options chain for a ticker from optionsprofitcalculator.com.

    Args:
        ticker: The stock ticker symbol (e.g., 'QQQ').

    Returns:
        A list of dictionaries, where each dictionary represents an option contract.
    """
    print(f"Fetching data for {ticker}...", file=sys.stderr)

    underlying_price = get_underlying_price(ticker)
    if underlying_price is None:
        return []  # Stop if we can't get the underlying price

    api_url = f"https://www.optionsprofitcalculator.com/ajax/getOptions?stock={ticker}&reqId=1"

    try:
        response = session.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: Network issue when fetching options for '{ticker}'. {e}", file=sys.stderr)
        return []
    except requests.exceptions.JSONDecodeError:
        print(f"Error: Failed to decode JSON response for '{ticker}'. API may have changed.", file=sys.stderr)
        return []

    all_contracts = []
    timestamp = int(time.time())

    # The JSON structure is {'options': {expiration: {side: {strike: {prices}}}}}
    if 'options' not in data or not data['options']:
        print(f"Warning: No options data found for '{ticker}' in API response.", file=sys.stderr)
        return []

    for expiration_date, chain in data['options'].items():
        for side, contracts in chain.items():  # side is 'c' for calls, 'p' for puts
            for strike, prices in contracts.items():
                contract = {
                    "timestamp": timestamp,
                    "ticker": ticker,
                    "underlying_price": underlying_price,
                    "expiration": expiration_date,
                    "strike": float(strike),
                    "type": side,
                    "bid": float(prices.get('b', 0.0)),
                    "ask": float(prices.get('a', 0.0)),
                    "last": float(prices.get('l', 0.0)),
                    "open_interest": int(prices.get('oi', 0)),
                    "volume": int(prices.get('v', 0)),
                }
                all_contracts.append(contract)

    return all_contracts

def main():
    """
    Main execution function. Parses command-line arguments and prints option data.
    """
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <TICKER1,TICKER2,...>", file=sys.stderr)
        print(f"Example: python {sys.argv[0]} SPY,QQQ", file=sys.stderr)
        sys.exit(1)

    tickers_arg = sys.argv[1]
    tickers = [ticker.strip().upper() for ticker in tickers_arg.split(',')]

    for ticker in tickers:
        if not ticker:
            continue

        options_data = get_options_chain(ticker)

        for option in options_data:
            # Format and print each option contract as a CSV row to standard output
            print(
                f"{option['timestamp']},"
                f"{option['ticker']},"
                f"{option['underlying_price']:.2f},"
                f"{option['expiration']},"
                f"{option['strike']:.2f},"
                f"{option['type']},"
                f"{option['bid']:.2f},"
                f"{option['ask']:.2f},"
                f"{option['last']:.2f},"
                f"{option['open_interest']},"
                f"{option['volume']}"
            )

if __name__ == "__main__":
    main()
