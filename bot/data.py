"""Fetch market data with TWO free sources and a hard time limit.

Primary: Kraken. Backup: Coinbase. If one is slow or down, we use the other.
Every network call runs inside a worker with a strict deadline, so the bot
can NEVER get stuck waiting forever (important on shared cloud servers).
"""
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import pandas as pd
import requests

KRAKEN = "https://api.kraken.com/0/public"
COINBASE = "https://api.exchange.coinbase.com"

KRAKEN_PAIRS = {"BTC/USD": "XBTUSD", "ETH/USD": "ETHUSD"}
COINBASE_PAIRS = {"BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD"}

HARD_DEADLINE = 20          # seconds — absolute max per network call
_executor = ThreadPoolExecutor(max_workers=4)
_cache = {}


def _get_json(url, params=None):
    """HTTP GET with a hard deadline that cannot hang, even if the
    server drips data slowly."""
    def _do():
        r = requests.get(url, params=params, timeout=(5, 10),
                         headers={"User-Agent": "paper-trading-bot"})
        r.raise_for_status()
        return r.json()

    future = _executor.submit(_do)
    try:
        return future.result(timeout=HARD_DEADLINE)
    except FutureTimeout:
        future.cancel()
        raise TimeoutError(f"{url} took longer than {HARD_DEADLINE}s")


# ---------------- OHLC candles ----------------
def _ohlc_kraken(symbol, interval):
    data = _get_json(f"{KRAKEN}/OHLC",
                     {"pair": KRAKEN_PAIRS[symbol], "interval": interval})
    if data.get("error"):
        raise RuntimeError(f"Kraken error: {data['error']}")
    candles = next(v for k, v in data["result"].items() if k != "last")
    df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close",
                                        "vwap", "volume", "count"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s")
    return df[["time", "open", "high", "low", "close", "volume"]]


def _ohlc_coinbase(symbol, interval):
    rows = _get_json(f"{COINBASE}/products/{COINBASE_PAIRS[symbol]}/candles",
                     {"granularity": interval * 60})
    # Coinbase rows: [time, low, high, open, close, volume], newest first
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["time", "open", "high", "low", "close", "volume"]]


def get_ohlc(symbol: str, interval: int = 60) -> pd.DataFrame:
    """Hourly candles. Tries Kraken, falls back to Coinbase.
    Also keeps the last good copy in memory as an emergency fallback."""
    key = (symbol, interval, int(time.time() // 60))
    if key in _cache:
        return _cache[key]

    last_err = None
    for fetch in (_ohlc_kraken, _ohlc_coinbase):
        try:
            df = fetch(symbol, interval)
            if len(df) >= 50:
                for k in [k for k in _cache if isinstance(k, tuple) and len(k) == 3]:
                    del _cache[k]
                _cache[key] = df
                _cache[("last_good_ohlc", symbol, interval)] = df
                return df
        except Exception as e:
            last_err = e

    stale = _cache.get(("last_good_ohlc", symbol, interval))
    if stale is not None:
        return stale
    raise RuntimeError(f"All data sources failed: {last_err}")


# ---------------- current price ----------------
def _price_kraken(symbol):
    data = _get_json(f"{KRAKEN}/Ticker", {"pair": KRAKEN_PAIRS[symbol]})
    if data.get("error"):
        raise RuntimeError(f"Kraken error: {data['error']}")
    ticker = next(iter(data["result"].values()))
    return float(ticker["c"][0])


def _price_coinbase(symbol):
    data = _get_json(f"{COINBASE}/products/{COINBASE_PAIRS[symbol]}/ticker")
    return float(data["price"])


def get_price(symbol: str) -> float:
    last_err = None
    for fetch in (_price_kraken, _price_coinbase):
        try:
            price = fetch(symbol)
            _cache[("last_good_price", symbol)] = price
            return price
        except Exception as e:
            last_err = e

    stale = _cache.get(("last_good_price", symbol))
    if stale is not None:
        return stale
    raise RuntimeError(f"All price sources failed: {last_err}")
