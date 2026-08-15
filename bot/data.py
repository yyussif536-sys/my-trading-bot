"""Fetch market data from Kraken's free public API (no account needed)."""
import requests
import pandas as pd
import time

KRAKEN = "https://api.kraken.com/0/public"

# Kraken uses odd internal names for some pairs
PAIRS = {
    "BTC/USD": "XBTUSD",
    "ETH/USD": "ETHUSD",
}

_cache = {}


def get_ohlc(symbol: str, interval: int = 60) -> pd.DataFrame:
    """Get up to ~720 candles. interval is in minutes (60 = hourly).
    Returns DataFrame with time, open, high, low, close, volume."""
    key = (symbol, interval, int(time.time() // 60))  # cache for 1 minute
    if key in _cache:
        return _cache[key]

    pair = PAIRS[symbol]
    r = requests.get(f"{KRAKEN}/OHLC", params={"pair": pair, "interval": interval}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken error: {data['error']}")
    result = data["result"]
    candles = next(v for k, v in result.items() if k != "last")

    df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    for col in ["open", "high", "low", "close", "vwap", "volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s")
    df = df[["time", "open", "high", "low", "close", "volume"]]
    _cache.clear()
    _cache[key] = df
    return df


def get_price(symbol: str) -> float:
    """Current market price."""
    pair = PAIRS[symbol]
    r = requests.get(f"{KRAKEN}/Ticker", params={"pair": pair}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken error: {data['error']}")
    ticker = next(iter(data["result"].values()))
    return float(ticker["c"][0])  # last trade price
