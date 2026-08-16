"""Market data engine with FOUR independent free sources.

Free cloud hosts share internet addresses among many users, so public
crypto APIs often throttle them. We try Kraken, Coinbase, Bitstamp and
OKX in turn — whichever answers first wins. Every call runs in its own
disposable thread with a hard deadline, so nothing can ever hang the bot.
Candles are cached for 5 minutes to stay well under rate limits.
"""
import threading
import time

import pandas as pd
import requests

HARD_DEADLINE = 25       # seconds max per network call
CANDLE_CACHE_SEC = 300   # refetch candles at most every 5 minutes
PRICE_CACHE_SEC = 45     # refetch price at most every 45 seconds

_cache = {}
_lock = threading.Lock()


def _fetch_json(url, params=None):
    """GET with a hard wall-clock deadline. The worker thread is abandoned
    (daemon) if it overruns — the caller never blocks longer than the deadline."""
    result = {}

    def _do():
        try:
            r = requests.get(url, params=params, timeout=(5, 15),
                             headers={"User-Agent": "Mozilla/5.0 (paper-trading-bot)"})
            r.raise_for_status()
            result["data"] = r.json()
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(HARD_DEADLINE)
    if "data" in result:
        return result["data"]
    if "error" in result:
        raise result["error"]
    raise TimeoutError(f"{url} exceeded {HARD_DEADLINE}s")


def _std_df(rows):
    """rows: list of [ts_sec, open, high, low, close, volume] -> clean DataFrame."""
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="s")
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


# ---------------- candle sources (hourly) ----------------
def _ohlc_kraken(symbol):
    pair = {"BTC/USD": "XBTUSD", "ETH/USD": "ETHUSD"}[symbol]
    data = _fetch_json("https://api.kraken.com/0/public/OHLC",
                       {"pair": pair, "interval": 60})
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    candles = next(v for k, v in data["result"].items() if k != "last")
    return _std_df([[c[0], c[1], c[2], c[3], c[4], c[6]] for c in candles])


def _ohlc_coinbase(symbol):
    prod = {"BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD"}[symbol]
    rows = _fetch_json(f"https://api.exchange.coinbase.com/products/{prod}/candles",
                       {"granularity": 3600})
    # [time, low, high, open, close, volume] newest first
    return _std_df([[r[0], r[3], r[2], r[1], r[4], r[5]] for r in rows])


def _ohlc_bitstamp(symbol):
    pair = {"BTC/USD": "btcusd", "ETH/USD": "ethusd"}[symbol]
    data = _fetch_json(f"https://www.bitstamp.net/api/v2/ohlc/{pair}/",
                       {"step": 3600, "limit": 720})
    rows = data["data"]["ohlc"]
    return _std_df([[int(r["timestamp"]), r["open"], r["high"], r["low"],
                     r["close"], r["volume"]] for r in rows])


def _ohlc_okx(symbol):
    inst = {"BTC/USD": "BTC-USDT", "ETH/USD": "ETH-USDT"}[symbol]
    data = _fetch_json("https://www.okx.com/api/v5/market/candles",
                       {"instId": inst, "bar": "1H", "limit": "300"})
    # [ts_ms, o, h, l, c, vol, ...] newest first
    return _std_df([[int(r[0]) // 1000, r[1], r[2], r[3], r[4], r[5]]
                    for r in data["data"]])


# ---------------- price sources ----------------
def _price_kraken(symbol):
    pair = {"BTC/USD": "XBTUSD", "ETH/USD": "ETHUSD"}[symbol]
    data = _fetch_json("https://api.kraken.com/0/public/Ticker", {"pair": pair})
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return float(next(iter(data["result"].values()))["c"][0])


def _price_coinbase(symbol):
    prod = {"BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD"}[symbol]
    return float(_fetch_json(
        f"https://api.exchange.coinbase.com/products/{prod}/ticker")["price"])


def _price_bitstamp(symbol):
    pair = {"BTC/USD": "btcusd", "ETH/USD": "ethusd"}[symbol]
    return float(_fetch_json(
        f"https://www.bitstamp.net/api/v2/ticker/{pair}/")["last"])


def _price_okx(symbol):
    inst = {"BTC/USD": "BTC-USDT", "ETH/USD": "ETH-USDT"}[symbol]
    data = _fetch_json("https://www.okx.com/api/v5/market/ticker", {"instId": inst})
    return float(data["data"][0]["last"])


# ---------------- public API with rotation + caching ----------------
_OHLC_SOURCES = [("kraken", _ohlc_kraken), ("coinbase", _ohlc_coinbase),
                 ("bitstamp", _ohlc_bitstamp), ("okx", _ohlc_okx)]
_PRICE_SOURCES = [("kraken", _price_kraken), ("coinbase", _price_coinbase),
                  ("bitstamp", _price_bitstamp), ("okx", _price_okx)]

# remember which source worked last and try it first next time
_preferred = {"ohlc": 0, "price": 0}


def _try_sources(kind, sources, symbol):
    order = list(range(len(sources)))
    start = _preferred[kind]
    order = order[start:] + order[:start]
    last_err = None
    for idx in order:
        name, fn = sources[idx]
        try:
            out = fn(symbol)
            _preferred[kind] = idx
            return out, name
        except Exception as e:
            last_err = e
    raise RuntimeError(f"all sources failed, last: {last_err}")


def get_ohlc(symbol: str, interval: int = 60) -> pd.DataFrame:
    """Hourly candles from the first source that answers (cached 5 min)."""
    now = time.time()
    with _lock:
        hit = _cache.get(("ohlc", symbol))
        if hit and now - hit[1] < CANDLE_CACHE_SEC:
            return hit[0]

    try:
        df, _src = _try_sources("ohlc", _OHLC_SOURCES, symbol)
        if len(df) < 50:
            raise RuntimeError("too few candles")
        with _lock:
            _cache[("ohlc", symbol)] = (df, now)
        return df
    except Exception:
        with _lock:
            hit = _cache.get(("ohlc", symbol))
        if hit:                      # stale data beats no data
            return hit[0]
        raise


def get_price(symbol: str) -> float:
    """Latest price from the first source that answers (cached 45s)."""
    now = time.time()
    with _lock:
        hit = _cache.get(("price", symbol))
        if hit and now - hit[1] < PRICE_CACHE_SEC:
            return hit[0]

    try:
        price, _src = _try_sources("price", _PRICE_SOURCES, symbol)
        with _lock:
            _cache[("price", symbol)] = (price, now)
        return price
    except Exception:
        with _lock:
            hit = _cache.get(("price", symbol))
        if hit:
            return hit[0]
        raise
