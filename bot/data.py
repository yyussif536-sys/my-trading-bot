"""Market data engine with MANY independent free sources.

Cloud hosts share internet addresses among thousands of users, so popular
crypto APIs often throttle them. We keep a long list of sources — big
exchanges AND less-crowded ones — try each with a hard deadline, and
remember which one worked to use it first next time.
"""
import threading
import time

import pandas as pd
import requests

HARD_DEADLINE = 10       # seconds max per network call
CANDLE_CACHE_SEC = 300   # refetch candles at most every 5 minutes
PRICE_CACHE_SEC = 60     # refetch price at most every minute

_cache = {}
_lock = threading.Lock()


def _fetch_json(url, params=None):
    """GET with a hard wall-clock deadline — can never hang the bot."""
    result = {}

    def _do():
        try:
            r = requests.get(url, params=params, timeout=(4, 8),
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
    """rows: [ts_sec, open, high, low, close, volume] -> clean DataFrame."""
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="s")
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


# ================= candle sources (hourly) =================
def _ohlc_kraken(symbol):
    pair = {"BTC/USD": "XBTUSD", "ETH/USD": "ETHUSD"}[symbol]
    data = _fetch_json("https://api.kraken.com/0/public/OHLC",
                       {"pair": pair, "interval": 60})
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    candles = next(v for k, v in data["result"].items() if k != "last")
    return _std_df([[c[0], c[1], c[2], c[3], c[4], c[6]] for c in candles])


def _ohlc_binanceus(symbol):
    sym = {"BTC/USD": "BTCUSD", "ETH/USD": "ETHUSD"}[symbol]
    rows = _fetch_json("https://api.binance.us/api/v3/klines",
                       {"symbol": sym, "interval": "1h", "limit": 720})
    return _std_df([[int(r[0]) // 1000, r[1], r[2], r[3], r[4], r[5]] for r in rows])


def _ohlc_gemini(symbol):
    sym = {"BTC/USD": "btcusd", "ETH/USD": "ethusd"}[symbol]
    rows = _fetch_json(f"https://api.gemini.com/v2/candles/{sym}/1hr")
    return _std_df([[int(r[0]) // 1000, r[1], r[2], r[3], r[4], r[5]] for r in rows])


def _ohlc_bitfinex(symbol):
    sym = {"BTC/USD": "tBTCUSD", "ETH/USD": "tETHUSD"}[symbol]
    rows = _fetch_json(f"https://api-pub.bitfinex.com/v2/candles/trade%3A1h%3A{sym}/hist",
                       {"limit": 720})
    # [ts_ms, open, close, high, low, volume]
    return _std_df([[int(r[0]) // 1000, r[1], r[3], r[4], r[2], r[5]] for r in rows])


def _ohlc_kucoin(symbol):
    sym = {"BTC/USD": "BTC-USDT", "ETH/USD": "ETH-USDT"}[symbol]
    data = _fetch_json("https://api.kucoin.com/api/v1/market/candles",
                       {"type": "1hour", "symbol": sym})
    # [time_s, open, close, high, low, volume, turnover] newest first
    return _std_df([[int(r[0]), r[1], r[3], r[4], r[2], r[5]] for r in data["data"]])


def _ohlc_okx(symbol):
    inst = {"BTC/USD": "BTC-USDT", "ETH/USD": "ETH-USDT"}[symbol]
    data = _fetch_json("https://www.okx.com/api/v5/market/candles",
                       {"instId": inst, "bar": "1H", "limit": "300"})
    return _std_df([[int(r[0]) // 1000, r[1], r[2], r[3], r[4], r[5]]
                    for r in data["data"]])


def _ohlc_bitstamp(symbol):
    pair = {"BTC/USD": "btcusd", "ETH/USD": "ethusd"}[symbol]
    data = _fetch_json(f"https://www.bitstamp.net/api/v2/ohlc/{pair}/",
                       {"step": 3600, "limit": 720})
    return _std_df([[int(r["timestamp"]), r["open"], r["high"], r["low"],
                     r["close"], r["volume"]] for r in data["data"]["ohlc"]])


def _ohlc_coinbase(symbol):
    prod = {"BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD"}[symbol]
    rows = _fetch_json(f"https://api.exchange.coinbase.com/products/{prod}/candles",
                       {"granularity": 3600})
    return _std_df([[r[0], r[3], r[2], r[1], r[4], r[5]] for r in rows])


def _ohlc_yahoo(symbol):
    sym = {"BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD"}[symbol]
    data = _fetch_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                       {"interval": "1h", "range": "30d"})
    res = data["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    rows = []
    for i, t in enumerate(ts):
        if q["close"][i] is None:
            continue
        rows.append([t, q["open"][i], q["high"][i], q["low"][i],
                     q["close"][i], q["volume"][i] or 0])
    return _std_df(rows)


def _ohlc_coingecko(symbol):
    coin = {"BTC/USD": "bitcoin", "ETH/USD": "ethereum"}[symbol]
    data = _fetch_json(f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
                       {"vs_currency": "usd", "days": "30"})
    vols = {int(t // 3600000): v for t, v in data.get("total_volumes", [])}
    rows = []
    for ts_ms, price in data["prices"]:
        rows.append([int(ts_ms // 1000), price, price, price, price,
                     vols.get(int(ts_ms // 3600000), 0.0)])
    return _std_df(rows)


# ================= price sources =================
def _price_kraken(symbol):
    pair = {"BTC/USD": "XBTUSD", "ETH/USD": "ETHUSD"}[symbol]
    data = _fetch_json("https://api.kraken.com/0/public/Ticker", {"pair": pair})
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return float(next(iter(data["result"].values()))["c"][0])


def _price_binanceus(symbol):
    sym = {"BTC/USD": "BTCUSD", "ETH/USD": "ETHUSD"}[symbol]
    return float(_fetch_json("https://api.binance.us/api/v3/ticker/price",
                             {"symbol": sym})["price"])


def _price_gemini(symbol):
    sym = {"BTC/USD": "btcusd", "ETH/USD": "ethusd"}[symbol]
    return float(_fetch_json(f"https://api.gemini.com/v1/pubticker/{sym}")["last"])


def _price_bitfinex(symbol):
    sym = {"BTC/USD": "tBTCUSD", "ETH/USD": "tETHUSD"}[symbol]
    return float(_fetch_json(f"https://api-pub.bitfinex.com/v2/ticker/{sym}")[6])


def _price_kucoin(symbol):
    sym = {"BTC/USD": "BTC-USDT", "ETH/USD": "ETH-USDT"}[symbol]
    data = _fetch_json("https://api.kucoin.com/api/v1/market/orderbook/level1",
                       {"symbol": sym})
    return float(data["data"]["price"])


def _price_okx(symbol):
    inst = {"BTC/USD": "BTC-USDT", "ETH/USD": "ETH-USDT"}[symbol]
    data = _fetch_json("https://www.okx.com/api/v5/market/ticker", {"instId": inst})
    return float(data["data"][0]["last"])


def _price_bitstamp(symbol):
    pair = {"BTC/USD": "btcusd", "ETH/USD": "ethusd"}[symbol]
    return float(_fetch_json(f"https://www.bitstamp.net/api/v2/ticker/{pair}/")["last"])


def _price_coinbase(symbol):
    prod = {"BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD"}[symbol]
    return float(_fetch_json(
        f"https://api.exchange.coinbase.com/products/{prod}/ticker")["price"])


def _price_coinpaprika(symbol):
    coin = {"BTC/USD": "btc-bitcoin", "ETH/USD": "eth-ethereum"}[symbol]
    data = _fetch_json(f"https://api.coinpaprika.com/v1/tickers/{coin}")
    return float(data["quotes"]["USD"]["price"])


def _price_yahoo(symbol):
    sym = {"BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD"}[symbol]
    data = _fetch_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                       {"interval": "1h", "range": "1d"})
    return float(data["chart"]["result"][0]["meta"]["regularMarketPrice"])


def _price_coingecko(symbol):
    coin = {"BTC/USD": "bitcoin", "ETH/USD": "ethereum"}[symbol]
    data = _fetch_json("https://api.coingecko.com/api/v3/simple/price",
                       {"ids": coin, "vs_currencies": "usd"})
    return float(data[coin]["usd"])


# ================= rotation + caching =================
_OHLC_SOURCES = [
    ("binanceus", _ohlc_binanceus), ("gemini", _ohlc_gemini),
    ("bitfinex", _ohlc_bitfinex), ("kucoin", _ohlc_kucoin),
    ("kraken", _ohlc_kraken), ("okx", _ohlc_okx),
    ("bitstamp", _ohlc_bitstamp), ("coinbase", _ohlc_coinbase),
    ("yahoo", _ohlc_yahoo), ("coingecko", _ohlc_coingecko),
]
_PRICE_SOURCES = [
    ("binanceus", _price_binanceus), ("gemini", _price_gemini),
    ("bitfinex", _price_bitfinex), ("kucoin", _price_kucoin),
    ("kraken", _price_kraken), ("okx", _price_okx),
    ("bitstamp", _price_bitstamp), ("coinbase", _price_coinbase),
    ("coinpaprika", _price_coinpaprika), ("yahoo", _price_yahoo),
    ("coingecko", _price_coingecko),
]

_preferred = {"ohlc": 0, "price": 0}
last_used = {"ohlc": None, "price": None}   # visible in diagnostics


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
            last_used[kind] = name
            return out
        except Exception as e:
            last_err = e
    raise RuntimeError(f"all sources failed, last: {str(last_err)[:120]}")


def get_ohlc(symbol: str, interval: int = 60) -> pd.DataFrame:
    now = time.time()
    with _lock:
        hit = _cache.get(("ohlc", symbol))
        if hit and now - hit[1] < CANDLE_CACHE_SEC:
            return hit[0]
    try:
        df = _try_sources("ohlc", _OHLC_SOURCES, symbol)
        if len(df) < 50:
            raise RuntimeError("too few candles")
        with _lock:
            _cache[("ohlc", symbol)] = (df, now)
        return df
    except Exception:
        with _lock:
            hit = _cache.get(("ohlc", symbol))
        if hit:
            return hit[0]
        raise


def get_price(symbol: str) -> float:
    now = time.time()
    with _lock:
        hit = _cache.get(("price", symbol))
        if hit and now - hit[1] < PRICE_CACHE_SEC:
            return hit[0]
    try:
        price = _try_sources("price", _PRICE_SOURCES, symbol)
        with _lock:
            _cache[("price", symbol)] = (price, now)
        return price
    except Exception:
        with _lock:
            hit = _cache.get(("price", symbol))
        if hit:
            return hit[0]
        raise


def diagnose(symbol: str = "BTC/USD") -> dict:
    """Test every price source — used by /api/diag to see what works
    from wherever the bot is hosted."""
    report = {}
    for name, fn in _PRICE_SOURCES:
        t0 = time.time()
        try:
            price = fn(symbol)
            report[name] = f"OK {price:.0f} ({time.time()-t0:.1f}s)"
        except Exception as e:
            report[name] = f"FAIL {type(e).__name__}: {str(e)[:60]}"
    return report
