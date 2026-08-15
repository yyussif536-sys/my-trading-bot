"""Technical indicators computed with pure pandas (no extra libraries needed)."""
import pandas as pd
import numpy as np


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index: 0-100. Below 30 = oversold, above 70 = overbought."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return close.ewm(span=period, adjust=False).mean()


def macd(close: pd.Series):
    """MACD line, signal line, histogram."""
    macd_line = ema(close, 12) - ema(close, 26)
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger(close: pd.Series, period: int = 20):
    """%B: where price sits inside the Bollinger bands (0 = lower band, 1 = upper)."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper, lower = mid + 2 * std, mid - 2 * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pct_b.fillna(0.5)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw candles into the feature table the ML model learns from."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    out["rsi"] = rsi(close)
    out["ema_fast_ratio"] = close / ema(close, 9) - 1        # price vs fast EMA
    out["ema_slow_ratio"] = close / ema(close, 21) - 1       # price vs slow EMA
    out["ema_cross"] = ema(close, 9) / ema(close, 21) - 1    # fast vs slow EMA
    macd_line, signal_line, hist = macd(close)
    out["macd_hist"] = hist / close                          # normalized
    out["boll_pct_b"] = bollinger(close)
    out["ret_1"] = close.pct_change(1)                       # recent returns
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)
    out["volatility"] = close.pct_change().rolling(12).std()
    out["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    return out
