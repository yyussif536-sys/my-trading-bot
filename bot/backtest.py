"""Walk-forward backtest: replay history candle by candle.
At each step the model only sees data up to that point (no cheating),
then we simulate the trade it would have made."""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from .indicators import build_features
from .portfolio import FEE_RATE, STOP_LOSS, TAKE_PROFIT

TRAIN_WINDOW = 400      # candles used for each training
RETRAIN_EVERY = 24      # retrain once per simulated day (hourly candles)
BUY_THRESHOLD = 0.58
SELL_THRESHOLD = 0.42


def run_backtest(df: pd.DataFrame, starting_cash: float = 10_000.0) -> dict:
    X_all = build_features(df)
    close = df["close"]

    cash = starting_cash
    qty = 0.0
    entry = 0.0
    trades = []
    equity_curve = []
    model = None

    start = TRAIN_WINDOW
    for i in range(start, len(df) - 1):
        price = close.iloc[i]

        # retrain periodically on a sliding window of the past
        if model is None or (i - start) % RETRAIN_EVERY == 0:
            lo = max(0, i - TRAIN_WINDOW)
            X = X_all.iloc[lo:i]
            future_ret = close.pct_change().shift(-1).iloc[lo:i]
            y = (future_ret > 0).astype(int)
            valid = X.notna().all(axis=1) & future_ret.notna()
            if valid.sum() > 50:
                model = GradientBoostingClassifier(
                    n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42
                )
                model.fit(X[valid], y[valid])

        if model is None or X_all.iloc[i].isna().any():
            equity_curve.append({"time": str(df["time"].iloc[i]), "equity": cash + qty * price})
            continue

        prob_up = model.predict_proba(X_all.iloc[[i]])[0][1]

        # risk management first
        if qty > 0:
            change = price / entry - 1
            if change <= STOP_LOSS or change >= TAKE_PROFIT:
                gross = qty * price
                cash += gross * (1 - FEE_RATE)
                trades.append({"time": str(df["time"].iloc[i]), "side": "SELL", "price": price,
                               "pnl": round(gross * (1 - FEE_RATE) - qty * entry, 2),
                               "reason": "stop/take"})
                qty = 0.0

        # signals
        if qty == 0 and prob_up >= BUY_THRESHOLD:
            budget = cash * 0.25
            if budget > 10:
                qty = budget * (1 - FEE_RATE) / price
                entry = price
                cash -= budget
                trades.append({"time": str(df["time"].iloc[i]), "side": "BUY",
                               "price": price, "pnl": None, "reason": f"prob_up={prob_up:.2f}"})
        elif qty > 0 and prob_up <= SELL_THRESHOLD:
            gross = qty * price
            cash += gross * (1 - FEE_RATE)
            trades.append({"time": str(df["time"].iloc[i]), "side": "SELL", "price": price,
                           "pnl": round(gross * (1 - FEE_RATE) - qty * entry, 2),
                           "reason": f"prob_up={prob_up:.2f}"})
            qty = 0.0

        equity_curve.append({"time": str(df["time"].iloc[i]), "equity": round(cash + qty * price, 2)})

    # close any open position at the end
    if qty > 0:
        price = close.iloc[-1]
        gross = qty * price
        cash += gross * (1 - FEE_RATE)
        trades.append({"time": str(df["time"].iloc[-1]), "side": "SELL", "price": price,
                       "pnl": round(gross * (1 - FEE_RATE) - qty * entry, 2), "reason": "end of test"})

    final = cash
    closed = [t for t in trades if t["side"] == "SELL"]
    wins = [t for t in closed if t["pnl"] > 0]

    # compare against just buying and holding
    bh_return = (close.iloc[-1] / close.iloc[start] - 1) * 100

    equities = [p["equity"] for p in equity_curve]
    peak, max_dd = equities[0] if equities else 0, 0.0
    for e in equities:
        peak = max(peak, e)
        max_dd = min(max_dd, e / peak - 1)

    return {
        "final_equity": round(float(final), 2),
        "total_return_pct": round(float(final / starting_cash - 1) * 100, 2),
        "buy_and_hold_pct": round(float(bh_return), 2),
        "num_trades": len(closed),
        "win_rate_pct": round(100 * len(wins) / len(closed), 1) if closed else None,
        "max_drawdown_pct": round(float(max_dd) * 100, 2),
        "trades": trades[-40:],
        "equity_curve": equity_curve,
    }
