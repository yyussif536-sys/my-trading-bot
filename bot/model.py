"""The 'AI' part: a Gradient Boosting classifier that learns from history
whether the NEXT hour's price is likely to go up or down.

Why this model (and not a fancy neural network)?
- Works well on small datasets (Kraken gives us ~720 candles)
- Fast to retrain on the fly
- Battle-tested for tabular data like technical indicators
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from .indicators import build_features

HORIZON = 1          # predict 1 candle ahead
MIN_TRAIN_ROWS = 100


class PricePredictor:
    def __init__(self):
        self.model = None
        self.accuracy = None       # walk-forward validation accuracy
        self.last_trained = None

    def _make_dataset(self, df: pd.DataFrame):
        X = build_features(df)
        future_ret = df["close"].pct_change(HORIZON).shift(-HORIZON)
        y = (future_ret > 0).astype(int)         # 1 = price went up next hour
        valid = X.notna().all(axis=1) & future_ret.notna()
        return X[valid], y[valid]

    def train(self, df: pd.DataFrame):
        """Train on all history; estimate accuracy on the most recent 20%
        (trained only on data BEFORE it — no peeking into the future)."""
        X, y = self._make_dataset(df)
        if len(X) < MIN_TRAIN_ROWS:
            raise ValueError(f"Not enough data: {len(X)} rows")

        split = int(len(X) * 0.8)
        val_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )
        val_model.fit(X.iloc[:split], y.iloc[:split])
        preds = val_model.predict(X.iloc[split:])
        self.accuracy = float((preds == y.iloc[split:]).mean())

        # final model trained on everything
        self.model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )
        self.model.fit(X, y)
        self.last_trained = pd.Timestamp.utcnow()
        return self.accuracy

    def predict(self, df: pd.DataFrame) -> dict:
        """Probability that the next candle closes higher than now."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        X = build_features(df)
        latest = X.iloc[[-1]]
        prob_up = float(self.model.predict_proba(latest)[0][1])
        return {
            "prob_up": prob_up,
            "signal": "BUY" if prob_up >= 0.58 else ("SELL" if prob_up <= 0.42 else "HOLD"),
            "confidence": abs(prob_up - 0.5) * 2,
        }

    def feature_importance(self) -> dict:
        if self.model is None:
            return {}
        names = self.model.feature_names_in_
        return dict(sorted(
            zip(names, self.model.feature_importances_.round(4).tolist()),
            key=lambda kv: -kv[1],
        ))
