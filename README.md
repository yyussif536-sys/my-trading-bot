# 🤖 AI Crypto Trading Bot (Paper Trading)

A beginner-friendly AI trading bot that paper-trades **BTC/USD** with **$10,000 of virtual money** using real, live market prices from Kraken (free public API — no account or keys needed).

> ⚠️ **Educational only.** No real money is ever at risk. Simulated results never guarantee real profits. Crypto is extremely volatile.

## How the AI works

1. **Data** — pulls ~720 hourly BTC/USD candles from Kraken (`bot/data.py`)
2. **Features** — computes 11 technical indicators: RSI, EMA crossovers, MACD, Bollinger %B, recent returns, volatility, volume (`bot/indicators.py`)
3. **Model** — a **Gradient Boosting classifier** (scikit-learn) learns from history to estimate the probability the *next hour's* price closes higher (`bot/model.py`)
4. **Signals** —
   - probability ≥ 58% → **BUY** (invests 25% of the portfolio)
   - probability ≤ 42% → **SELL** (closes the position)
   - in between → **HOLD**
5. **Risk management** — every position has an automatic **3% stop-loss** and **5% take-profit**, plus realistic 0.26% trading fees
6. **Retraining** — the model retrains itself every hour as new candles arrive

## The dashboard

- Live BTC price, AI signal, and the model's confidence
- Portfolio value, return, win rate, trade history
- Which indicators the model relies on most (feature importance)
- **📊 Run backtest** — replays the last ~30 days candle-by-candle (walk-forward, no look-ahead cheating) and compares the strategy vs. simply buying & holding
- Pause/resume auto-trading, reset the portfolio anytime

## Run it yourself

```bash
pip install flask scikit-learn pandas requests
cd trading-bot
python app.py        # open http://localhost:5000
```

## Project structure

```
trading-bot/
├── app.py               # Flask web server + trading loop
├── templates/index.html # dashboard UI
└── bot/
    ├── data.py          # Kraken market data
    ├── indicators.py    # technical indicators
    ├── model.py         # ML model (the "AI")
    ├── portfolio.py     # paper-trading wallet + risk rules
    └── backtest.py      # walk-forward backtester
```

## Honest expectations 📉

Predicting markets is *hard*. A validation accuracy of ~55–58% is typical — barely better than a coin flip, and fees eat into small edges. Our own backtest often **loses to buy-and-hold**. That's the most valuable lesson this bot teaches: test everything before risking real money. Use this project to learn how trading systems, ML features, and risk management work.

## Ideas to extend it

- Add ETH/USD or more coins (edit `PAIRS` in `bot/data.py` and `SYMBOL` in `app.py`)
- Try different thresholds, stop-loss/take-profit levels, or trade sizes
- Add more features: funding rates, order-book data, news sentiment
- Try other models: RandomForest, XGBoost, LSTM
- Connect a real exchange's **testnet** API for true paper-order execution
