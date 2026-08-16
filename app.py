"""AI Crypto Paper-Trading Bot — web dashboard + trading loop.

Run:  python app.py   then open the dashboard in your browser.

The bot:
 1. pulls hourly BTC/USD candles from Kraken (free, no API key)
 2. trains a Gradient Boosting model to predict the next hour's direction
 3. paper-trades $10,000 of fake money with stop-loss / take-profit
"""
import os
import threading
import time
import traceback
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

from bot.data import get_ohlc, get_price
from bot.model import PricePredictor
from bot.portfolio import Portfolio
from bot.backtest import run_backtest

SYMBOL = "BTC/USD"
LOOP_SECONDS = 60          # check the market every minute
RETRAIN_MINUTES = 60       # retrain the model every hour (new candle)

app = Flask(__name__)
predictor = PricePredictor()
portfolio = Portfolio()

state = {
    "status": "starting",
    "last_update": None,
    "price": None,
    "prediction": None,
    "model_accuracy": None,
    "last_trained": None,
    "log": [],           # recent bot decisions
    "auto_trade": True,
    "error": None,
}
_lock = threading.Lock()


def log(msg: str):
    entry = {"time": datetime.now(timezone.utc).strftime("%H:%M:%S"), "msg": msg}
    with _lock:
        state["log"].insert(0, entry)
        state["log"] = state["log"][:300]
    print(f"[{entry['time']}] {msg}", flush=True)


def bot_loop():
    log("🤖 Robot started watching the market...")
    last_train = 0.0
    while True:
        try:
            df = get_ohlc(SYMBOL, interval=60)
            price = get_price(SYMBOL)

            if time.time() - last_train > RETRAIN_MINUTES * 60 or predictor.model is None:
                acc = predictor.train(df)
                last_train = time.time()
                log(f"Model retrained on {len(df)} candles — validation accuracy {acc:.1%}")

            pred = predictor.predict(df)

            with _lock:
                state.update({
                    "status": "running",
                    "price": price,
                    "prediction": pred,
                    "model_accuracy": predictor.accuracy,
                    "last_trained": str(predictor.last_trained),
                    "last_update": datetime.now(timezone.utc).isoformat(),
                    "error": None,
                })

            # safety net always runs
            risk_trade = portfolio.check_risk(SYMBOL, price)
            if risk_trade:
                log(f"⚠️ {risk_trade['reason']} — sold {risk_trade['qty']:.6f} BTC "
                    f"at ${price:,.0f} (PnL ${risk_trade['pnl']:+,.2f})")

            if state["auto_trade"]:
                if pred["signal"] == "BUY":
                    t = portfolio.buy(SYMBOL, price, f"AI: prob_up={pred['prob_up']:.2f}")
                    if t:
                        log(f"🟢 BUY {t['qty']:.6f} BTC at ${price:,.0f} "
                            f"(model says {pred['prob_up']:.0%} chance of rising)")
                elif pred["signal"] == "SELL":
                    t = portfolio.sell(SYMBOL, price, f"AI: prob_up={pred['prob_up']:.2f}")
                    if t:
                        log(f"🔴 SELL {t['qty']:.6f} BTC at ${price:,.0f} "
                            f"(PnL ${t['pnl']:+,.2f})")

        except Exception as e:
            with _lock:
                state["status"] = "error"
                state["error"] = str(e)
            log(f"Error: {e}")
            traceback.print_exc()

        time.sleep(LOOP_SECONDS)


# ---------------- API ----------------
@app.route("/")
def index():
    return render_template("money.html")     # simple money-focused site


@app.route("/pro")
def pro():
    return render_template("index.html")     # advanced/technical dashboard


@app.route("/api/status")
def api_status():
    price = state["price"]
    prices = {SYMBOL: price} if price else {}

    # current indicator values — exactly what the AI sees right now
    indicators_now = {}
    try:
        from bot.indicators import build_features
        df = get_ohlc(SYMBOL, interval=60)
        latest = build_features(df).iloc[-1]
        indicators_now = {k: (None if v != v else round(float(v), 5)) for k, v in latest.items()}
    except Exception:
        pass

    with _lock:
        return jsonify({
            **{k: v for k, v in state.items() if k != "log"},
            "log": state["log"][:200],
            "portfolio": portfolio.summary(prices),
            "trades": portfolio.trades[::-1],          # FULL trade history
            "feature_importance": predictor.feature_importance(),
            "indicators_now": indicators_now,
            "symbol": SYMBOL,
        })


@app.route("/api/candles")
def api_candles():
    df = get_ohlc(SYMBOL, interval=60)
    return jsonify(df.tail(200).assign(time=df["time"].astype(str)).to_dict("records"))


@app.route("/api/toggle", methods=["POST"])
def api_toggle():
    with _lock:
        state["auto_trade"] = not state["auto_trade"]
    log(f"Auto-trading {'ENABLED' if state['auto_trade'] else 'PAUSED'}")
    return jsonify({"auto_trade": state["auto_trade"]})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    portfolio.reset()
    log("Portfolio reset to $10,000")
    return jsonify({"ok": True})


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    df = get_ohlc(SYMBOL, interval=60)
    result = run_backtest(df)
    log(f"Backtest done: {result['total_return_pct']:+.2f}% over {result['num_trades']} trades "
        f"(buy & hold: {result['buy_and_hold_pct']:+.2f}%)")
    return jsonify(result)


@app.route("/ping")
def ping():
    """Keep-alive endpoint — an external service pings this every 5 minutes
    so the free host never puts the bot to sleep."""
    return jsonify({"ok": True, "status": state["status"], "time": datetime.now(timezone.utc).isoformat()})


@app.route("/api/diag")
def api_diag():
    """Test every data source from this server — shows which ones work here."""
    from bot.data import diagnose, last_used
    return jsonify({"sources": diagnose(SYMBOL), "currently_using": last_used})


# Start the trading loop — self-healing version.
# If the loop thread ever dies or didn't start in this process (which can
# happen on hosting services after sleep/restart), any web request revives it.
_loop_thread = None
_loop_pid = None
_loop_lock = threading.Lock()


def start_bot_once():
    global _loop_thread, _loop_pid
    with _loop_lock:
        alive = _loop_thread is not None and _loop_thread.is_alive() and _loop_pid == os.getpid()
        if not alive:
            _loop_pid = os.getpid()
            _loop_thread = threading.Thread(target=bot_loop, daemon=True)
            _loop_thread.start()


@app.before_request
def _ensure_loop_running():
    start_bot_once()


start_bot_once()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
