"""Paper-trading portfolio: fake money, real prices.
Tracks cash, positions, and full trade history. Saved to disk so it
survives restarts."""
import json
import os
from datetime import datetime, timezone

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "portfolio_state.json")

STARTING_CASH = 10_000.0
FEE_RATE = 0.0026          # 0.26% taker fee, like Kraken's real fee
TRADE_FRACTION = 0.25      # risk at most 25% of portfolio per trade
STOP_LOSS = -0.03          # close position if it drops 3%
TAKE_PROFIT = 0.05         # close position if it gains 5%


class Portfolio:
    def __init__(self):
        self.cash = STARTING_CASH
        self.positions = {}    # symbol -> {"qty": float, "entry_price": float, "opened": iso}
        self.trades = []       # completed + open trade log
        self.load()

    # ---------- persistence ----------
    def load(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                d = json.load(f)
            self.cash = d["cash"]
            self.positions = d["positions"]
            self.trades = d["trades"]

    def save(self):
        with open(STATE_FILE, "w") as f:
            json.dump({"cash": self.cash, "positions": self.positions, "trades": self.trades}, f, indent=2)

    def reset(self):
        self.cash = STARTING_CASH
        self.positions = {}
        self.trades = []
        self.save()

    # ---------- trading ----------
    def buy(self, symbol: str, price: float, reason: str):
        if symbol in self.positions:
            return None  # already long — one position per symbol keeps things simple
        equity = self.value({symbol: price})
        budget = min(self.cash, equity * TRADE_FRACTION)
        if budget < 10:
            return None
        fee = budget * FEE_RATE
        qty = (budget - fee) / price
        self.cash -= budget
        self.positions[symbol] = {
            "qty": qty, "entry_price": price,
            "opened": datetime.now(timezone.utc).isoformat(),
        }
        trade = {
            "time": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "side": "BUY", "price": price, "qty": qty, "fee": fee,
            "reason": reason, "pnl": None,
        }
        self.trades.append(trade)
        self.save()
        return trade

    def sell(self, symbol: str, price: float, reason: str):
        pos = self.positions.get(symbol)
        if not pos:
            return None
        gross = pos["qty"] * price
        fee = gross * FEE_RATE
        cost = pos["qty"] * pos["entry_price"]
        pnl = gross - fee - cost
        self.cash += gross - fee
        del self.positions[symbol]
        trade = {
            "time": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "side": "SELL", "price": price, "qty": pos["qty"], "fee": fee,
            "reason": reason, "pnl": round(pnl, 2),
        }
        self.trades.append(trade)
        self.save()
        return trade

    def check_risk(self, symbol: str, price: float):
        """Automatic stop-loss / take-profit — the bot's safety net."""
        pos = self.positions.get(symbol)
        if not pos:
            return None
        change = price / pos["entry_price"] - 1
        if change <= STOP_LOSS:
            return self.sell(symbol, price, f"STOP-LOSS hit ({change:+.2%})")
        if change >= TAKE_PROFIT:
            return self.sell(symbol, price, f"TAKE-PROFIT hit ({change:+.2%})")
        return None

    # ---------- reporting ----------
    def value(self, prices: dict) -> float:
        total = self.cash
        for sym, pos in self.positions.items():
            total += pos["qty"] * prices.get(sym, pos["entry_price"])
        return total

    def summary(self, prices: dict) -> dict:
        equity = self.value(prices)
        closed = [t for t in self.trades if t["side"] == "SELL"]
        wins = [t for t in closed if (t["pnl"] or 0) > 0]
        open_positions = []
        for sym, pos in self.positions.items():
            cur = prices.get(sym, pos["entry_price"])
            open_positions.append({
                "symbol": sym, "qty": pos["qty"], "entry_price": pos["entry_price"],
                "current_price": cur,
                "unrealized_pnl": round(pos["qty"] * (cur - pos["entry_price"]), 2),
                "change_pct": round((cur / pos["entry_price"] - 1) * 100, 2),
                "opened": pos["opened"],
            })
        return {
            "cash": round(self.cash, 2),
            "equity": round(equity, 2),
            "total_return_pct": round((equity / STARTING_CASH - 1) * 100, 2),
            "open_positions": open_positions,
            "closed_trades": len(closed),
            "win_rate_pct": round(100 * len(wins) / len(closed), 1) if closed else None,
            "realized_pnl": round(sum(t["pnl"] or 0 for t in closed), 2),
        }
