"""
Trading bot skeleton: Alpaca (execution + data) + Claude (signal reasoning) + Railway (hosting).

Design principle: Claude never places an order directly. It returns a structured
opinion; a separate deterministic risk layer decides whether/how to act on it.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone

import anthropic
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("trading-bot")

# ---------- Config (all from env vars — set these in Railway) ----------
ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_PAPER = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

WATCHLIST = os.environ.get("WATCHLIST", "AAPL,MSFT").split(",")
MAX_POSITION_PCT = float(os.environ.get("MAX_POSITION_PCT", "0.05"))   # 5% of equity per name
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.65"))       # ignore low-confidence reads
DAILY_LOSS_LIMIT_PCT = float(os.environ.get("DAILY_LOSS_LIMIT_PCT", "0.03"))  # kill switch
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "900"))              # 15 min default

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ---------- Market data ----------
def get_snapshot(symbol: str) -> dict:
    quote_req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    quote = data_client.get_stock_latest_quote(quote_req)[symbol]

    bars_req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        limit=10,
    )
    bars = data_client.get_stock_bars(bars_req).df

    closes = bars["close"].tolist() if not bars.empty else []
    return {
        "symbol": symbol,
        "bid": quote.bid_price,
        "ask": quote.ask_price,
        "last_10_closes": closes,
    }


# ---------- Claude: structured signal, not a trade order ----------
DECISION_SCHEMA_PROMPT = """You are a market analysis assistant. Given the data below, \
return ONLY a JSON object, no other text, with this exact shape:
{"action": "buy" | "sell" | "hold", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}

This is one input into a system with its own independent risk controls. \
You are not placing a trade — just giving a structured read."""


def get_claude_signal(snapshot: dict) -> dict:
    message = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=DECISION_SCHEMA_PROMPT,
        messages=[{"role": "user", "content": json.dumps(snapshot)}],
    )
    text = message.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("Could not parse Claude response, defaulting to hold: %s", text)
        return {"action": "hold", "confidence": 0.0, "reasoning": "parse_error"}


# ---------- Deterministic risk layer ----------
def daily_kill_switch_tripped(account) -> bool:
    equity = float(account.equity)
    last_equity = float(account.last_equity)
    if last_equity == 0:
        return False
    day_pl_pct = (equity - last_equity) / last_equity
    if day_pl_pct <= -DAILY_LOSS_LIMIT_PCT:
        log.error("Daily loss limit hit (%.2f%%). Halting trading.", day_pl_pct * 100)
        return True
    return False


def existing_position_qty(symbol: str) -> float:
    try:
        pos = trading_client.get_open_position(symbol)
        return float(pos.qty)
    except Exception:
        return 0.0


def size_order(symbol: str, snapshot: dict, account) -> int:
    equity = float(account.equity)
    max_dollars = equity * MAX_POSITION_PCT
    price = snapshot["ask"] or snapshot["bid"]
    if not price:
        return 0
    return max(int(max_dollars // price), 0)


def execute_if_allowed(symbol: str, signal: dict, snapshot: dict, account):
    if signal["confidence"] < MIN_CONFIDENCE:
        log.info("%s: confidence %.2f below threshold, skipping", symbol, signal["confidence"])
        return

    action = signal["action"]
    if action == "hold":
        return

    qty_held = existing_position_qty(symbol)

    if action == "buy" and qty_held > 0:
        log.info("%s: already holding a position, skipping buy", symbol)
        return
    if action == "sell" and qty_held <= 0:
        log.info("%s: no position to sell, skipping", symbol)
        return

    if action == "buy":
        qty = size_order(symbol, snapshot, account)
        if qty <= 0:
            log.info("%s: computed order size is 0, skipping", symbol)
            return
        order = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
    else:  # sell
        order = MarketOrderRequest(symbol=symbol, qty=qty_held, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)

    log.info("Submitting %s order for %s: %s", action, symbol, signal["reasoning"])
    trading_client.submit_order(order)


# ---------- Main loop ----------
def run_cycle():
    account = trading_client.get_account()
    if daily_kill_switch_tripped(account):
        return

    for symbol in WATCHLIST:
        try:
            snapshot = get_snapshot(symbol)
            signal = get_claude_signal(snapshot)
            log.info("%s -> %s (conf %.2f): %s", symbol, signal["action"], signal["confidence"], signal["reasoning"])
            execute_if_allowed(symbol, signal, snapshot, account)
        except Exception as e:
            log.exception("Error processing %s: %s", symbol, e)


if __name__ == "__main__":
    log.info("Starting trading bot. Paper trading: %s", ALPACA_PAPER)
    while True:
        run_cycle()
        time.sleep(POLL_SECONDS)
