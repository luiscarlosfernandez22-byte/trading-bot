"""
Options trading bot skeleton: long calls/puts only.

Same pattern as the stock bot: Claude gives a directional read (bullish/bearish/
neutral + confidence), never touches an order directly. Deterministic code picks
the actual contract and enforces risk limits.

Long options can lose 100% of premium, and lose value every day from theta decay
even if you're right on direction but wrong on timing. This skeleton adds two
protections the stock bot didn't need:
  - a hard per-position stop-loss (sell automatically if the contract drops X%)
  - a forced close before expiration, regardless of any signal

Requires your Alpaca account to have options trading enabled (any level covers
long calls/puts) and paper trading enabled for testing.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone

import anthropic
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    GetOptionContractsRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, ContractType, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest, OptionLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("options-bot")

# ---------- Config ----------
ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_PAPER = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

WATCHLIST = os.environ.get("WATCHLIST", "AAPL,MSFT").split(",")
MAX_PREMIUM_PCT = float(os.environ.get("MAX_PREMIUM_PCT", "0.02"))     # 2% of equity per trade, max
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.70"))       # higher bar than stocks
STOP_LOSS_PCT = float(os.environ.get("OPTION_STOP_LOSS_PCT", "0.50"))  # close if contract drops 50%
MIN_DAYS_TO_EXPIRY = int(os.environ.get("MIN_DAYS_TO_EXPIRY", "30"))
MAX_DAYS_TO_EXPIRY = int(os.environ.get("MAX_DAYS_TO_EXPIRY", "45"))
FORCE_CLOSE_DTE = int(os.environ.get("FORCE_CLOSE_DTE", "5"))          # close no matter what inside this window
DAILY_LOSS_LIMIT_PCT = float(os.environ.get("DAILY_LOSS_LIMIT_PCT", "0.03"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "900"))

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
stock_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
option_data_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ---------- Underlying market data ----------
def get_stock_snapshot(symbol: str) -> dict:
    quote = stock_data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
    bars = stock_data_client.get_stock_bars(
        StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, limit=10)
    ).df
    closes = bars["close"].tolist() if not bars.empty else []
    return {"symbol": symbol, "bid": quote.bid_price, "ask": quote.ask_price, "last_10_closes": closes}


# ---------- Claude: directional read only, never a trade ----------
DIRECTION_PROMPT = """You are a market analysis assistant. Given the data below, return \
ONLY a JSON object, no other text:
{"direction": "bullish" | "bearish" | "neutral", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}

This feeds a system with its own independent contract selection and risk controls. \
You are not selecting a contract or placing a trade — just giving a directional read."""


def get_direction_signal(snapshot: dict) -> dict:
    message = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=DIRECTION_PROMPT,
        messages=[{"role": "user", "content": json.dumps(snapshot)}],
    )
    text = message.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("Could not parse Claude response, defaulting to neutral: %s", text)
        return {"direction": "neutral", "confidence": 0.0, "reasoning": "parse_error"}


# ---------- Contract selection (deterministic, not Claude) ----------
def pick_contract(symbol: str, direction: str, underlying_price: float):
    contract_type = ContractType.CALL if direction == "bullish" else ContractType.PUT

    min_exp = (datetime.now(timezone.utc) + timedelta(days=MIN_DAYS_TO_EXPIRY)).date()
    max_exp = (datetime.now(timezone.utc) + timedelta(days=MAX_DAYS_TO_EXPIRY)).date()

    req = GetOptionContractsRequest(
        underlying_symbols=[symbol],
        status=AssetStatus.ACTIVE,
        type=contract_type,
        expiration_date_gte=min_exp,
        expiration_date_lte=max_exp,
    )
    contracts = trading_client.get_option_contracts(req).option_contracts
    if not contracts:
        return None

    # Pick the strike closest to at-the-money
    closest = min(contracts, key=lambda c: abs(float(c.strike_price) - underlying_price))
    return closest


def days_to_expiry(contract) -> int:
    exp = datetime.strptime(str(contract.expiration_date), "%Y-%m-%d").date()
    return (exp - datetime.now(timezone.utc).date()).days


# ---------- Risk layer ----------
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


def manage_open_positions():
    """Close any option position that's hit its stop-loss or is close to expiring, no signal needed."""
    positions = trading_client.get_all_positions()
    for pos in positions:
        if pos.asset_class != "us_option":
            continue

        unrealized_pct = float(pos.unrealized_plpc)  # already a fraction, e.g. -0.5 = -50%
        if unrealized_pct <= -STOP_LOSS_PCT:
            log.warning("%s hit stop-loss (%.1f%%), closing", pos.symbol, unrealized_pct * 100)
            trading_client.close_position(pos.symbol, ClosePositionRequest(percentage="100"))
            continue

        try:
            contracts = trading_client.get_option_contracts(
                GetOptionContractsRequest(symbol=pos.symbol)
            ).option_contracts
            if contracts and days_to_expiry(contracts[0]) <= FORCE_CLOSE_DTE:
                log.warning("%s within %d days of expiry, closing", pos.symbol, FORCE_CLOSE_DTE)
                trading_client.close_position(pos.symbol, ClosePositionRequest(percentage="100"))
        except Exception as e:
            log.exception("Error checking expiry for %s: %s", pos.symbol, e)


def already_holding(symbol: str) -> bool:
    positions = trading_client.get_all_positions()
    return any(p.symbol.startswith(symbol) and p.asset_class == "us_option" for p in positions)


def size_contracts(account, ask_price: float) -> int:
    if not ask_price or ask_price <= 0:
        return 0
    equity = float(account.equity)
    max_dollars = equity * MAX_PREMIUM_PCT
    cost_per_contract = ask_price * 100  # options are quoted per share, contracts are 100 shares
    return max(int(max_dollars // cost_per_contract), 0)


def execute_if_allowed(symbol: str, signal: dict, snapshot: dict, account):
    if signal["direction"] == "neutral" or signal["confidence"] < MIN_CONFIDENCE:
        log.info("%s: no trade (direction=%s, confidence=%.2f)", symbol, signal["direction"], signal["confidence"])
        return

    if already_holding(symbol):
        log.info("%s: already holding an option position, skipping", symbol)
        return

    underlying_price = snapshot["ask"] or snapshot["bid"]
    contract = pick_contract(symbol, signal["direction"], underlying_price)
    if contract is None:
        log.info("%s: no suitable contract found in expiry window", symbol)
        return

    quote = option_data_client.get_option_latest_quote(
        OptionLatestQuoteRequest(symbol_or_symbols=contract.symbol)
    )[contract.symbol]
    ask_price = quote.ask_price
    qty = size_contracts(account, ask_price)
    if qty <= 0:
        log.info("%s: computed contract quantity is 0, skipping", symbol)
        return

    log.info(
        "Buying %d contract(s) of %s (%s, strike %s, exp %s): %s",
        qty, contract.symbol, signal["direction"], contract.strike_price, contract.expiration_date, signal["reasoning"],
    )
    order = MarketOrderRequest(symbol=contract.symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
    trading_client.submit_order(order)


# ---------- Main loop ----------
def run_cycle():
    account = trading_client.get_account()
    if daily_kill_switch_tripped(account):
        return

    manage_open_positions()  # stop-losses and forced expiry closes happen every cycle, independent of signals

    for symbol in WATCHLIST:
        try:
            snapshot = get_stock_snapshot(symbol)
            signal = get_direction_signal(snapshot)
            log.info("%s -> %s (conf %.2f): %s", symbol, signal["direction"], signal["confidence"], signal["reasoning"])
            execute_if_allowed(symbol, signal, snapshot, account)
        except Exception as e:
            log.exception("Error processing %s: %s", symbol, e)


if __name__ == "__main__":
    log.info("Starting options bot (long calls/puts only). Paper trading: %s", ALPACA_PAPER)
    while True:
        run_cycle()
        time.sleep(POLL_SECONDS)
