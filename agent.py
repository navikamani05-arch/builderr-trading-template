"""
agent.py — Aggressive Large-Cap Momentum
==========================================
Strategy : Pure momentum — buy the strongest large-cap stocks, hold them,
           rotate when leadership changes.
Universe : Top large-cap US stocks by liquidity + broad ETFs as risk filter.
Aggression: Aggressive — higher concentration, faster signals, less cash drag.

HOW IT WORKS — three layers:

  Layer 1 — RISK-OFF SWITCH
    If SPY is below its 50-day SMA → go to cash entirely.
    Aggressive doesn't mean reckless; one clean exit rule keeps drawdowns
    from killing your Calmar. 50-day (not 20-day) gives fewer whipsaws
    since we're holding individual stocks which are noisier than ETFs.

  Layer 2 — CROSS-SECTIONAL MOMENTUM
    Score every large-cap in our watchlist by blended momentum:
      50% × 3-month return  +  50% × 1-month return
    Also require the stock is above its own 50-day SMA (trend confirmation).
    Pick the top TOP_N winners with positive scores.
    This is classic cross-sectional momentum — backed by decades of evidence
    as the strongest factor in large-cap equities.

  Layer 3 — AGGRESSIVE SIZING
    Equal weight across winners, but sized up to MAX_POSITION (28%) per name.
    No vol-parity here — aggressive means leaning into conviction.
    Hard leverage cap at 1.45× (rule = 1.50×), all stocks count 1×.
    Max 5 positions to keep concentration high and signal clean.

No lookahead. Long-only. Gross leverage ≤ 1.45×. ≤ 28% per name.
"""

from __future__ import annotations

import math
from typing import Any

LARGE_CAPS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "AVGO",
    "MU", "QCOM", "MRVL", "AMAT", "LRCX",
    "JPM", "V", "MA", "GS", "MS",
    "UNH", "LLY", "ABBV", "JNJ",
    "XOM", "CVX",
    "COST", "HD", "NFLX", "CRM", "PLTR",
]

BENCHMARK = "SPY"
MIN_PRICE = 10.0

TREND_WINDOW = 50
STOCK_TREND_WIN = 50
MOM_LONG = 63
MOM_SHORT = 21
MOM_LONG_WT = 0.50
MOM_SHORT_WT = 0.50
TOP_N = 5
MAX_POSITION = 0.28
MIN_BARS = 70
GROSS_CAP = 1.40
REBAL_THRESHOLD = 0.03


def _closes(bars: list[dict]) -> list[float]:
    return [float(b["close"]) for b in bars]


def _sma(prices: list[float], window: int) -> float | None:
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window


def _momentum(prices: list[float], lookback: int) -> float | None:
    if len(prices) < lookback + 1:
        return None
    base = prices[-(lookback + 1)]
    if base <= 0:
        return None
    return (prices[-1] / base) - 1.0


def _equity(portfolio_state: dict, cash: float) -> float:
    last_prices = portfolio_state.get("last_prices", {})
    positions = portfolio_state.get("positions", [])
    holdings_val = sum(
        p["quantity"] * last_prices.get(p["ticker"], p.get("avg_cost", 0))
        for p in positions
    )
    return cash + holdings_val


def _current_holdings(portfolio_state: dict) -> dict[str, int]:
    return {p["ticker"]: int(p["quantity"]) for p in portfolio_state.get("positions", [])}


def _get_price(ticker: str, portfolio_state: dict, market_state: dict) -> float | None:
    price = portfolio_state.get("last_prices", {}).get(ticker)
    if not price or price <= 0:
        bars = market_state.get(ticker, [])
        price = bars[-1]["close"] if bars else None
    return float(price) if price and price > 0 else None


def _liquidate_all(holdings: dict[str, int]) -> list[dict]:
    return [
        {"ticker": t, "side": "sell", "quantity": q}
        for t, q in holdings.items()
        if q > 0
    ]


def decide(
    market_state: dict[str, list[dict]],
    portfolio_state: dict[str, Any],
    cash: float,
) -> list[dict]:
    spy_bars = market_state.get(BENCHMARK, [])
    if len(spy_bars) < MIN_BARS:
        return []

    spy_closes = _closes(spy_bars)
    holdings = _current_holdings(portfolio_state)
    equity = _equity(portfolio_state, cash)

    spy_sma = _sma(spy_closes, TREND_WINDOW)
    risk_on = (spy_sma is not None) and (spy_closes[-1] > spy_sma)

    if not risk_on:
        return _liquidate_all(holdings)

    scores: dict[str, float] = {}

    for ticker in LARGE_CAPS:
        bars = market_state.get(ticker, [])
        if not bars or len(bars) < MOM_LONG + 2:
            continue

        closes = _closes(bars)
        price = closes[-1]

        if price < MIN_PRICE:
            continue

        stock_sma = _sma(closes, STOCK_TREND_WIN)
        if stock_sma is None or price < stock_sma:
            continue

        mom_long = _momentum(closes, MOM_LONG)
        mom_short = _momentum(closes, MOM_SHORT)

        if mom_long is None or mom_short is None:
            continue

        blend = MOM_LONG_WT * mom_long + MOM_SHORT_WT * mom_short

        if blend > 0:
            scores[ticker] = blend

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winners = [t for t, _ in ranked[:TOP_N]]

    if not winners:
        return _liquidate_all(holdings)

    n = len(winners)
    raw_weight = min(1.0 / n, MAX_POSITION)

    total_gross = raw_weight * n
    if total_gross > GROSS_CAP:
        raw_weight = GROSS_CAP / n

    targets = {ticker: raw_weight for ticker in winners}

    orders: list[dict] = []

    for ticker, qty in holdings.items():
        if ticker not in winners and qty > 0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": qty})

    for ticker, weight in targets.items():
        price = _get_price(ticker, portfolio_state, market_state)
        if not price:
            continue

        target_qty = int((equity * weight) / price)
        current_qty = holdings.get(ticker, 0)
        diff = target_qty - current_qty

        current_weight = (current_qty * price) / equity if equity > 0 else 0
        drift = abs(current_weight - weight)

        if diff > 0 and (current_qty == 0 or drift > REBAL_THRESHOLD):
            orders.append({"ticker": ticker, "side": "buy", "quantity": diff})
        elif diff < 0 and drift > REBAL_THRESHOLD:
            orders.append({"ticker": ticker, "side": "sell", "quantity": abs(diff)})

    return orders
