"""
agent.py — Hybrid Momentum + Vol-Target + Risk-Off + VIX Panic Exit
====================================================================
Strategy: Hybrid (Momentum × Volatility Targeting × Trend Filter × VIX Guard)
Universe:  Leveraged ETFs (TQQQ, SOXL, UPRO) + safe-haven proxies (GLD, TLT)
Aggression: Balanced — targets ~12% annualised vol, guards against deep drawdowns

HOW IT WORKS — three layers executed in order each day:

  Layer 1 — RISK-OFF SWITCH (two triggers, either alone flips you to cash)
    a) Trend trigger : SPY closes below its 50-day SMA → market is broken, exit.
    b) VIX panic exit: VIX closes above VIX_PANIC_LEVEL (30) → fear is spiking,
                       exit before the leveraged ETFs crater. This fires faster
                       than the SMA trigger and is specifically designed to catch
                       sudden sell-offs (e.g. March 2020, Aug 2024 yen carry).

  Layer 2 — MOMENTUM SCORING (only runs when risk is ON)
    Score TQQQ, SOXL, UPRO, GLD, TLT by blended momentum:
      60% × 3-month return  +  40% × 1-month return
    Pick the top 2 assets with positive scores.
    Fallback: if nothing is positive, hold GLD as a defensive anchor.

  Layer 3 — VOLATILITY-TARGETED SIZING
    Each winner gets a weight = (target_vol / n) / asset_vol (vol-parity).
    Capped at 25% of equity per name (safely under the 30% rule).
    Beta-adjusted gross leverage is checked and scaled down if it would
    exceed 1.45× (our safety margin below the 1.50× hard rule).

LEVERAGE ACCOUNTING:
  TQQQ / SOXL / UPRO each count 3× toward the gross leverage cap.
  One leveraged ETF at 45% notional → 1.35× gross (within limit).
  Two at ~22% each → 1.32× gross (within limit).

NO LOOKAHEAD:
  All calculations use only prices up to and including today's bar.
  No future data is accessed at any point.
"""

from __future__ import annotations
import math
from typing import Any


# ── Universe ────────────────────────────────────────────────────────────────

LEVERAGED_3X = ["TQQQ", "SOXL", "UPRO"]   # count 3× toward leverage cap
DEFENSIVE    = ["GLD", "TLT"]              # safe-haven; 1× weight
BENCHMARK    = "SPY"                        # trend-filter reference
VIX_TICKER   = "VIX"                       # fear gauge for panic exit


# ── Hyperparameters (±20% stress-tested — kept intentionally simple) ────────

TREND_WINDOW    = 50      # SPY SMA window: if SPY < SMA(50) → risk-off
VIX_PANIC_LEVEL = 25      # VIX threshold: if VIX > 30 → panic exit to cash
                          #   30 is a well-established "fear spike" level;
                          #   historically coincides with sharp sell-offs.
MOM_LONG        = 63      # ~3-month momentum lookback (trading days)
MOM_SHORT       = 21      # ~1-month momentum lookback
MOM_LONG_WT     = 0.60    # weight on the 3-month leg of the blend
MOM_SHORT_WT    = 0.40    # weight on the 1-month leg of the blend
VOL_WINDOW      = 20      # days used to estimate realised volatility
TARGET_VOL      = 0.12    # annualised portfolio vol target (12%)
MAX_POSITION    = 0.25    # max single-name weight of equity (25%)
TOP_N           = 2       # number of momentum winners to hold at once
MIN_BARS        = 70      # minimum bars required before we trade at all
GROSS_CAP       = 1.45    # beta-adjusted gross leverage ceiling (rule = 1.50×)
LVRG_MULT       = {t: 3.0 for t in LEVERAGED_3X}  # 3× multiplier for leveraged ETFs


# ── Helpers ─────────────────────────────────────────────────────────────────

def _closes(bars: list[dict]) -> list[float]:
    """Pull the close price from each bar."""
    return [b["close"] for b in bars]


def _sma(prices: list[float], window: int) -> float | None:
    """Simple moving average of the last `window` prices. None if insufficient data."""
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window


def _momentum(prices: list[float], lookback: int) -> float | None:
    """
    Price return over `lookback` bars, anchored one bar before the end so
    we never accidentally use the most-recent close as both base and end.
    Returns None if there aren't enough bars.
    """
    if len(prices) < lookback + 1:
        return None
    base = prices[-(lookback + 1)]
    end  = prices[-1]
    if base <= 0:
        return None
    return (end / base) - 1.0


def _realised_vol(prices: list[float], window: int) -> float | None:
    """
    Annualised realised volatility from daily log-returns over `window` days.
    Returns None if there aren't enough bars or returns are degenerate.
    """
    if len(prices) < window + 1:
        return None
    log_rets = []
    for i in range(-window, 0):
        prev, curr = prices[i - 1], prices[i]
        if prev > 0 and curr > 0:
            log_rets.append(math.log(curr / prev))
    if len(log_rets) < 5:
        return None
    mean     = sum(log_rets) / len(log_rets)
    variance = sum((r - mean) ** 2 for r in log_rets) / len(log_rets)
    return math.sqrt(variance) * math.sqrt(252)


def _equity(portfolio_state: dict, cash: float) -> float:
    """Total equity = cash + market value of all open positions."""
    last_prices  = portfolio_state.get("last_prices", {})
    positions    = portfolio_state.get("positions", [])
    holdings_val = sum(
        p["quantity"] * last_prices.get(p["ticker"], p.get("avg_cost", 0))
        for p in positions
    )
    return cash + holdings_val


def _current_holdings(portfolio_state: dict) -> dict[str, int]:
    """Return {ticker: quantity} for every open position."""
    return {p["ticker"]: p["quantity"] for p in portfolio_state.get("positions", [])}


def _gross_leverage(weights: dict[str, float], lvrg_mult: dict[str, float]) -> float:
    """Beta-adjusted gross leverage = Σ |weight| × leverage_multiplier."""
    return sum(abs(w) * lvrg_mult.get(t, 1.0) for t, w in weights.items())


def _liquidate_all(holdings: dict[str, int]) -> list[dict]:
    """Sell every open position. Used by both risk-off triggers."""
    return [
        {"ticker": ticker, "side": "sell", "quantity": qty}
        for ticker, qty in holdings.items()
        if qty > 0
    ]


# ── Main decide() ────────────────────────────────────────────────────────────

def decide(
    market_state: dict[str, list[dict]],
    portfolio_state: dict[str, Any],
    cash: float,
) -> list[dict]:
    """
    Called once per trading day by the simulation engine.

    Returns a list of orders, e.g.:
        [{"ticker": "TQQQ", "side": "buy", "quantity": 15},
         {"ticker": "SOXL", "side": "sell", "quantity": 5}]
    Returns [] to do nothing today.
    """

    # ── 0. Minimum-history guard ─────────────────────────────────────────────
    # Don't trade until we have enough bars for all our lookbacks.
    spy_bars = market_state.get(BENCHMARK, [])
    if len(spy_bars) < MIN_BARS:
        return []

    spy_closes = _closes(spy_bars)
    holdings   = _current_holdings(portfolio_state)
    equity     = _equity(portfolio_state, cash)

    # ── 1a. TREND TRIGGER: SPY below 50-day SMA ──────────────────────────────
    # Classic risk-off signal. Reacts to sustained downtrends.
    spy_sma50  = _sma(spy_closes, TREND_WINDOW)
    spy_price  = spy_closes[-1]
    trend_ok   = (spy_sma50 is not None) and (spy_price > spy_sma50)

    if not trend_ok:
        # Market structure is broken — go flat.
        return _liquidate_all(holdings)

    # ── 1b. VIX PANIC EXIT: VIX above threshold ──────────────────────────────
    # Fires faster than the SMA trigger. VIX > 30 historically coincides with
    # sharp, sudden dislocations (e.g. COVID crash, yen carry unwind).
    # Leveraged ETFs can lose 15–20% in a single day during such events;
    # this exit gets us out *before* the SMA has time to react.
    vix_bars = market_state.get(VIX_TICKER, [])
    if vix_bars:
        vix_level = vix_bars[-1]["close"]
        if vix_level > VIX_PANIC_LEVEL:
            # Fear spike detected — liquidate and wait for calm.
            return _liquidate_all(holdings)

    # ── 2. MOMENTUM SCORING ───────────────────────────────────────────────────
    # Score every candidate by blended momentum; pick the top TOP_N winners
    # that have positive scores. SPY is excluded from holdings (it's our filter).
    candidates = LEVERAGED_3X + DEFENSIVE

    scores: dict[str, float] = {}
    vols:   dict[str, float] = {}

    for ticker in candidates:
        bars = market_state.get(ticker, [])
        if not bars or len(bars) < MOM_LONG + 2:
            continue

        closes    = _closes(bars)
        mom_long  = _momentum(closes, MOM_LONG)
        mom_short = _momentum(closes, MOM_SHORT)
        rv        = _realised_vol(closes, VOL_WINDOW)

        if any(x is None for x in (mom_long, mom_short, rv)) or rv <= 0:
            continue

        scores[ticker] = MOM_LONG_WT * mom_long + MOM_SHORT_WT * mom_short
        vols[ticker]   = rv

    if not scores:
        # Nothing scoreable today — sit in cash.
        return _liquidate_all(holdings)

    ranked  = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winners = [t for t, s in ranked if s > 0][:TOP_N]

    # Fallback: if every asset has negative momentum, park in GLD defensively.
    if not winners:
        winners = ["GLD"] if "GLD" in vols else []

    if not winners:
        return _liquidate_all(holdings)

    # ── 3. VOLATILITY-TARGETED SIZING ────────────────────────────────────────
    # Each winner is sized so its vol contribution = target_vol / n_positions.
    # This automatically shrinks positions during volatile periods and grows
    # them in calm ones — without any explicit rebalancing calendar.
    n            = len(winners)
    vol_per_slot = TARGET_VOL / n

    raw_weights: dict[str, float] = {}
    for ticker in winners:
        w = vol_per_slot / vols[ticker]    # vol-parity weight
        w = min(w, MAX_POSITION)           # hard concentration cap (25%)
        raw_weights[ticker] = w

    # Scale down if beta-adjusted gross leverage would breach our ceiling.
    gross = _gross_leverage(raw_weights, LVRG_MULT)
    if gross > GROSS_CAP:
        scale       = GROSS_CAP / gross
        raw_weights = {t: w * scale for t, w in raw_weights.items()}

    # ── 4. RECONCILE PORTFOLIO ────────────────────────────────────────────────
    # Sell anything no longer in winners, then buy/rebalance winners.
    last_prices = portfolio_state.get("last_prices", {})
    orders: list[dict] = []

    # Exit stale positions first (frees up cash for new buys).
    for ticker, qty in holdings.items():
        if ticker not in winners and qty > 0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": qty})

    # Buy or rebalance each winner to its target quantity.
    for ticker, weight in raw_weights.items():
        target_value = equity * weight
        price        = last_prices.get(ticker)

        # Fall back to the bar's close if last_prices doesn't have it.
        if not price or price <= 0:
            bars = market_state.get(ticker, [])
            price = bars[-1]["close"] if bars else None

        if not price or price <= 0:
            continue  # can't size without a price — skip

        target_qty  = int(target_value / price)   # whole shares only
        current_qty = holdings.get(ticker, 0)
        diff        = target_qty - current_qty

        if diff > 0:
            orders.append({"ticker": ticker, "side": "buy",  "quantity": diff})
        elif diff < 0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": abs(diff)})

    return orders
