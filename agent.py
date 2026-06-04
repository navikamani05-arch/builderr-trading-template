"""
agent.py — Hybrid Momentum + Vol-Target + Risk-Off + TLT Rotation (v2.1)
=========================================================================
Strategy: Hybrid (Momentum × Volatility Targeting × Trend Filter × TLT Rotation)
Universe:  Leveraged ETFs (TQQQ, SOXL, UPRO) + safe-haven proxy (TLT)
Aggression: Balanced — targets ~12% annualised vol, guards against deep drawdowns

WHAT'S IN THIS VERSION:
  - SPY 20-day SMA risk-off switch (fast exit)
  - VIX > 25 panic exit
  - TLT rotation during risk-off (instead of pure cash — bonds rally in selloffs)
  - Blended momentum 40/60 short/long weighting
  - Vol-targeted position sizing, capped at 25% per name
  - Leverage safety check (beta-adjusted gross cap at 1.45x)

DELIBERATELY LEFT OUT (hurt the calm uptrend in testing):
  - Re-entry filter (kept agent undersized during uptrends)
  - Crash stop (too many false exits in trending markets)
  - Vol-regime TOP_N switching (reduced returns in calm periods)

No lookahead: all calculations use only prices up to today's bar.
"""

from __future__ import annotations
import math
from typing import Any


# ── Universe ────────────────────────────────────────────────────────────────

LEVERAGED_3X = ["TQQQ", "SOXL", "UPRO"]
BENCHMARK    = "SPY"
VIX_TICKER   = "VIX"
SAFE_HAVEN   = "TLT"      # rotate here during risk-off instead of pure cash


# ── Hyperparameters ──────────────────────────────────────────────────────────

TREND_WINDOW    = 20      # SPY SMA window — if SPY < SMA(20) → risk-off
VIX_PANIC_LEVEL = 25      # VIX threshold — if VIX > 25 → rotate to TLT
MOM_LONG        = 63      # ~3-month momentum lookback
MOM_SHORT       = 21      # ~1-month momentum lookback
MOM_LONG_WT     = 0.40    # weight on 3-month leg
MOM_SHORT_WT    = 0.60    # weight on 1-month leg (recency matters more for leveraged ETFs)
VOL_WINDOW      = 20      # days for realised vol estimate
TARGET_VOL      = 0.12    # annualised portfolio vol target (12%)
MAX_POSITION    = 0.25    # max single-name weight of equity (25%)
TOP_N           = 2       # momentum winners to hold simultaneously
MIN_BARS        = 70      # minimum bars before trading
GROSS_CAP       = 1.45    # beta-adjusted gross leverage ceiling (rule = 1.50x)
LVRG_MULT       = {t: 3.0 for t in LEVERAGED_3X}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _closes(bars: list[dict]) -> list[float]:
    return [b["close"] for b in bars]


def _sma(prices: list[float], window: int) -> float | None:
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window


def _momentum(prices: list[float], lookback: int) -> float | None:
    if len(prices) < lookback + 1:
        return None
    base = prices[-(lookback + 1)]
    end  = prices[-1]
    if base <= 0:
        return None
    return (end / base) - 1.0


def _realised_vol(prices: list[float], window: int) -> float | None:
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
    last_prices  = portfolio_state.get("last_prices", {})
    positions    = portfolio_state.get("positions", [])
    holdings_val = sum(
        p["quantity"] * last_prices.get(p["ticker"], p.get("avg_cost", 0))
        for p in positions
    )
    return cash + holdings_val


def _current_holdings(portfolio_state: dict) -> dict[str, int]:
    return {p["ticker"]: p["quantity"] for p in portfolio_state.get("positions", [])}


def _gross_leverage(weights: dict[str, float], lvrg_mult: dict[str, float]) -> float:
    return sum(abs(w) * lvrg_mult.get(t, 1.0) for t, w in weights.items())


def _get_price(ticker: str, portfolio_state: dict, market_state: dict) -> float | None:
    price = portfolio_state.get("last_prices", {}).get(ticker)
    if not price or price <= 0:
        bars  = market_state.get(ticker, [])
        price = bars[-1]["close"] if bars else None
    return price if price and price > 0 else None


def _rotate_to_tlt(
    holdings: dict[str, int],
    portfolio_state: dict,
    market_state: dict,
    equity: float,
) -> list[dict]:
    """
    Sell everything except TLT, then buy TLT at 90% of equity.
    During risk-off, TLT often appreciates as investors flee to bonds —
    so this actively works for us rather than just sitting idle in cash.
    """
    orders = []

    # Sell all non-TLT positions
    for ticker, qty in holdings.items():
        if ticker != SAFE_HAVEN and qty > 0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": qty})

    # Buy TLT
    price = _get_price(SAFE_HAVEN, portfolio_state, market_state)
    if price:
        target_qty  = int((equity * 0.90) / price)
        current_qty = holdings.get(SAFE_HAVEN, 0)
        diff        = target_qty - current_qty
        if diff > 0:
            orders.append({"ticker": SAFE_HAVEN, "side": "buy",  "quantity": diff})
        elif diff < 0:
            orders.append({"ticker": SAFE_HAVEN, "side": "sell", "quantity": abs(diff)})

    return orders


# ── Main decide() ─────────────────────────────────────────────────────────────

def decide(
    market_state: dict[str, list[dict]],
    portfolio_state: dict[str, Any],
    cash: float,
) -> list[dict]:
    """
    Called once per trading day. Returns a list of orders.
    Returns [] to do nothing today.
    """

    # ── 0. Minimum-history guard ─────────────────────────────────────────────
    spy_bars = market_state.get(BENCHMARK, [])
    if len(spy_bars) < MIN_BARS:
        return []

    spy_closes = _closes(spy_bars)
    holdings   = _current_holdings(portfolio_state)
    equity     = _equity(portfolio_state, cash)

    # ── 1. RISK-OFF CHECK ────────────────────────────────────────────────────

    # Trend trigger: SPY vs 20-day SMA
    spy_sma   = _sma(spy_closes, TREND_WINDOW)
    spy_price = spy_closes[-1]
    above_sma = (spy_sma is not None) and (spy_price > spy_sma)

    # VIX panic trigger
    vix_bars  = market_state.get(VIX_TICKER, [])
    vix_level = vix_bars[-1]["close"] if vix_bars else 0
    vix_calm  = (vix_level == 0) or (vix_level < VIX_PANIC_LEVEL)

    risk_on = above_sma and vix_calm

    if not risk_on:
        # Rotate to TLT — actively defensive rather than idle cash
        return _rotate_to_tlt(holdings, portfolio_state, market_state, equity)

    # ── 2. MOMENTUM SCORING ──────────────────────────────────────────────────
    # Score TQQQ, SOXL, UPRO by blended momentum. Pick top TOP_N with
    # positive scores. Fall back to GLD if nothing is positive.
    scores: dict[str, float] = {}
    vols:   dict[str, float] = {}

    for ticker in LEVERAGED_3X:
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

    ranked  = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winners = [t for t, s in ranked if s > 0][:TOP_N]

    # Fallback: no positive leveraged ETF momentum → hold GLD defensively
    if not winners:
        gld_bars = market_state.get("GLD", [])
        if gld_bars and len(gld_bars) >= VOL_WINDOW + 1:
            gld_closes = _closes(gld_bars)
            gld_rv     = _realised_vol(gld_closes, VOL_WINDOW)
            if gld_rv and gld_rv > 0:
                winners       = ["GLD"]
                vols["GLD"]   = gld_rv

    if not winners:
        # Nothing attractive — rotate to TLT and wait
        return _rotate_to_tlt(holdings, portfolio_state, market_state, equity)

    # ── 3. VOLATILITY-TARGETED SIZING ────────────────────────────────────────
    n            = len(winners)
    vol_per_slot = TARGET_VOL / n

    raw_weights: dict[str, float] = {}
    for ticker in winners:
        w = vol_per_slot / vols[ticker]
        w = min(w, MAX_POSITION)
        raw_weights[ticker] = w

    # Scale down if gross leverage breaches ceiling
    gross = _gross_leverage(raw_weights, LVRG_MULT)
    if gross > GROSS_CAP:
        scale       = GROSS_CAP / gross
        raw_weights = {t: w * scale for t, w in raw_weights.items()}

    # ── 4. RECONCILE PORTFOLIO ───────────────────────────────────────────────
    orders: list[dict] = []

    # Exit positions no longer in winners
    for ticker, qty in holdings.items():
        if ticker not in winners and qty > 0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": qty})

    # Buy or rebalance winners
    for ticker, weight in raw_weights.items():
        target_value = equity * weight
        price        = _get_price(ticker, portfolio_state, market_state)
        if not price:
            continue
        target_qty  = int(target_value / price)
        current_qty = holdings.get(ticker, 0)
        diff        = target_qty - current_qty
        if diff > 0:
            orders.append({"ticker": ticker, "side": "buy",  "quantity": diff})
        elif diff < 0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": abs(diff)})

    return orders
