import os
import json
import time
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import requests

# OpenAI (new SDK)
from openai import OpenAI

# =============================
# OPTIONAL PDF SUPPORT (ReportLab)
# =============================
REPORTLAB_OK = True
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
except Exception:
    REPORTLAB_OK = False

st.set_page_config(page_title="FA→TA Trading + AI", layout="wide")

# =============================
# Helpers
# =============================
def normalize_ticker(raw: str, market: str) -> str:
    t = (raw or "").strip().upper()
    if not t:
        return t
    if market == "BIST":
        if not t.endswith(".IS"):
            t = f"{t}.IS"
    return t

def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan

def _flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.dropna()

def fmt_pct(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "N/A"
        return f"{x*100:.2f}%"
    except Exception:
        return "N/A"

def fmt_num(x: float, nd=2) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "N/A"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "N/A"

# =============================
# Indicators
# =============================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(alpha=1 / period, adjust=False).mean()
    roll_down = pd.Series(down, index=close.index).ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = mid + std_mult * sd
    lower = mid - std_mult * sd
    return mid, upper, lower

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def max_drawdown(eq: pd.Series) -> float:
    if eq is None or len(eq) == 0:
        return 0.0
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())

# =============================
# Feature builder
# =============================
def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["EMA50"] = ema(df["Close"], int(cfg["ema_fast"]))
    df["EMA200"] = ema(df["Close"], int(cfg["ema_slow"]))
    df["RSI"] = rsi(df["Close"], int(cfg["rsi_period"]))
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"], 12, 26, 9)
    df["BB_mid"], df["BB_upper"], df["BB_lower"] = bollinger(df["Close"], int(cfg["bb_period"]), float(cfg["bb_std"]))
    df["ATR"] = atr(df["High"], df["Low"], df["Close"], int(cfg["atr_period"]))
    df["OBV"] = obv(df["Close"], df["Volume"])
    df["OBV_EMA"] = ema(df["OBV"], 21)
    df["VOL_SMA"] = df["Volume"].rolling(int(cfg["vol_sma"])).mean()
    df["ATR_PCT"] = (df["ATR"] / df["Close"]).replace([np.inf, -np.inf], np.nan)
    return df

# =============================
# Market regime filter (SPY) - only USA
# =============================
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_spy_regime_ok() -> bool:
    spy = yf.download("SPY", period="10y", interval="1d", auto_adjust=False, progress=False)
    spy = _flatten_yf(spy)
    if spy.empty or len(spy) < 260:
        return True
    spy["EMA200"] = ema(spy["Close"], 200)
    last = spy.iloc[-1]
    return bool(last["Close"] > last["EMA200"])

# =============================
# Strategy: scoring + checkpoints
# =============================
def signal_with_checkpoints(df: pd.DataFrame, cfg: dict, market_filter_ok: bool):
    df = df.copy()

    liq_ok = (df["Volume"] > df["VOL_SMA"]).fillna(False)
    trend_ok = (df["Close"] > df["EMA200"]) & (df["EMA50"] > df["EMA200"])

    rsi_ok = df["RSI"] > cfg["rsi_entry_level"]
    rsi_cross = (df["RSI"] > cfg["rsi_entry_level"]) & (df["RSI"].shift(1) <= cfg["rsi_entry_level"])

    macd_ok = df["MACD_hist"] > 0
    macd_turn = (df["MACD_hist"] > 0) & (df["MACD_hist"].shift(1) <= 0)

    atr_pct = (df["ATR"] / df["Close"]).replace([np.inf, -np.inf], np.nan)
    vol_ok = atr_pct < cfg["atr_pct_max"]

    bb_ok = df["Close"] > df["BB_mid"]
    bb_break = (df["Close"] > df["BB_upper"]) & trend_ok

    obv_ok = df["OBV"] > df["OBV_EMA"]

    w = {"liq": 10, "trend": 25, "rsi": 15, "macd": 15, "vol": 10, "bb": 15, "obv": 10}
    score = (
        w["liq"] * liq_ok.astype(int)
        + w["trend"] * trend_ok.astype(int)
        + w["rsi"] * rsi_ok.astype(int)
        + w["macd"] * macd_ok.astype(int)
        + w["vol"] * vol_ok.astype(int)
        + w["bb"] * (bb_ok | bb_break).astype(int)
        + w["obv"] * obv_ok.astype(int)
    ).astype(float)

    entry_triggers = (rsi_cross.astype(int) + macd_turn.astype(int) + bb_break.astype(int)) >= 2
    entry = trend_ok & vol_ok & liq_ok & entry_triggers & market_filter_ok

    exit_ = (
        (df["Close"] < df["EMA50"])
        | (df["MACD_hist"] < 0)
        | (df["RSI"] < cfg["rsi_exit_level"])
        | (df["Close"] < df["BB_mid"])
    )

    df["SCORE"] = score
    df["ENTRY"] = entry.astype(int)
    df["EXIT"] = exit_.astype(int)

    last = df.iloc[-1]
    cp = {
        "Market Filter OK": bool(market_filter_ok),
        "Liquidity (Volume > VolSMA)": bool(last["Volume"] > last["VOL_SMA"]) if pd.notna(last["VOL_SMA"]) else False,
        "Trend (Close>EMA200 & EMA50>EMA200)": bool((last["Close"] > last["EMA200"]) and (last["EMA50"] > last["EMA200"]))
        if pd.notna(last["EMA200"]) else False,
        f"RSI > {cfg['rsi_entry_level']}": bool(last["RSI"] > cfg["rsi_entry_level"]) if pd.notna(last["RSI"]) else False,
        "MACD Hist > 0": bool(last["MACD_hist"] > 0) if pd.notna(last["MACD_hist"]) else False,
        f"ATR% < {cfg['atr_pct_max']:.2%}": bool((last["ATR"] / last["Close"]) < cfg["atr_pct_max"])
        if pd.notna(last["ATR"]) and pd.notna(last["Close"]) else False,
        "Bollinger (Close>BB_mid or Breakout)": bool((last["Close"] > last["BB_mid"]) or (last["Close"] > last["BB_upper"]))
        if pd.notna(last["BB_mid"]) else False,
        "OBV > OBV_EMA": bool(last["OBV"] > last["OBV_EMA"]) if pd.notna(last["OBV_EMA"]) else False,
    }
    return df, cp

# =============================
# Backtest (long-only) + metrics
# =============================
def backtest_long_only(df: pd.DataFrame, cfg: dict, risk_free_annual: float):
    df = df.copy()
    entry_sig = df["ENTRY"].shift(1).fillna(0).astype(int)
    exit_sig = df["EXIT"].shift(1).fillna(0).astype(int)

    cash = float(cfg["initial_capital"])
    shares = 0.0
    stop = np.nan

    trades = []
    equity_curve = []

    commission = cfg["commission_bps"] / 10000.0
    slippage = cfg["slippage_bps"] / 10000.0

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        price = float(row["Close"])

        position_value = shares * price * (1 - slippage)
        equity = cash + position_value

        if shares > 0 and pd.notna(row["ATR"]) and row["ATR"] > 0:
            new_stop = price - cfg["atr_stop_mult"] * float(row["ATR"])
            stop = max(stop, new_stop) if pd.notna(stop) else new_stop

        if shares > 0:
            stop_hit = pd.notna(stop) and (price <= stop)
            if exit_sig.iloc[i] == 1 or stop_hit:
                sell_price = price * (1 - slippage)
                gross = shares * sell_price
                fee = gross * commission
                cash += (gross - fee)

                trades[-1]["exit_date"] = date
                trades[-1]["exit_price"] = sell_price
                trades[-1]["exit_reason"] = "STOP" if stop_hit else "RULE_EXIT"
                trades[-1]["pnl"] = cash - trades[-1]["equity_before"]

                shares = 0.0
                stop = np.nan

        position_value = shares * price * (1 - slippage)
        equity = cash + position_value

        if shares == 0 and entry_sig.iloc[i] == 1 and pd.notna(row["ATR"]) and row["ATR"] > 0:
            risk_cash = equity * cfg["risk_per_trade"]
            stop_dist = cfg["atr_stop_mult"] * float(row["ATR"])
            if stop_dist > 0:
                qty = risk_cash / stop_dist
                buy_price = price * (1 + slippage)
                cost = qty * buy_price
                fee = cost * commission
                total_cost = cost + fee

                if total_cost <= cash:
                    cash -= total_cost
                    shares = qty
                    stop = buy_price - cfg["atr_stop_mult"] * float(row["ATR"])

                    trades.append(
                        {
                            "entry_date": date,
                            "entry_price": buy_price,
                            "exit_date": None,
                            "exit_price": None,
                            "exit_reason": None,
                            "shares": shares,
                            "equity_before": equity,
                            "pnl": None,
                        }
                    )

        position_value = shares * price * (1 - slippage)
        equity = cash + position_value
        equity_curve.append((date, equity))

    eq = pd.Series([v for _, v in equity_curve], index=[d for d, _ in equity_curve], name="equity").astype(float)
    eq = eq.replace([np.inf, -np.inf], np.nan).dropna()

    ret = eq.pct_change().dropna()
    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    ann_return = (1 + total_return) ** (252 / max(1, len(ret))) - 1 if len(ret) > 0 else 0.0
    ann_vol = float(ret.std() * np.sqrt(252)) if len(ret) > 1 else 0.0

    rf_daily = (1 + float(risk_free_annual)) ** (1 / 252) - 1
    excess = ret - rf_daily

    sharpe = float((excess.mean() * 252) / (excess.std() * np.sqrt(252))) if len(ret) > 1 and excess.std() > 0 else 0.0
    downside = excess.copy()
    downside[downside > 0] = 0
    downside_dev = float(np.sqrt((downside**2).mean()) * np.sqrt(252)) if len(downside) > 1 else 0.0
    sortino = float((excess.mean() * 252) / downside_dev) if downside_dev > 0 else 0.0

    mdd = max_drawdown(eq)
    calmar = float(ann_return / abs(mdd)) if mdd < 0 else 0.0

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tdf["pnl"] = tdf["pnl"].astype(float)
        tdf["return_%"] = (tdf["pnl"] / tdf["equity_before"]) * 100
        tdf["holding_days"] = (pd.to_datetime(tdf["exit_date"]) - pd.to_datetime(tdf["entry_date"])).dt.days

    profit_factor = 0.0
    if not tdf.empty and "pnl" in tdf.columns:
        gross_profit = float(tdf.loc[tdf["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-tdf.loc[tdf["pnl"] < 0, "pnl"].sum())
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0 and gross_loss == 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

    metrics = {
        "Total Return": float(total_return),
        "Annualized Return": float(ann_return),
        "Annualized Volatility": float(ann_vol),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Calmar": float(calmar),
        "Max Drawdown": float(mdd),
        "Trades": int(len(tdf)) if not tdf.empty else 0,
        "Win Rate": float((tdf["pnl"] > 0).mean()) if not tdf.empty else 0.0,
        "Profit Factor": float(profit_factor) if np.isfinite(profit_factor) else float("inf"),
    }
    return eq, tdf, metrics

# =============================
# Fundamentals (USA + BIST) via yfinance info
# =============================
def _fix_debt_to_equity(x: float) -> float:
    if pd.notna(x) and x > 10:
        return x / 100.0
    return x

@st.cache_data(ttl=12 * 3600, show_spinner=False)
def fetch_fundamentals_generic(ticker: str, market: str) -> dict:
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    out = {
        "ticker": ticker,
        "market": market,
        "marketCap": safe_float(info.get("marketCap")),
        "trailingPE": safe_float(info.get("trailingPE")),
        "forwardPE": safe_float(info.get("forwardPE")),
        "pegRatio": safe_float(info.get("pegRatio")),
        "priceToSalesTrailing12Months": safe_float(info.get("priceToSalesTrailing12Months")),
        "priceToBook": safe_float(info.get("priceToBook")),
        "returnOnEquity": safe_float(info.get("returnOnEquity")),
        "profitMargins": safe_float(info.get("profitMargins")),
        "operatingMargins": safe_float(info.get("operatingMargins")),
        "debtToEquity": safe_float(info.get("debtToEquity")),
        "revenueGrowth": safe_float(info.get("revenueGrowth")),
        "earningsGrowth": safe_float(info.get("earningsGrowth")),
        "freeCashflow": safe_float(info.get("freeCashflow")),
        "currentPrice": safe_float(info.get("currentPrice")),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "longName": info.get("longName", "") or info.get("shortName", ""),
    }
    out["debtToEquity"] = _fix_debt_to_equity(out["debtToEquity"])
    return out

def fundamental_score_row(row: dict, mode: str, thresholds: dict) -> Tuple[float, dict, bool]:
    b = {}

    def ok(name, cond, weight, available: bool):
        b[name] = {"ok": bool(cond) if available else False, "weight": weight, "available": bool(available)}
        return (weight if (available and cond) else 0.0), (weight if available else 0.0), (1 if available else 0)

    score = 0.0
    total_w = 0.0
    avail_cnt = 0
    ok_cnt = 0

    def A(x):
        return pd.notna(x)

    if mode == "Quality":
        s, tw, ac = ok("ROE", A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"], 20, A(row["returnOnEquity"]))
        score += s; total_w += tw; avail_cnt += ac; ok_cnt += (1 if (A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"]) else 0)

        s, tw, ac = ok("Op Margin", A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"], 15, A(row["operatingMargins"]))
        score += s; total_w += tw; avail_cnt += ac; ok_cnt += (1 if (A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"]) else 0)

        s, tw, ac = ok("Debt/Equity", A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"], 20, A(row["debtToEquity"]))
        score += s; total_w += tw; avail_cnt += ac; ok_cnt += (1 if (A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"]) else 0)

        s, tw, ac = ok("Profit Margin", A(row["profitMargins"]) and row["profitMargins"] >= thresholds["profit_margin"], 15, A(row["profitMargins"]))
        score += s; total_w += tw; avail_cnt += ac; ok_cnt += (1 if (A(row["profitMargins"]) and row["profitMargins"] >= thresholds["profit_margin"]) else 0)

        s, tw, ac = ok("FCF", A(row["freeCashflow"]) and row["freeCashflow"] > 0, 30, A(row["freeCashflow"]))
        score += s; total_w += tw; avail_cnt += ac; ok_cnt += (1 if (A(row["freeCashflow"]) and row["freeCashflow"] > 0) else 0)

    elif mode == "Value":
        s, tw, ac = ok("Forward P/E", A(row["forwardPE"]) and row["forwardPE"] <= thresholds["fpe"], 30, A(row["forwardPE"]))
        score += s; total_w += tw; avail_cnt += ac; ok_cnt += (1 if (A(row["forwardPE"]) and row["forwardPE"] <= thresholds["fpe"]) else 0)

        s, tw, ac = ok("PEG", A(row["pegRatio"]) and row["pegRatio"] <= thresholds["peg"], 20, A(row["pegRatio"]))
        score += s; total_w += tw; avail_cnt += ac; ok_cnt += (1 if (A(row["pegRatio"]) and row["pegRatio"] <= thresholds["peg"]) else 0)

        s, tw, ac = ok("P/S", A(row["priceToSalesTrailing12Months"]) and row["priceToSalesTrailing12Months"] <= thresholds["ps"], 20, A(row["priceToSalesTrailing12Months"]))
        score += s; total_w += tw; avail_cnt += ac; ok_cnt += (1 if (A(row["priceToSalesTrailing12Months"]) and row["priceToSalesTrailing12Months"] <= thresholds["ps"]) else 0)

        s, tw, ac = ok("P/B", A(row["priceToBook"]) and row["priceToBook"] <= thresholds["pb"], 15, A(row["priceToBook"]))
        score += s; total_w += tw; avail_cnt += ac; ok_cnt += (1 if (A(row["priceToBook"]) and row["priceToBook"] <= thresholds["pb"]) else 0)

        s, tw, ac = ok("ROE", A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"], 15, A(row["returnOnEquity"]))
        score += s; total_w += tw; avail_cnt += ac; ok_cnt += (1 if (A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"]) else 0)

    else:  # Growth
        s, tw, ac = ok("Revenue Growth", A(row["revenueGrowth"]) and row["revenueGrowth"] >= thresholds["rev_g"], 35, A(row["revenueGrowth"]))
        score += s; total_w += tw; avail_cnt += ac; ok_cnt += (1 if (A(row["revenueGrowth"]) and row["revenueGrowth"] >= thresholds["rev_g"]) else 0)

        s, tw, ac = ok("Earnings Growth", A(row["earningsGrowth"]) and row["earningsGrowth"] >= thresholds["earn_g"], 35, A(row["earningsGrowth"]))
        score += s; total_w += tw; avail_cnt += ac; ok_cnt += (1 if (A(row["earningsGrowth"]) and row["earningsGrowth"] >= thresholds["earn_g"]) else 0)

        s, tw, ac = ok("Op Margin", A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"], 15, A(row["operatingMargins"]))
        score += s; total_w += tw; avail_cnt += ac; ok_cnt += (1 if (A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"]) else 0)

        s, tw, ac = ok("Debt/Equity", A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"], 15, A(row["debtToEquity"]))
        score += s; total_w += tw; avail_cnt += ac; ok_cnt += (1 if (A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"]) else 0)

    score_pct = (score / total_w) * 100 if total_w > 0 else 0.0
    min_coverage = int(thresholds.get("min_coverage", 3))
    min_ok = int(thresholds["min_ok"])
    pass_bool = (score_pct >= thresholds["min_score"]) and (ok_cnt >= min_ok) and (avail_cnt >= min_coverage)
    return float(score_pct), b, bool(pass_bool)

# =============================
# Universe helpers
# =============================
US_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "NFLX", "JPM", "XOM", "SPY", "QQQ"]
US_EXT = ["AVGO", "AMD", "ORCL", "COST", "KO", "PEP", "JNJ", "PG", "V", "MA", "UNH", "HD", "CRM", "ADBE"]

BIST100_FALLBACK = [
    "AEFES","AGHOL","AHGAZ","AKBNK","AKFGY","AKSA","AKSEN","ALARK","ALBRK","ALFAS",
    "ARCLK","ARDYZ","ASELS","ASTOR","BIMAS","BRSAN","BRYAT","BSOKE","CCOLA","CEMTS",
    "CIMSA","DOAS","ECILC","EGEEN","EKGYO","ENJSA","ENKAI","EREGL","EUPWR","FROTO",
    "GARAN","GESAN","GUBRF","HALKB","HEKTS","ISCTR","ISGYO","ISMEN","KARDM","KCAER",
    "KCHOL","KLGYO","KONTR","KOZAA","KOZAL","KRDMD","MAVI","MGROS","ODAS","OTKAR",
    "OYAKC","PETKM","PGSUS","QNBFB","SAHOL","SASA","SDTTR","SISE","SKBNK","SMRTG",
    "SOKM","SRVGY","TAVHL","TCELL","THYAO","TKFEN","TMSN","TOASO","TRGYO","TSKB",
    "TSPOR","TTKOM","TTRAK","TUKAS","TUPRS","TURSG","ULKER","VAKBN","VESBE","VESTL",
    "YKBNK","ZOREN"
]

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_sp500_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0; +https://streamlit.io)"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        dfu = tables[0]
        return sorted(dfu["Symbol"].astype(str).str.upper().tolist())
    except Exception:
        return sorted(list(set(US_TICKERS + US_EXT)))

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_nasdaq100_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0; +https://streamlit.io)"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        for t in tables:
            if "Ticker" in t.columns:
                return sorted(t["Ticker"].astype(str).str.upper().tolist())
        return []
    except Exception:
        return []

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_bist100_tickers() -> List[str]:
    candidates = [
        "https://www.borsaistanbul.com/en/indices/bist-stock-indices/bist-100",
        "https://www.borsaistanbul.com/tr/sayfa/195/bist-pay-endeksleri",
        "https://www.borsaistanbul.com/tr/sayfa/194/endeksler",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0; +https://streamlit.io)"}

    for url in candidates:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            tables = pd.read_html(r.text)
            for t in tables:
                sym_col = None
                for c in t.columns:
                    cl = str(c).lower()
                    if "kod" in cl or "symbol" in cl or "sembol" in cl or "ticker" in cl:
                        sym_col = c
                        break
                if sym_col is not None:
                    syms = t[sym_col].astype(str).str.upper().str.replace(".IS", "", regex=False).tolist()
                    syms = [s.strip() for s in syms if s.strip()]
                    if len(syms) >= 50:
                        return sorted(list(set(syms)))
        except Exception:
            pass

    return sorted(list(set(BIST100_FALLBACK)))

# =============================
# Target price band (non-LLM)
# =============================
def local_levels(close: pd.Series, lookback: int = 120):
    s = close.tail(lookback).dropna()
    if len(s) < 10:
        return []
    levels = list(np.quantile(s.values, [0.1, 0.25, 0.5, 0.75, 0.9]))
    levels += [float(s.tail(20).max()), float(s.tail(20).min())]
    levels = sorted(list(set([round(float(x), 2) for x in levels if np.isfinite(x)])))
    return levels

def target_price_band(df: pd.DataFrame):
    last = df.iloc[-1]
    px = float(last["Close"])
    atrv = float(last["ATR"]) if pd.notna(last.get("ATR", np.nan)) else np.nan
    if not np.isfinite(atrv) or atrv <= 0:
        return {"base": px, "bull": None, "bear": None, "levels": local_levels(df["Close"])}

    bull1 = px + 1.5 * atrv
    bull2 = px + 3.0 * atrv
    bear1 = px - 1.5 * atrv
    bear2 = px - 3.0 * atrv

    lv = local_levels(df["Close"])
    above = [x for x in lv if x >= px]
    below = [x for x in lv if x <= px]
    r1 = min(above) if above else None
    s1 = max(below) if below else None

    return {"base": px, "bull": (bull1, bull2, r1), "bear": (bear1, bear2, s1), "levels": lv}

# =============================
# Live price helper
# =============================
@st.cache_data(ttl=30, show_spinner=False)
def get_live_price(ticker: str) -> dict:
    out = {"last_price": np.nan, "currency": "", "exchange": "", "asof": ""}
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi:
            out["last_price"] = safe_float(fi.get("last_price") or fi.get("lastPrice"))
            out["currency"] = fi.get("currency") or ""
            out["exchange"] = fi.get("exchange") or ""
            out["asof"] = str(fi.get("last_trade_time") or fi.get("lastTradeDate") or "")
    except Exception:
        pass
    return out

# =============================
# LLM helpers
# =============================
def df_snapshot_for_llm(df: pd.DataFrame, n: int = 140) -> dict:
    use_cols = ["Open","High","Low","Close","Volume","EMA50","EMA200","RSI","MACD","MACD_signal","MACD_hist",
                "BB_mid","BB_upper","BB_lower","ATR","ATR_PCT","VOL_SMA","SCORE","ENTRY","EXIT"]
    cols = [c for c in use_cols if c in df.columns]
    tail = df[cols].tail(n).copy()
    tail.index = tail.index.astype(str)
    return {"cols": cols, "n": int(len(tail)), "last_index": str(tail.index[-1]) if len(tail) else None, "rows": tail.to_dict(orient="records")}

def build_ai_context(ticker: str, market: str, latest: pd.Series, checkpoints: dict, metrics: dict, tp: dict, live: dict, df: pd.DataFrame) -> dict:
    return {
        "ticker": ticker,
        "market": market,
        "live_price": live,
        "latest_bar": {
            "time": str(latest.name),
            "close": float(latest["Close"]),
            "score": float(latest.get("SCORE", np.nan)),
            "rsi": float(latest.get("RSI", np.nan)),
            "ema50": float(latest.get("EMA50", np.nan)),
            "ema200": float(latest.get("EMA200", np.nan)),
            "atr": float(latest.get("ATR", np.nan)),
            "atr_pct": float(latest.get("ATR_PCT", np.nan)),
        },
        "checkpoints": checkpoints,
        "backtest_metrics": metrics,
        "target_price_band": tp,
        "data_snapshot": df_snapshot_for_llm(df, n=160),
        "constraints": [
            "Yatırım tavsiyesi verme. Sadece eğitim amaçlı analiz.",
            "Hedef fiyatı tek sayı değil; senaryo (bull/base/bear) bandı olarak açıkla.",
            "Mutlaka riskler ve geçersiz kılacak koşulları yaz (invalidations).",
        ],
    }

def call_openai(messages, model: str, temperature: float = 0.2):
    api_key = st.secrets.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY bulunamadı. Streamlit Cloud > Secrets'e OPENAI_API_KEY=... ekleyin.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return resp.choices[0].message.content

# =============================
# Presets
# =============================
PRESETS = {
    "Defansif": {"rsi_entry_level": 52, "rsi_exit_level": 46, "atr_pct_max": 0.06, "atr_stop_mult": 3.5},
    "Dengeli": {"rsi_entry_level": 50, "rsi_exit_level": 45, "atr_pct_max": 0.08, "atr_stop_mult": 3.0},
    "Agresif": {"rsi_entry_level": 48, "rsi_exit_level": 43, "atr_pct_max": 0.10, "atr_stop_mult": 2.5},
}

# =============================
# Screener row finder + merge
# =============================
def find_screener_row(sdf: pd.DataFrame, ticker: str) -> Optional[Dict[str, Any]]:
    if sdf is None or sdf.empty or "ticker" not in sdf.columns:
        return None
    t = (ticker or "").upper().strip()
    t_naked = t.replace(".IS", "")

    tmp = sdf.copy()
    tmp["_tk"] = tmp["ticker"].astype(str).str.upper().str.strip()
    tmp["_tk_naked"] = tmp["_tk"].str.replace(".IS", "", regex=False)

    m = tmp[(tmp["_tk"] == t) | (tmp["_tk"] == f"{t_naked}.IS") | (tmp["_tk_naked"] == t_naked)]
    if m.empty:
        return None
    row = m.iloc[0].drop(labels=["_tk", "_tk_naked"], errors="ignore").to_dict()
    return row

def merge_fa_row(screener_row: Optional[Dict[str, Any]], fundamentals: Optional[Dict[str, Any]], fa_eval: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if fundamentals:
        out.update(fundamentals)
    if screener_row:
        out.update(screener_row)
    if fa_eval:
        out["FA_mode"] = fa_eval.get("mode")
        out["FA_score"] = fa_eval.get("score")
        out["FA_pass"] = fa_eval.get("passed")
        out["FA_ok_count"] = fa_eval.get("ok_cnt")
        out["FA_coverage"] = fa_eval.get("coverage")
    return out

# =============================
# REPORT EXPORT (Robust): HTML always, PDF if available
# =============================
def build_html_report(
    title: str,
    meta: dict,
    checkpoints: dict,
    metrics: dict,
    tp: dict,
    rr_info: dict,
    figs: Dict[str, go.Figure],
    fa_row: Optional[Dict[str, Any]] = None
) -> bytes:
    def esc(x):
        return (str(x)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

    # Plotly figures (keeps charts exactly)
    fig_blocks = []
    first = True
    for name, fig in (figs or {}).items():
        fig_html = fig.to_html(full_html=False, include_plotlyjs=("cdn" if first else False))
        first = False
        fig_blocks.append(f"<h3>{esc(name)}</h3>{fig_html}")

    cp_list = "".join([f"<li>{'✅' if v else '❌'} {esc(k)}</li>" for k, v in checkpoints.items()])

    bull = tp.get("bull")
    bear = tp.get("bear")
    levels = tp.get("levels", []) or []
    levels_txt = ", ".join([f"{x:.2f}" for x in levels[:120]])

    show_cols = [
        ("ticker", "Ticker"),
        ("longName", "Name"),
        ("FA_pass", "FA_pass"),
        ("FA_score", "FA_score"),
        ("FA_ok_count", "FA_ok_count"),
        ("FA_coverage", "FA_coverage"),
        ("sector", "Sector"),
        ("industry", "Industry"),
        ("trailingPE", "Trailing PE"),
        ("forwardPE", "Forward PE"),
        ("pegRatio", "PEG"),
        ("priceToSalesTrailing12Months", "P/S"),
        ("priceToBook", "P/B"),
        ("returnOnEquity", "ROE"),
        ("operatingMargins", "Op Margin"),
        ("profitMargins", "Profit Margin"),
        ("debtToEquity", "Debt/Equity"),
        ("revenueGrowth", "Revenue Growth"),
        ("earningsGrowth", "Earnings Growth"),
        ("marketCap", "Market Cap"),
    ]

    fa_rows_html = ""
    if fa_row:
        for key, label in show_cols:
            val = fa_row.get(key, "")
            fa_rows_html += f"<tr><td><b>{esc(label)}</b></td><td>{esc(val)}</td></tr>"
    else:
        fa_rows_html = "<tr><td colspan='2'>Screener satırı bulunamadı (screener çalıştırılmamış olabilir).</td></tr>"

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{esc(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .muted {{ color: #666; font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px; }}
    h1,h2,h3 {{ margin: 0 0 8px 0; }}
    ul {{ margin: 8px 0 0 18px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td {{ border-top: 1px solid #eee; padding: 6px 8px; vertical-align: top; }}
    @media print {{
      .no-print {{ display: none; }}
      body {{ margin: 10mm; }}
    }}
  </style>
</head>
<body>
  <div class="no-print card" style="background:#fff7e6;border-color:#ffd591;">
    <b>PDF yapmak için:</b> Bu dosyayı indir → tarayıcıda aç → <b>Ctrl+P</b> → <b>Save as PDF</b>.
  </div>

  <h1>{esc(title)}</h1>
  <div class="muted">
    Generated: {esc(time.strftime('%Y-%m-%d %H:%M:%S'))}<br>
    Market: {esc(meta.get('market'))} | Ticker: {esc(meta.get('ticker'))} | Interval: {esc(meta.get('interval'))} | Period: {esc(meta.get('period'))}<br>
    Preset: {esc(meta.get('preset'))} | EMA: {esc(meta.get('ema_fast'))}/{esc(meta.get('ema_slow'))} | RSI: {esc(meta.get('rsi_period'))} | BB: {esc(meta.get('bb_period'))}/{esc(meta.get('bb_std'))} | ATR: {esc(meta.get('atr_period'))} | VolSMA: {esc(meta.get('vol_sma'))}
  </div>

  <div class="grid" style="margin-top:14px;">
    <div class="card">
      <h2>Checkpoints</h2>
      <ul>{cp_list}</ul>
    </div>
    <div class="card">
      <h2>Backtest</h2>
      <div>Total Return: {metrics.get('Total Return',0)*100:.1f}%</div>
      <div>Ann Return: {metrics.get('Annualized Return',0)*100:.1f}%</div>
      <div>Sharpe: {metrics.get('Sharpe',0):.2f}</div>
      <div>Max DD: {metrics.get('Max Drawdown',0)*100:.1f}%</div>
      <div>Trades: {metrics.get('Trades',0)}</div>
      <div>Win Rate: {metrics.get('Win Rate',0)*100:.1f}%</div>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h2>Target Band</h2>
    <div>Base: {tp.get('base',0):.2f}</div>
    <div>Bull: {(bull[0] if bull else 0):.2f} → {(bull[1] if bull else 0):.2f} | R1: {(bull[2] if bull else 'N/A')}</div>
    <div>Bear: {(bear[0] if bear else 0):.2f} → {(bear[1] if bear else 0):.2f} | S1: {(bear[2] if bear else 'N/A')}</div>
    <div>RR: {('N/A' if rr_info.get('rr') is None else f"1:{rr_info.get('rr'):.2f}")}</div>
    <div class="muted">Levels: {esc(levels_txt)}</div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h2>Fundamental Screener Snapshot (Selected Ticker)</h2>
    <table>{fa_rows_html}</table>
  </div>

  <div style="margin-top:18px;">
    {''.join(fig_blocks)}
  </div>
</body>
</html>
"""
    return html.encode("utf-8")

def _plotly_fig_to_png_bytes(fig: go.Figure) -> Optional[bytes]:
    # works if kaleido is available
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None

def _pdf_write_lines(c, lines: List[str], x: float, y: float, lh: float, bottom: float):
    for line in lines:
        if y <= bottom:
            c.showPage()
            y = A4[1] - 2.0 * cm
        c.drawString(x, y, (line or "")[:220])
        y -= lh
    return y

def generate_pdf_report(
    *,
    title: str,
    subtitle: str,
    meta: dict,
    checkpoints: dict,
    ta_summary: dict,
    target_band: dict,
    rr_info: dict,
    backtest_metrics: dict,
    fa_row: Optional[Dict[str, Any]],
    levels: Optional[List[float]],
    trades_df: Optional[pd.DataFrame],
    figs: Optional[Dict[str, go.Figure]],
    include_charts: bool = True,
) -> Optional[bytes]:
    if not REPORTLAB_OK:
        return None

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    left = 1.6 * cm
    right = W - 1.6 * cm
    top = H - 1.6 * cm
    bottom = 1.6 * cm
    y = top

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, y, title[:90]); y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(left, y, subtitle[:140]); y -= 14

    c.setFont("Helvetica", 9)
    y = _pdf_write_lines(
        c,
        [
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Market: {meta.get('market','')} | Ticker: {meta.get('ticker','')} | Interval: {meta.get('interval','')} | Period: {meta.get('period','')}",
            f"Preset: {meta.get('preset','')} | EMA: {meta.get('ema_fast','')}/{meta.get('ema_slow','')} | RSI: {meta.get('rsi_period','')} | BB: {meta.get('bb_period','')}/{meta.get('bb_std','')} | ATR: {meta.get('atr_period','')} | VolSMA: {meta.get('vol_sma','')}",
        ],
        left, y, 12, bottom
    )
    y -= 6

    # TA Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Technical Summary"); y -= 14
    c.setFont("Helvetica", 9)
    y = _pdf_write_lines(
        c,
        [
            f"Recommendation: {ta_summary.get('rec','')}",
            f"Last Close (bar): {ta_summary.get('close','N/A')} | Live/Last: {ta_summary.get('live','N/A')}",
            f"Score: {ta_summary.get('score','N/A')} | RSI: {ta_summary.get('rsi','N/A')} | EMA50: {ta_summary.get('ema50','N/A')} | EMA200: {ta_summary.get('ema200','N/A')} | ATR%: {ta_summary.get('atr_pct','N/A')}",
        ],
        left, y, 12, bottom
    )
    y -= 6

    # Checkpoints
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Checkpoints (Last Bar)"); y -= 14
    c.setFont("Helvetica", 9)
    cp_lines = [f"[{'OK' if v else 'NO'}] {k}" for k, v in checkpoints.items()]
    y = _pdf_write_lines(c, cp_lines, left, y, 11, bottom)
    y -= 6

    # Target band + RR
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Target Price Band (Scenario)"); y -= 14
    c.setFont("Helvetica", 9)

    base = target_band.get("base")
    bull = target_band.get("bull")
    bear = target_band.get("bear")
    rr = rr_info.get("rr")
    stop = rr_info.get("stop")

    band_lines = [f"Base: {fmt_num(base)}"]
    if bull:
        band_lines.append(f"Bull: {fmt_num(bull[0])} -> {fmt_num(bull[1])} | Near Resistance: {fmt_num(bull[2])}")
    else:
        band_lines.append("Bull: N/A")
    if bear:
        band_lines.append(f"Bear: {fmt_num(bear[0])} -> {fmt_num(bear[1])} | Near Support: {fmt_num(bear[2])}")
    else:
        band_lines.append("Bear: N/A")

    band_lines.append(f"RR (bull1 vs ATR stop): {'N/A' if rr is None else f'1:{rr:.2f}'} | Stop(ATR): {fmt_num(stop)}")
    y = _pdf_write_lines(c, band_lines, left, y, 12, bottom)
    y -= 6

    # Levels
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Levels (Approx.)"); y -= 14
    c.setFont("Helvetica", 9)
    if levels:
        lv_lines = [", ".join([fmt_num(x) for x in levels[i:i+10]]) for i in range(0, len(levels), 10)]
    else:
        lv_lines = ["N/A"]
    y = _pdf_write_lines(c, lv_lines, left, y, 11, bottom)
    y -= 6

    # Backtest
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Backtest Summary (Long-only)"); y -= 14
    c.setFont("Helvetica", 9)
    bm = backtest_metrics or {}
    y = _pdf_write_lines(
        c,
        [
            f"Total Return: {fmt_pct(bm.get('Total Return'))} | Ann Return: {fmt_pct(bm.get('Annualized Return'))} | Ann Vol: {fmt_pct(bm.get('Annualized Volatility'))}",
            f"Sharpe: {fmt_num(bm.get('Sharpe'), 2)} | Sortino: {fmt_num(bm.get('Sortino'), 2)} | Calmar: {fmt_num(bm.get('Calmar'), 2)}",
            f"Max DD: {fmt_pct(bm.get('Max Drawdown'))} | Trades: {bm.get('Trades','')} | Win Rate: {fmt_pct(bm.get('Win Rate'))} | Profit Factor: {fmt_num(bm.get('Profit Factor'), 2)}",
        ],
        left, y, 12, bottom
    )
    y -= 6

    # Fundamental Screener Snapshot (selected ticker row)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Fundamental Screener Snapshot (Selected Ticker)"); y -= 14
    c.setFont("Helvetica", 9)
    if fa_row:
        keys = [
            "ticker","longName","FA_pass","FA_score","FA_ok_count","FA_coverage",
            "sector","industry","trailingPE","forwardPE","pegRatio","priceToSalesTrailing12Months","priceToBook",
            "returnOnEquity","operatingMargins","profitMargins","debtToEquity","revenueGrowth","earningsGrowth","marketCap"
        ]
        lines = [f"{k}: {fa_row.get(k)}" for k in keys if k in fa_row]
        if not lines:
            lines = ["(No fields)"]
    else:
        lines = ["Screener satırı bulunamadı (screener çalıştırılmamış olabilir)."]
    y = _pdf_write_lines(c, lines, left, y, 11, bottom)
    y -= 6

    # Trades (first rows)
    if trades_df is not None and not trades_df.empty:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Trades (first 25 rows)"); y -= 14
        c.setFont("Helvetica", 8)
        td = trades_df.copy().head(25)
        cols = [cc for cc in ["entry_date","entry_price","exit_date","exit_price","exit_reason","pnl","return_%","holding_days"] if cc in td.columns]
        header = " | ".join(cols)
        y = _pdf_write_lines(c, [header], left, y, 10, bottom)
        for _, r in td.iterrows():
            row_txt = " | ".join([str(r.get(k, ""))[:18] for k in cols])
            y = _pdf_write_lines(c, [row_txt], left, y, 10, bottom)
        y -= 6

    # Charts pages (optional)
    chart_added = False
    if include_charts and figs:
        for name, fig in figs.items():
            img = _plotly_fig_to_png_bytes(fig)
            if not img:
                continue
            chart_added = True
            c.showPage()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(left, top, f"Chart: {name}")
            img_reader = ImageReader(BytesIO(img))
            usable_w = (right - left)
            usable_h = (H - 3.2*cm - 2.0*cm)
            c.drawImage(img_reader, left, 2.0*cm, width=usable_w, height=usable_h, preserveAspectRatio=True, anchor='c')

    # If charts were requested but couldn't be added, add a note
    if include_charts and figs and not chart_added:
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(left, top, "Charts could not be embedded.")
        c.setFont("Helvetica", 10)
        c.drawString(left, top - 18, "Reason: Plotly image export needs 'kaleido' in requirements.txt.")
        c.drawString(left, top - 34, "Fallback: Download HTML report and print to PDF (keeps charts).")

    c.save()
    buf.seek(0)
    return buf.read()

# =============================
# RR helper
# =============================
def rr_from_atr_stop(latest_row: pd.Series, tp_dict: dict, cfg: dict):
    close = float(latest_row["Close"])
    atrv = float(latest_row.get("ATR", np.nan)) if pd.notna(latest_row.get("ATR", np.nan)) else np.nan
    if not np.isfinite(atrv) or atrv <= 0:
        return {"rr": None, "stop": None, "risk": None, "reward": None}

    stop = close - float(cfg["atr_stop_mult"]) * atrv
    risk = close - stop

    if not tp_dict.get("bull"):
        return {"rr": None, "stop": stop, "risk": risk, "reward": None}

    bull1, _bull2, _r1 = tp_dict["bull"]
    reward = float(bull1) - close

    if risk <= 0 or reward <= 0:
        return {"rr": None, "stop": stop, "risk": risk, "reward": reward}

    return {"rr": float(reward / risk), "stop": float(stop), "risk": float(risk), "reward": float(reward)}

def fmt_rr(rr):
    if rr is None or (isinstance(rr, float) and (not np.isfinite(rr))):
        return "N/A"
    return f"1:{rr:.2f}"

def pct_dist(level: float, base: float):
    if level is None or not np.isfinite(level) or base == 0:
        return None
    return (level / base - 1.0) * 100.0

# =============================
# UI State
# =============================
st.title("📈 FA→TA Trading Uygulaması + 🤖 AI Analiz")
st.caption("Önce fundamental ile evreni daralt, sonra teknik analizle giriş/çıkış zamanla. Otomatik emir göndermez.")

if "screener_df" not in st.session_state:
    st.session_state.screener_df = pd.DataFrame()
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = [{"role": "assistant", "content": "Sorunu yaz: örn. “Riskler ne, hedef bant ne, hangi şartta çıkarım?”"}]
if "ta_ran" not in st.session_state:
    st.session_state.ta_ran = False

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("Piyasa")
    market = st.selectbox("Market", ["USA", "BIST"], index=0)

    st.header("1) Fundamental Screener (opsiyonel)")
    use_fa_default = True
    use_fa = st.checkbox("Fundamental filtreyi kullan", value=use_fa_default)
    fa_mode = st.selectbox("Fundamental Mod", ["Quality", "Value", "Growth"], index=0, disabled=(not use_fa))

    st.caption("Eşikler (Genel) — BIST'te coverage düşük olabilir")
    roe = st.slider("ROE min", 0.0, 0.40, 0.15, 0.01, disabled=(not use_fa))
    op_margin = st.slider("Operating Margin min", 0.0, 0.40, 0.10, 0.01, disabled=(not use_fa))
    profit_margin = st.slider("Profit Margin min", 0.0, 0.40, 0.08, 0.01, disabled=(not use_fa))
    dte = st.slider("Debt/Equity max", 0.0, 3.0, 1.0, 0.05, disabled=(not use_fa))
    fpe = st.slider("Forward P/E max", 0.0, 60.0, 20.0, 1.0, disabled=(not use_fa))
    peg = st.slider("PEG max", 0.0, 5.0, 1.5, 0.1, disabled=(not use_fa))
    ps = st.slider("P/S max", 0.0, 30.0, 6.0, 0.5, disabled=(not use_fa))
    pb = st.slider("P/B max", 0.0, 30.0, 6.0, 0.5, disabled=(not use_fa))
    rev_g = st.slider("Revenue Growth min", 0.0, 0.50, 0.10, 0.01, disabled=(not use_fa))
    earn_g = st.slider("Earnings Growth min", 0.0, 0.50, 0.10, 0.01, disabled=(not use_fa))

    min_score = st.slider("Min Fundamental Score", 0, 100, 60, 1, disabled=(not use_fa))
    min_ok = st.slider("Min OK sayısı", 1, 5, 3, 1, disabled=(not use_fa))
    min_coverage = st.slider(
        "Min Coverage (NaN olmayan metrik sayısı)", 1, 5, 3, 1, disabled=(not use_fa),
        help="BIST'te yfinance bazı metrikleri boş getirebilir. Coverage düşükse PASS zorlaşır."
    )

    thresholds = {
        "roe": roe, "op_margin": op_margin, "profit_margin": profit_margin,
        "dte": dte, "fpe": fpe, "peg": peg, "ps": ps, "pb": pb,
        "rev_g": rev_g, "earn_g": earn_g,
        "min_score": min_score, "min_ok": min_ok, "min_coverage": min_coverage,
    }

    # Universe
    if market == "USA":
        sp = get_sp500_tickers()
        ndx = get_nasdaq100_tickers()
        universe = sorted(list(set(sp + ndx)))
        st.caption(f"Universe: S&P500 + Nasdaq100 (unique: {len(universe)})")
    else:
        bist = get_bist100_tickers()
        universe = sorted(list(set(bist)))
        st.caption(f"Universe: BIST100 (unique: {len(universe)})")

    run_screener = st.button("🔎 Screener Çalıştır", type="secondary", disabled=(not use_fa))

    st.divider()
    st.header("2) Teknik Analiz + Backtest")
    preset_name = st.selectbox("Teknik Mod", list(PRESETS.keys()), index=1)

    st.subheader("Sembol (TA)")
    if st.session_state.selected_ticker:
        st.caption(f"Screener seçimi: **{st.session_state.selected_ticker}**")
        raw_ticker = st.text_input("Sembol", value=st.session_state.selected_ticker)
    else:
        raw_ticker = st.text_input("Sembol (USA: AAPL, SPY) / BIST: THYAO", value="AAPL" if market == "USA" else "THYAO")

    ticker = normalize_ticker(raw_ticker, market)

    st.subheader("Zaman Aralığı")
    interval = st.selectbox("Interval", ["1d", "1h", "30m"], index=0)
    period = st.selectbox("Periyot", ["6mo", "1y", "2y", "5y", "10y"], index=3)

    st.divider()
    st.subheader("Teknik Parametreler")
    ema_fast = st.number_input("EMA Fast (trend içi)", min_value=5, max_value=100, value=50, step=1)
    ema_slow = st.number_input("EMA Slow (trend filtresi)", min_value=50, max_value=400, value=200, step=1)
    rsi_period = st.number_input("RSI Period", min_value=5, max_value=30, value=14, step=1)
    bb_period = st.number_input("Bollinger Period", min_value=10, max_value=50, value=20, step=1)
    bb_std = st.number_input("Bollinger Std", min_value=1.0, max_value=3.5, value=2.0, step=0.1)
    atr_period = st.number_input("ATR Period", min_value=5, max_value=30, value=14, step=1)
    vol_sma = st.number_input("Volume SMA", min_value=5, max_value=60, value=20, step=1)

    st.subheader("Market Filter")
    use_spy_filter = st.checkbox("SPY > EMA200 filtresi (sadece USA)", value=True, disabled=(market != "USA"))

    st.subheader("Risk / Backtest")
    initial_capital = st.number_input("Başlangıç Sermayesi", min_value=100.0, value=10000.0, step=500.0)
    risk_per_trade = st.slider("Trade başı risk (equity %)", min_value=0.002, max_value=0.05, value=0.01, step=0.001)
    commission_bps = st.number_input("Komisyon (bps)", min_value=0.0, value=5.0, step=1.0)
    slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, value=2.0, step=1.0)
    risk_free_annual = st.number_input("Risk-Free (yıllık, ör: 0.05 = %5)", min_value=0.0, value=0.0, step=0.01)

    st.divider()
    st.header("3) AI Ayarları")
    ai_on = st.checkbox("AI Chat aktif", value=True)
    ai_model = st.text_input("Model", value="gpt-4.1-mini", help="OpenAI model adı")
    ai_temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    run_btn = st.button("🚀 Teknik Analizi Çalıştır", type="primary")
    if run_btn:
        st.session_state.ta_ran = True

# -----------------------------
# Fundamental screener action
# -----------------------------
if run_screener and use_fa:
    with st.spinner(f"Fundamental veriler çekiliyor ({market})..."):
        rows = []
        for tk in universe:
            tk_norm = normalize_ticker(tk, market)
            f = fetch_fundamentals_generic(tk_norm, market=market)
            score, breakdown, passed = fundamental_score_row(f, fa_mode, thresholds)
            f["FA_score"] = score
            f["FA_pass"] = passed
            f["FA_ok_count"] = sum(1 for v in breakdown.values() if v.get("available") and v.get("ok"))
            f["FA_coverage"] = sum(1 for v in breakdown.values() if v.get("available"))
            rows.append(f)

        sdf = pd.DataFrame(rows)
        if not sdf.empty:
            sdf["FA_pass_int"] = sdf["FA_pass"].astype(int)
            sdf = sdf.sort_values(["FA_pass_int", "FA_score", "FA_coverage"], ascending=[False, False, False]).drop(columns=["FA_pass_int"])
        st.session_state.screener_df = sdf.copy()

# -----------------------------
# Data loader
# -----------------------------
cfg = {
    "ema_fast": ema_fast, "ema_slow": ema_slow, "rsi_period": rsi_period,
    "bb_period": bb_period, "bb_std": bb_std, "atr_period": atr_period, "vol_sma": vol_sma,
    "initial_capital": initial_capital, "risk_per_trade": risk_per_trade,
    "commission_bps": commission_bps, "slippage_bps": slippage_bps,
}
cfg.update(PRESETS[preset_name])

@st.cache_data(show_spinner=False)
def load_data_cached(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    return _flatten_yf(df)

# If TA not ran yet: show screener (if any) and stop
if not st.session_state.ta_ran:
    if use_fa and not st.session_state.screener_df.empty:
        st.subheader(f"🧾 Fundamental Screener Sonuçları ({market})")
        sdf = st.session_state.screener_df.copy()

        show_cols = [
            "ticker", "longName", "FA_pass", "FA_score", "FA_ok_count", "FA_coverage",
            "sector", "industry",
            "trailingPE", "forwardPE", "pegRatio", "priceToSalesTrailing12Months", "priceToBook",
            "returnOnEquity", "operatingMargins", "profitMargins", "debtToEquity",
            "revenueGrowth", "earningsGrowth",
            "marketCap"
        ]
        sdf_show = sdf[[c for c in show_cols if c in sdf.columns]].copy()
        st.dataframe(sdf_show, use_container_width=True, height=360)

        pass_list = sdf.loc[sdf["FA_pass"] == True, "ticker"].tolist()
        if len(pass_list) == 0:
            st.warning("Bu eşiklerle PASS çıkan hisse yok. Eşikleri gevşet / mode değiştir / coverage düşür.")
        else:
            st.success(f"PASS sayısı: {len(pass_list)}")
            picked = st.selectbox("PASS listesinden hisse seç (TA’ya gönder)", pass_list, index=0)
            if st.button("➡️ Seçimi Teknik Analize Aktar"):
                st.session_state.selected_ticker = picked
                st.rerun()

    st.info("Soldan ayarları yapıp **Teknik Analizi Çalıştır**’a bas.")
    st.stop()

# =============================
# Run TA pipeline
# =============================
market_filter_ok = True
if market == "USA" and use_spy_filter:
    with st.spinner("SPY rejimi kontrol ediliyor..."):
        market_filter_ok = get_spy_regime_ok()

with st.spinner(f"Veri indiriliyor: {ticker}"):
    df_raw = load_data_cached(ticker, period, interval)

if df_raw.empty:
    st.error(
        f"Veri gelmedi: {ticker}\n\n"
        "BIST için THYAO formatı otomatik THYAO.IS olur.\n"
        "BIST’te 1d interval ve 5y/10y periyot daha stabil."
    )
    st.stop()

if len(df_raw) < 260 and interval == "1d":
    st.warning("Günlükte 260 bar altı: metrikler daha oynak olabilir. (5y/10y seçmek daha iyi)")

df = build_features(df_raw, cfg)
df, checkpoints = signal_with_checkpoints(df, cfg, market_filter_ok=market_filter_ok)
latest = df.iloc[-1]

live = get_live_price(ticker)
live_price = live.get("last_price", np.nan)

if int(latest["ENTRY"]) == 1:
    rec = "AL"
elif int(latest["EXIT"]) == 1:
    rec = "SAT"
else:
    rec = "İZLE (Güçlü Trend)" if latest["SCORE"] >= 80 else ("BEKLE (Orta)" if latest["SCORE"] >= 60 else "UZAK DUR")

eq, tdf, metrics = backtest_long_only(df, cfg, risk_free_annual=risk_free_annual)
tp = target_price_band(df)
rr_info = rr_from_atr_stop(latest, tp, cfg)

# =============================
# Build figures once (Dashboard + Export)
# =============================
fig_price = go.Figure()
fig_price.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA Fast"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA Slow"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(dash="dot")))
fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], name="BB Mid", line=dict(dash="dot")))
fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(dash="dot")))
entries = df[df["ENTRY"] == 1]
exits = df[df["EXIT"] == 1]
fig_price.add_trace(go.Scatter(x=entries.index, y=entries["Close"], mode="markers", name="ENTRY", marker=dict(symbol="triangle-up", size=10)))
fig_price.add_trace(go.Scatter(x=exits.index, y=exits["Close"], mode="markers", name="EXIT", marker=dict(symbol="triangle-down", size=10)))
fig_price.update_layout(height=600, xaxis_rangeslider_visible=False)

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
fig_rsi.add_hline(y=70)
fig_rsi.add_hline(y=30)
fig_rsi.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"))
fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
fig_macd.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))

fig_atr = go.Figure()
fig_atr.add_trace(go.Scatter(x=df.index, y=df["ATR_PCT"] * 100, name="ATR%"))
fig_atr.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="%")

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity"))
fig_eq.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))

figs_for_report = {
    "Price + EMA + Bollinger + Signals": fig_price,
    "RSI": fig_rsi,
    "MACD": fig_macd,
    "ATR%": fig_atr,
    "Equity Curve": fig_eq,
}

# =============================
# Tabs
# =============================
tab_dash, tab_export = st.tabs(["📊 Dashboard", "📄 Rapor (PDF/HTML)"])

with tab_dash:
    # Screener display (if available)
    if use_fa and not st.session_state.screener_df.empty:
        st.subheader(f"🧾 Fundamental Screener Sonuçları ({market})")
        sdf = st.session_state.screener_df.copy()

        show_cols = [
            "ticker", "longName", "FA_pass", "FA_score", "FA_ok_count", "FA_coverage",
            "sector", "industry",
            "trailingPE", "forwardPE", "pegRatio", "priceToSalesTrailing12Months", "priceToBook",
            "returnOnEquity", "operatingMargins", "profitMargins", "debtToEquity",
            "revenueGrowth", "earningsGrowth",
            "marketCap"
        ]
        sdf_show = sdf[[c for c in show_cols if c in sdf.columns]].copy()
        st.dataframe(sdf_show, use_container_width=True, height=360)

        pass_list = sdf.loc[sdf["FA_pass"] == True, "ticker"].tolist()
        if len(pass_list) == 0:
            st.warning("Bu eşiklerle PASS çıkan hisse yok. Eşikleri gevşet / mode değiştir / coverage düşür.")
        else:
            st.success(f"PASS sayısı: {len(pass_list)}")
            picked = st.selectbox("PASS listesinden hisse seç (TA’ya gönder)", pass_list, index=0)
            if st.button("➡️ Seçimi Teknik Analize Aktar"):
                st.session_state.selected_ticker = picked
                st.rerun()

    # Summary metrics
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Market", market)
    c2.metric("Sembol", ticker)
    c3.metric("Daily Close (bar)", f"{latest['Close']:.2f}")
    c4.metric("Live/Last", f"{live_price:.2f}" if np.isfinite(live_price) else "N/A")
    c5.metric("Skor", f"{latest['SCORE']:.0f}/100")
    c6.metric("Sinyal", rec)
    c7.metric("SPY Rejim", "BULL ✅" if (market == "USA" and market_filter_ok) else ("BEAR ❌" if market == "USA" else "N/A"))

    st.caption("Not: Daily Close (1d bar) ile Live/Last farklı olabilir. Piyasa açıkken 1d bar kapanışı güncellenmez.")

    st.subheader("✅ Kontrol Noktaları (Son Bar)")
    cp_cols = st.columns(3)
    for i, (k, v) in enumerate(checkpoints.items()):
        with cp_cols[i % 3]:
            st.write(("🟢 " if v else "🔴 ") + k)

    # Target band + RR
    st.subheader("🎯 Hedef Fiyat Bandı (Senaryo)")
    base_px = float(tp["base"])
    rr_str = fmt_rr(rr_info.get("rr"))

    bcol1, bcol2, bcol3 = st.columns(3)
    bcol1.metric("Base", f"{base_px:.2f}")

    if tp.get("bull"):
        bull1, bull2, r1 = tp["bull"]
        bcol2.metric("Bull Band", f"{bull1:.2f} → {bull2:.2f}")
        if r1 is not None and np.isfinite(r1):
            bcol2.caption(f"Yakın direnç: {r1:.2f} ({pct_dist(r1, base_px):+.2f}%)")
    else:
        bcol2.metric("Bull Band", "N/A")
        r1 = None

    if tp.get("bear"):
        bear1, bear2, s1 = tp["bear"]
        bcol3.metric("Bear Band", f"{bear1:.2f} → {bear2:.2f}  |  RR {rr_str}")
        if s1 is not None and np.isfinite(s1):
            bcol3.caption(f"Yakın destek: {s1:.2f} ({pct_dist(s1, base_px):+.2f}%)")
    else:
        bcol3.metric("Bear Band", f"N/A  |  RR {rr_str}")
        s1 = None

    def render_levels_marked(levels: List[float], base: float, s1, r1):
        lines = []
        for lv in (levels or []):
            tag = ""
            if s1 is not None and np.isfinite(s1) and abs(float(lv) - float(s1)) < 1e-9:
                tag = " 🟩 Yakın Destek"
            if r1 is not None and np.isfinite(r1) and abs(float(lv) - float(r1)) < 1e-9:
                tag = " 🟥 Yakın Direnç"
            dist = pct_dist(float(lv), base)
            dist_txt = f"{dist:+.2f}%" if dist is not None else ""
            lines.append(f"- {float(lv):.2f}  ({dist_txt}){tag}")
        return "\n".join(lines) if lines else "_Seviye yok_"

    with st.expander("Seviye listesi (yaklaşık) — işaretli + fiyata uzaklık %", expanded=False):
        st.markdown(render_levels_marked(tp.get("levels", []), base_px, s1, r1))

    # Charts
    st.subheader("📊 Fiyat + EMA + Bollinger + Sinyaller")
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("📉 RSI / MACD / ATR%")
    colA, colB, colC = st.columns(3)
    colA.plotly_chart(fig_rsi, use_container_width=True)
    colB.plotly_chart(fig_macd, use_container_width=True)
    colC.plotly_chart(fig_atr, use_container_width=True)

    st.subheader("🧪 Backtest Özeti (Long-only)")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Return", f"{metrics['Total Return']*100:.1f}%")
    m2.metric("Ann Return", f"{metrics['Annualized Return']*100:.1f}%")
    m3.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
    m4.metric("Max DD", f"{metrics['Max Drawdown']*100:.1f}%")
    m5.metric("Trades", f"{metrics['Trades']}")
    m6.metric("Win Rate", f"{metrics['Win Rate']*100:.1f}%")

    with st.expander("Trade listesi", expanded=False):
        st.dataframe(tdf, use_container_width=True, height=240)

    with st.expander("Equity curve", expanded=False):
        st.plotly_chart(fig_eq, use_container_width=True)

    # AI Chat
    st.subheader("🤖 AI Analiz (Chat)")
    if not ai_on:
        st.info("AI Chat kapalı (soldan açabilirsin).")
    else:
        if not st.secrets.get("OPENAI_API_KEY", ""):
            st.warning("OPENAI_API_KEY bulunamadı. Streamlit Cloud > Secrets'e OPENAI_API_KEY=... ekleyin.")
        else:
            ai_ctx = build_ai_context(ticker, market, latest, checkpoints, metrics, tp, live, df)
            system_msg = {
                "role": "system",
                "content": (
                    "Sen bir finansal analiz asistanısın. Kullanıcının verdiği veriler üzerinden eğitim amaçlı yorum yap. "
                    "Kesin yatırım tavsiyesi verme. Çıktıda: (1) Özet, (2) Riskler, (3) Invalidation koşulları, (4) Senaryo bandı. "
                    "Kısa, net, maddeli yaz."
                ),
            }

            with st.expander("AI Chat", expanded=True):
                user_msg = st.text_input("Sorun:", value="", placeholder="Örn: Riskler ne, hedef bant ne, invalidation nedir?")
                if st.button("AI'ya Sor") and user_msg.strip():
                    st.session_state.ai_messages.append({"role": "user", "content": user_msg.strip()})
                    messages = [
                        system_msg,
                        {"role": "user", "content": "Context JSON:\n" + json.dumps(ai_ctx, ensure_ascii=False)},
                    ]
                    tail = st.session_state.ai_messages[-6:]
                    messages += tail
                    try:
                        reply = call_openai(messages, model=ai_model, temperature=ai_temp)
                        st.session_state.ai_messages.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(f"AI hata: {e}")

                for m in st.session_state.ai_messages:
                    st.write(f"**{m['role'].upper()}**: {m['content']}")

with tab_export:
    st.subheader("📄 Rapor İndir (En sorunsuz: HTML → tarayıcıdan PDF)")
    st.caption("HTML rapor: grafikler %100 gelir. PDF: reportlab + (grafikler için) kaleido varsa grafikleri gömer. Yoksa PDF metin ağırlıklı olur.")

    include_charts = st.checkbox("Rapor grafikleri dahil et", value=True)
    include_trades = st.checkbox("Trade listesi dahil et (ilk 25)", value=True)

    # Build FA row (from screener + current fundamentals)
    with st.spinner("Fundamental + screener satırı hazırlanıyor..."):
        f_single = fetch_fundamentals_generic(ticker, market=market)
        f_score, f_breakdown, f_pass = fundamental_score_row(f_single, fa_mode, thresholds)
        fa_eval = {
            "mode": fa_mode,
            "score": f_score,
            "passed": f_pass,
            "ok_cnt": sum(1 for v in f_breakdown.values() if v.get("available") and v.get("ok")),
            "coverage": sum(1 for v in f_breakdown.values() if v.get("available")),
        }
        screener_row = find_screener_row(st.session_state.get("screener_df", pd.DataFrame()), ticker)
        fa_row = merge_fa_row(screener_row, f_single, fa_eval)

    meta = {
        "market": market,
        "ticker": ticker,
        "interval": interval,
        "period": period,
        "preset": preset_name,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "rsi_period": rsi_period,
        "bb_period": bb_period,
        "bb_std": bb_std,
        "atr_period": atr_period,
        "vol_sma": vol_sma,
    }

    ta_summary = {
        "rec": rec,
        "close": fmt_num(float(latest["Close"]), 2),
        "live": fmt_num(float(live_price), 2) if np.isfinite(live_price) else "N/A",
        "score": fmt_num(float(latest.get("SCORE", np.nan)), 0),
        "rsi": fmt_num(float(latest.get("RSI", np.nan)), 2),
        "ema50": fmt_num(float(latest.get("EMA50", np.nan)), 2),
        "ema200": fmt_num(float(latest.get("EMA200", np.nan)), 2),
        "atr_pct": fmt_pct(float(latest.get("ATR_PCT", np.nan))) if pd.notna(latest.get("ATR_PCT", np.nan)) else "N/A",
    }

    # Always provide HTML (most robust + charts)
    html_bytes = build_html_report(
        title=f"FA→TA Trading Report - {ticker}",
        meta=meta,
        checkpoints=checkpoints,
        metrics=metrics,
        tp=tp,
        rr_info=rr_info,
        figs=(figs_for_report if include_charts else {}),
        fa_row=fa_row
    )
    st.download_button(
        "⬇️ HTML İndir (Önerilen) — Tarayıcıdan PDF’ye Yazdır",
        data=html_bytes,
        file_name=f"{ticker}_FA_TA_report.html",
        mime="text/html",
        use_container_width=True
    )

    st.divider()

    if not REPORTLAB_OK:
        st.warning("Doğrudan PDF için 'reportlab' gerekli. requirements.txt içine `reportlab` ekleyip redeploy edersen PDF butonu da aktif olur.")
    else:
        # PDF generate (may or may not embed charts depending on kaleido)
        if st.button("🧾 PDF Oluştur (reportlab)", use_container_width=True):
            with st.spinner("PDF oluşturuluyor..."):
                pdf_bytes = generate_pdf_report(
                    title=f"FA→TA Trading Report - {ticker}",
                    subtitle="Educational analysis (not investment advice). Generated from Streamlit dashboard outputs.",
                    meta=meta,
                    checkpoints=checkpoints,
                    ta_summary=ta_summary,
                    target_band=tp,
                    rr_info=rr_info,
                    backtest_metrics=metrics,
                    fa_row=fa_row,
                    levels=tp.get("levels", []),
                    trades_df=(tdf if include_trades else None),
                    figs=(figs_for_report if include_charts else None),
                    include_charts=include_charts
                )

            if pdf_bytes:
                st.success("PDF hazır ✅")
                st.download_button(
                    "⬇️ PDF İndir",
                    data=pdf_bytes,
                    file_name=f"{ticker}_FA_TA_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.info("Grafikler PDF’e gelmiyorsa: requirements.txt içine `kaleido` ekle. (HTML raporda grafikler her zaman gelir.)")
            else:
                st.error("PDF üretilemedi. HTML raporu indirip tarayıcıdan PDF’ye yazdırmanı öneririm.")
