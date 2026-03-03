import os
import json
import time
import io
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests

# AI SDKs
from openai import OpenAI
import google.generativeai as genai
from PIL import Image as PILImage

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

st.set_page_config(page_title="FA→TA Trading + AI PRO", layout="wide")

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
    
    mtf_ok = df.get("MTF_OK", pd.Series(True, index=df.index))
    rs_ok = df.get("RS_OK", pd.Series(True, index=df.index))

    w = {"liq": 10, "trend": 20, "rsi": 10, "macd": 10, "vol": 10, "bb": 10, "obv": 10, "mtf": 10, "rs": 10}
    score = (
        w["liq"] * liq_ok.astype(int)
        + w["trend"] * trend_ok.astype(int)
        + w["rsi"] * rsi_ok.astype(int)
        + w["macd"] * macd_ok.astype(int)
        + w["vol"] * vol_ok.astype(int)
        + w["bb"] * (bb_ok | bb_break).astype(int)
        + w["obv"] * obv_ok.astype(int)
        + w["mtf"] * mtf_ok.astype(int)
        + w["rs"] * rs_ok.astype(int)
    ).astype(float)

    entry_triggers = (rsi_cross.astype(int) + macd_turn.astype(int) + bb_break.astype(int)) >= 2
    entry = trend_ok & vol_ok & liq_ok & mtf_ok & rs_ok & entry_triggers & market_filter_ok

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
        "MTF (Haftalık Trend) OK": bool(last.get("MTF_OK", True)),
        "Göreceli Güç (Endekse Karşı) OK": bool(last.get("RS_OK", True)),
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
# Backtest (long-only) + metrics + Dynamic Risk
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
    
    consecutive_losses = 0

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
                
                pnl = cash - trades[-1]["equity_before"]

                trades[-1]["exit_date"] = date
                trades[-1]["exit_price"] = sell_price
                trades[-1]["exit_reason"] = "STOP" if stop_hit else "RULE_EXIT"
                trades[-1]["pnl"] = pnl
                
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                shares = 0.0
                stop = np.nan

        position_value = shares * price * (1 - slippage)
        equity = cash + position_value

        if shares == 0 and entry_sig.iloc[i] == 1 and pd.notna(row["ATR"]) and row["ATR"] > 0:
            current_risk_pct = cfg["risk_per_trade"] / 2.0 if consecutive_losses >= 2 else cfg["risk_per_trade"]
            
            risk_cash = equity * current_risk_pct
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
# Fundamentals (USA + BIST) 
# YFINANCE API (Anti-Ban Bekleme ve Fallback Eklendi)
# =============================
def _fix_debt_to_equity(x: float) -> float:
    if pd.notna(x) and x > 10:
        return x / 100.0
    return x

@st.cache_data(ttl=12 * 3600, show_spinner=False)
def fetch_fundamentals_generic(ticker: str, market: str) -> dict:
    t = yf.Ticker(ticker)
    info = {}
    try:
        # info komutu bulut sunucularında engellenebiliyor.
        info = t.info or {}
    except Exception:
        pass
    
    # Eğer info tamamen boş gelirse (engellenmişsek), bari fiyat ve marketcap alalım
    if not info:
        try:
            fast = t.fast_info
            info["marketCap"] = fast.get("marketCap")
            info["currentPrice"] = fast.get("lastPrice")
        except Exception:
            pass

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
        "sector": info.get("sector", "Bilinmiyor"),
        "industry": info.get("industry", "Bilinmiyor"),
        "longName": info.get("longName", "") or info.get("shortName", ticker),
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
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        dfu = tables[0]
        return sorted(dfu["Symbol"].astype(str).str.upper().tolist())
    except Exception:
        return sorted(list(set(US_TICKERS + US_EXT)))

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_nasdaq100_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        r = requests.get(url, timeout=20)
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

    for url in candidates:
        try:
            r = requests.get(url, timeout=20)
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
        ("ticker", "Ticker"),("longName", "Name"),("FA_pass", "FA_pass"),("FA_score", "FA_score"),
        ("FA_ok_count", "FA_ok_count"),("FA_coverage", "FA_coverage"),("sector", "Sector"),
        ("industry", "Industry"),("trailingPE", "Trailing PE"),("forwardPE", "Forward PE"),
        ("pegRatio", "PEG"),("priceToSalesTrailing12Months", "P/S"),("priceToBook", "P/B"),
        ("returnOnEquity", "ROE"),("operatingMargins", "Op Margin"),("profitMargins", "Profit Margin"),
        ("debtToEquity", "Debt/Equity"),("revenueGrowth", "Revenue Growth"),("earningsGrowth", "Earnings Growth"),
        ("marketCap", "Market Cap"),
    ]

    fa_rows_html = ""
    if fa_row:
        for key, label in show_cols:
            val = fa_row.get(key, "")
            fa_rows_html += f"<tr><td><b>{esc(label)}</b></td><td>{esc(val)}</td></tr>"
    else:
        fa_rows_html = "<tr><td colspan='2'>Screener satırı bulunamadı.</td></tr>"

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
    Generated: {esc(time.strftime('%Y-%m-%d %H:%M:%S'))}
    Market: {esc(meta.get('market'))} | Ticker: {esc(meta.get('ticker'))}
  </div>
  <div class="grid" style="margin-top:14px;">
    <div class="card">
      <h2>Checkpoints</h2>
      <ul>{cp_list}</ul>
    </div>
    <div class="card">
      <h2>Backtest</h2>
      <div>Total Return: {metrics.get('Total Return',0)*100:.1f}%</div>
      <div>Max DD: {metrics.get('Max Drawdown',0)*100:.1f}%</div>
      <div>Win Rate: {metrics.get('Win Rate',0)*100:.1f}%</div>
    </div>
  </div>
  <div class="card" style="margin-top:16px;">
    <h2>Screener Snapshot</h2>
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

def generate_pdf_report(*, title, subtitle, meta, checkpoints, ta_summary, target_band, rr_info, backtest_metrics, fa_row, levels, trades_df, figs, include_charts) -> Optional[bytes]:
    if not REPORTLAB_OK: return None
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    left, right, top, bottom = 1.6 * cm, W - 1.6 * cm, H - 1.6 * cm, 1.6 * cm
    y = top

    c.setFont("Helvetica-Bold", 16); c.drawString(left, y, title[:90]); y -= 18
    c.setFont("Helvetica", 10); c.drawString(left, y, subtitle[:140]); y -= 14
    c.setFont("Helvetica", 9)
    y = _pdf_write_lines(c, [f"Market: {meta.get('market')} | Ticker: {meta.get('ticker')}"], left, y, 12, bottom)
    
    # Backtest
    c.setFont("Helvetica-Bold", 12); c.drawString(left, y, "Backtest Summary"); y -= 14
    c.setFont("Helvetica", 9)
    y = _pdf_write_lines(c, [f"Total Return: {fmt_pct(backtest_metrics.get('Total Return'))} | Max DD: {fmt_pct(backtest_metrics.get('Max Drawdown'))}"], left, y, 12, bottom)

    if include_charts and figs:
        for name, fig in figs.items():
            img = _plotly_fig_to_png_bytes(fig)
            if img:
                c.showPage()
                c.setFont("Helvetica-Bold", 14); c.drawString(left, top, f"Chart: {name}")
                c.drawImage(ImageReader(BytesIO(img)), left, 2.0*cm, width=(right-left), height=(H-5.2*cm), preserveAspectRatio=True, anchor='c')
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
    if rr is None or (isinstance(rr, float) and (not np.isfinite(rr))): return "N/A"
    return f"1:{rr:.2f}"

def pct_dist(level: float, base: float):
    if level is None or not np.isfinite(level) or base == 0: return None
    return (level / base - 1.0) * 100.0

# =============================
# UI State
# =============================
st.title("📈 FA→TA Trading Uygulaması PRO + 🤖 AI Analiz")
st.caption("Gelişmiş Filtreleme, Walk-Forward, Çoklu Zaman Dilimi (MTF) ve Gemini Vision Entegrasyonu")

if "screener_df" not in st.session_state: st.session_state.screener_df = pd.DataFrame()
if "selected_ticker" not in st.session_state: st.session_state.selected_ticker = None
if "ai_messages" not in st.session_state: st.session_state.ai_messages = [{"role": "assistant", "content": "Sorunu yaz: örn. “Riskler ne, hedef bant ne?”"}]
if "ta_ran" not in st.session_state: st.session_state.ta_ran = False
if "heatmap_data" not in st.session_state: st.session_state.heatmap_data = pd.DataFrame()

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("Piyasa")
    market = st.selectbox("Market", ["USA", "BIST"], index=0)

    st.header("1) Fundamental Screener (opsiyonel)")
    use_fa = st.checkbox("Fundamental filtreyi kullan", value=True)
    fa_mode = st.selectbox("Fundamental Mod", ["Quality", "Value", "Growth"], index=0, disabled=(not use_fa))
    
    with st.expander("Eşik Ayarları"):
        st.info("Eğer tabloda veriler None geliyorsa, Min Coverage'ı 1 veya 0 yapın.")
        roe = st.slider("ROE min", 0.0, 0.40, 0.15, 0.01)
        op_margin = st.slider("Operating Margin min", 0.0, 0.40, 0.10, 0.01)
        profit_margin = st.slider("Profit Margin min", 0.0, 0.40, 0.08, 0.01)
        dte = st.slider("Debt/Equity max", 0.0, 3.0, 1.0, 0.05)
        fpe = st.slider("Forward P/E max", 0.0, 60.0, 20.0, 1.0)
        peg = st.slider("PEG max", 0.0, 5.0, 1.5, 0.1)
        ps = st.slider("P/S max", 0.0, 30.0, 6.0, 0.5)
        pb = st.slider("P/B max", 0.0, 30.0, 6.0, 0.5)
        rev_g = st.slider("Revenue Growth min", 0.0, 0.50, 0.10, 0.01)
        earn_g = st.slider("Earnings Growth min", 0.0, 0.50, 0.10, 0.01)
        min_score = st.slider("Min Fundamental Score", 0, 100, 40, 1)
        min_ok = st.slider("Min OK sayısı", 1, 5, 2, 1)
        min_coverage = st.slider("Min Coverage", 0, 5, 1, 1)

    thresholds = {"roe": roe, "op_margin": op_margin, "profit_margin": profit_margin, "dte": dte, "fpe": fpe, "peg": peg, "ps": ps, "pb": pb, "rev_g": rev_g, "earn_g": earn_g, "min_score": min_score, "min_ok": min_ok, "min_coverage": min_coverage}

    if market == "USA":
        universe = sorted(list(set(get_sp500_tickers() + get_nasdaq100_tickers())))
    else:
        universe = sorted(list(set(get_bist100_tickers())))

    run_screener = st.button("🔎 Screener Çalıştır", type="secondary", disabled=(not use_fa))

    st.divider()
    st.header("2) Teknik Analiz + Backtest")
    preset_name = st.selectbox("Teknik Mod", list(PRESETS.keys()), index=1)

    st.subheader("Sembol (TA)")
    raw_ticker = st.text_input("Sembol", value=st.session_state.selected_ticker if st.session_state.selected_ticker else ("AAPL" if market == "USA" else "THYAO"))
    ticker = normalize_ticker(raw_ticker, market)

    interval = st.selectbox("Interval", ["1d", "1h", "30m"], index=0)
    period = st.selectbox("Periyot", ["6mo", "1y", "2y", "5y", "10y"], index=3)

    with st.expander("Teknik Parametreler"):
        ema_fast = st.number_input("EMA Fast", value=50)
        ema_slow = st.number_input("EMA Slow", value=200)
        rsi_period = st.number_input("RSI Period", value=14)
        bb_period = st.number_input("Bollinger Period", value=20)
        bb_std = st.number_input("Bollinger Std", value=2.0)
        atr_period = st.number_input("ATR Period", value=14)
        vol_sma = st.number_input("Volume SMA", value=20)

    st.subheader("Gelişmiş Strateji Özellikleri")
    use_spy_filter = st.checkbox("Makro Endeks Filtresi (Sadece USA)", value=True, disabled=(market != "USA"))
    do_walk_forward = st.checkbox("Walk-Forward Analizi (Backtest'i Böl)", value=False, help="%70 Train (Geçmiş) - %30 Test (Yakın Zaman)")

    with st.expander("Risk / Backtest Ayarları"):
        initial_capital = st.number_input("Başlangıç Sermayesi", value=10000.0)
        risk_per_trade = st.slider("Trade başı risk (equity %)", 0.002, 0.05, 0.01)
        commission_bps = st.number_input("Komisyon (bps)", value=5.0)
        slippage_bps = st.number_input("Slippage (bps)", value=2.0)
        risk_free_annual = st.number_input("Risk-Free", value=0.0)

    st.divider()
    st.header("3) AI Ayarları")
    ai_on = st.checkbox("OpenAI Metin Chat Aktif", value=True)
    ai_model = st.text_input("OpenAI Model", value="gpt-4o-mini")
    st.caption("Gemini Vision (Grafik Okuma) API Key:")
    gemini_key_input = st.text_input("Gemini API Key", type="password", help="Grafiği yapay zekaya okutmak için gerekli (ÜCRETSİZDİR)")

    run_btn = st.button("🚀 Teknik Analizi Çalıştır", type="primary")
    if run_btn: st.session_state.ta_ran = True

# -----------------------------
# Fundamental screener action & Heatmap Data Pre-load
# -----------------------------
if run_screener and use_fa:
    with st.spinner(f"Fundamental veriler çekiliyor ({market}). Çok hızlı sorgularda Yahoo engelleyebilir, yavaş çekiliyor..."):
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
            time.sleep(0.2) # Anti-Ban Bekleme Süresi

        sdf = pd.DataFrame(rows)
        if not sdf.empty:
            sdf["FA_pass_int"] = sdf["FA_pass"].astype(int)
            sdf = sdf.sort_values(["FA_pass_int", "FA_score", "FA_coverage"], ascending=[False, False, False]).drop(columns=["FA_pass_int"])
        st.session_state.screener_df = sdf.copy()

    # Heatmap hesaplama eklentisi
    with st.spinner("Isı Haritası verileri (Son 1 Ay) hesaplanıyor..."):
        if not sdf.empty:
            tk_list = sdf["ticker"].tolist()
            try:
                hist = yf.download(tk_list, period="1mo", interval="1d", progress=False)["Close"]
                if isinstance(hist, pd.Series): 
                    hist = hist.to_frame()
                
                ret = (hist.iloc[-1] / hist.iloc[0] - 1) * 100
                ret_df = ret.reset_index()
                ret_df.columns = ["ticker", "momentum"]

                merged = pd.merge(sdf, ret_df, on="ticker")
                merged = merged.dropna(subset=["momentum", "sector"])
                
                merged["marketCap_viz"] = merged["marketCap"].fillna(0)
                merged.loc[merged["marketCap_viz"] <= 0, "marketCap_viz"] = 1
                st.session_state.heatmap_data = merged
            except Exception as e:
                st.error(f"Isı haritası oluşturulamadı: {e}")
                st.session_state.heatmap_data = pd.DataFrame()

# =============================
# Layout (Tabs)
# =============================
tab_dash, tab_heatmap, tab_export = st.tabs(["📊 Dashboard & TA", "🗺️ Sektörel Isı Haritası", "📄 Rapor"])

with tab_dash:
    if use_fa and "screener_df" in st.session_state and not st.session_state.screener_df.empty:
        st.subheader(f"🧾 Fundamental Screener Sonuçları ({market})")
        sdf = st.session_state.screener_df.copy()

        show_cols = ["ticker", "longName", "FA_pass", "FA_score", "FA_ok_count", "FA_coverage", "sector", "industry", "trailingPE", "forwardPE", "pegRatio", "priceToSalesTrailing12Months", "priceToBook", "returnOnEquity", "operatingMargins", "profitMargins", "debtToEquity", "revenueGrowth", "earningsGrowth", "marketCap"]
        sdf_show = sdf[[c for c in show_cols if c in sdf.columns]].copy()
        st.dataframe(sdf_show, use_container_width=True, height=360)

        pass_list = sdf.loc[sdf["FA_pass"] == True, "ticker"].tolist()
        if len(pass_list) == 0:
            st.warning("Bu eşiklerle PASS çıkan hisse yok. Sol menüden 'Min Coverage' ayarını 1'e veya 0'a düşürüp tekrar deneyin.")
        else:
            st.success(f"PASS sayısı: {len(pass_list)}")
            picked = st.selectbox("PASS listesinden hisse seç (TA’ya gönder)", pass_list, index=0)
            if st.button("➡️ Seçimi Teknik Analize Aktar"):
                st.session_state.selected_ticker = picked
                st.rerun()
                
    if not st.session_state.ta_ran:
        st.info("Sol menüden ayarları yapıp **Teknik Analizi Çalıştır**’a bas.")

with tab_heatmap:
    st.subheader("🗺️ Sektörel Isı Haritası (1 Aylık Para Akışı)")
    st.write("Piyasadaki sektörlerin son 1 aylık momentumunu ve ağırlıklarını gösterir.")
    if "heatmap_data" in st.session_state and not st.session_state.heatmap_data.empty:
        merged = st.session_state.heatmap_data
        fig_heat = px.treemap(
            merged, 
            path=[px.Constant("Tüm Piyasa"), "sector", "ticker"],
            values="marketCap_viz", 
            color="momentum",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title=f"{market} - Sektörel Momentum (1 Ay)"
        )
        fig_heat.update_layout(height=700, margin=dict(t=50, l=25, r=25, b=25))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Sol menüden **Screener Çalıştır** butonuna bastığınızda ısı haritası otomatik olarak burada belirecektir.")

if not st.session_state.ta_ran:
    st.stop()


# =============================
# Run TA pipeline
# =============================
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

market_filter_ok = True
if market == "USA" and use_spy_filter:
    with st.spinner("SPY rejimi kontrol ediliyor..."):
        market_filter_ok = get_spy_regime_ok()

with st.spinner(f"Ana Veri & MTF & Göreceli Güç İndiriliyor: {ticker}"):
    df_raw = load_data_cached(ticker, period, interval)
    
    if df_raw.empty:
        st.error(f"Veri gelmedi: {ticker}")
        st.stop()
        
    wk_df = load_data_cached(ticker, period, "1wk")
    if not wk_df.empty:
        wk_df["EMA50_wk"] = ema(wk_df["Close"], 50)
        wk_ema_daily = wk_df["EMA50_wk"].reindex(df_raw.index, method="ffill")
        df_raw["EMA50_wk"] = wk_ema_daily
        df_raw["MTF_OK"] = (df_raw["Close"] > df_raw["EMA50_wk"]).fillna(True)
    else:
        df_raw["MTF_OK"] = True

    idx_sym = "SPY" if market == "USA" else "XU100.IS"
    idx_df = load_data_cached(idx_sym, period, interval)
    if not idx_df.empty:
        idx_close = idx_df["Close"].reindex(df_raw.index, method="ffill")
        df_raw["RS"] = df_raw["Close"] / idx_close
        df_raw["RS_EMA50"] = ema(df_raw["RS"], 50)
        df_raw["RS_OK"] = (df_raw["RS"] > df_raw["RS_EMA50"]).fillna(True)
    else:
        df_raw["RS_OK"] = True

df = build_features(df_raw, cfg)
df, checkpoints = signal_with_checkpoints(df, cfg, market_filter_ok=market_filter_ok)
latest = df.iloc[-1]

live = get_live_price(ticker)
live_price = live.get("last_price", np.nan)

if int(latest["ENTRY"]) == 1: rec = "AL"
elif int(latest["EXIT"]) == 1: rec = "SAT"
else: rec = "İZLE (Güçlü Trend)" if latest["SCORE"] >= 80 else ("BEKLE (Orta)" if latest["SCORE"] >= 60 else "UZAK DUR")

if do_walk_forward and len(df) > 100:
    split_idx = int(len(df) * 0.7)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    eq_train, tdf_train, metrics_train = backtest_long_only(df_train, cfg, risk_free_annual)
    eq_test, tdf_test, metrics_test = backtest_long_only(df_test, cfg, risk_free_annual)
    
    eq, tdf, metrics = backtest_long_only(df, cfg, risk_free_annual)
else:
    eq, tdf, metrics = backtest_long_only(df, cfg, risk_free_annual)
    do_walk_forward = False

tp = target_price_band(df)
rr_info = rr_from_atr_stop(latest, tp, cfg)

# =============================
# Build figures
# =============================
fig_price = go.Figure()
fig_price.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA Fast", line=dict(color='blue')))
fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA Slow", line=dict(color='red')))
fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(dash="dot", color='gray')))
fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(dash="dot", color='gray')))
entries = df[df["ENTRY"] == 1]
exits = df[df["EXIT"] == 1]
fig_price.add_trace(go.Scatter(x=entries.index, y=entries["Close"]*0.98, mode="markers", name="ENTRY", marker=dict(symbol="triangle-up", size=12, color="green")))
fig_price.add_trace(go.Scatter(x=exits.index, y=exits["Close"]*1.02, mode="markers", name="EXIT", marker=dict(symbol="triangle-down", size=12, color="red")))
fig_price.update_layout(height=600, xaxis_rangeslider_visible=False, title=f"{ticker} - Fiyat & Sinyaller")

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity", line=dict(color="purple")))
if do_walk_forward:
    fig_eq.add_vline(x=df.index[split_idx], line_dash="dash", line_color="red", annotation_text="Test Başlangıcı")
fig_eq.update_layout(height=320, title="Sermaye Eğrisi (Equity Curve)")

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
fig_rsi.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), title="RSI")

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"))
fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
fig_macd.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), title="MACD")

fig_atr = go.Figure()
fig_atr.add_trace(go.Scatter(x=df.index, y=df["ATR_PCT"] * 100, name="ATR%"))
fig_atr.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), title="ATR %")

# Dash Tab Devamı
with tab_dash:
    st.divider()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Market", market)
    c2.metric("Sembol", ticker)
    c3.metric("Daily Close", f"{latest['Close']:.2f}")
    c4.metric("Live/Last", f"{live_price:.2f}" if np.isfinite(live_price) else "N/A")
    c5.metric("Skor", f"{latest['SCORE']:.0f}/100")
    c6.metric("Sinyal", rec)

    st.subheader("✅ Kontrol Noktaları (Genişletilmiş Filtreler)")
    cp_cols = st.columns(3)
    for i, (k, v) in enumerate(checkpoints.items()):
        with cp_cols[i % 3]:
            st.write(("🟢 " if v else "🔴 ") + k)

    st.plotly_chart(fig_price, use_container_width=True)
    
    st.subheader("📉 RSI / MACD / ATR%")
    colA, colB, colC = st.columns(3)
    colA.plotly_chart(fig_rsi, use_container_width=True)
    colB.plotly_chart(fig_macd, use_container_width=True)
    colC.plotly_chart(fig_atr, use_container_width=True)
    
    # GEMINI VISION
    st.markdown("### 👁️ Gemini Vision: Formasyon ve Grafik Yorumu")
    if st.button("Grafiği Gemini'ye Gönder ve Yorumlat", type="primary"):
        gemini_api = gemini_key_input or st.secrets.get("GEMINI_API_KEY", "")
        if not gemini_api:
            st.error("Lütfen sol menüdeki 'AI Ayarları' kısmına Gemini API Key'inizi girin.")
        else:
            with st.spinner("Grafik işleniyor ve Gemini'ye iletiliyor..."):
                try:
                    img_bytes = fig_price.to_image(format="png", width=1200, height=800, scale=2)
                    img_pil = PILImage.open(io.BytesIO(img_bytes))
                    
                    genai.configure(api_key=gemini_api)
                    model = genai.GenerativeModel("gemini-1.5-pro")
                    prompt = (
                        "Sen profesyonel bir kurumsal portföy yöneticisi ve teknik analistsin. "
                        "Grafikteki fiyat hareketlerini (Price Action), formasyon yapılarını, trendin durumunu (EMA'lara göre), "
                        "ve varsa gözle görülür destek/direnç kırılımlarını analiz et. "
                        "Yorumun kısa, yapılandırılmış, profesyonel tonda olsun."
                    )
                    response = model.generate_content([prompt, img_pil])
                    st.success("Analiz Tamamlandı!")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Grafik işlenemedi. Lütfen uygulamanın yeniden başlatılmasını bekleyin. Hata detayı: {e}")

    st.divider()
    st.subheader("🧪 Backtest Özeti")
    if do_walk_forward:
        st.write("📌 **Walk-Forward Analizi Aktif (Risk Yönetimli)**")
        col1, col2 = st.columns(2)
        with col1:
            st.info("Eğitim Dönemi (Train - İlk %70)")
            st.metric("Total Return", f"{metrics_train['Total Return']*100:.1f}%")
            st.metric("Max Drawdown", f"{metrics_train['Max Drawdown']*100:.1f}%")
            st.metric("Win Rate", f"{metrics_train['Win Rate']*100:.1f}%")
        with col2:
            st.success("Test Dönemi (Out-of-Sample - Son %30)")
            st.metric("Total Return", f"{metrics_test['Total Return']*100:.1f}%")
            st.metric("Max Drawdown", f"{metrics_test['Max Drawdown']*100:.1f}%")
            st.metric("Win Rate", f"{metrics_test['Win Rate']*100:.1f}%")
    else:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Return", f"{metrics['Total Return']*100:.1f}%")
        m2.metric("Ann Return", f"{metrics['Annualized Return']*100:.1f}%")
        m3.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
        m4.metric("Max DD", f"{metrics['Max Drawdown']*100:.1f}%")
        m5.metric("Trades", f"{metrics['Trades']}")
        m6.metric("Win Rate", f"{metrics['Win Rate']*100:.1f}%")

    st.plotly_chart(fig_eq, use_container_width=True)

with tab_export:
    st.subheader("📄 Rapor İndir")
    st.caption("HTML rapor: grafikler %100 gelir. PDF: reportlab + kaleido varsa grafikleri gömer.")

    include_charts = st.checkbox("Rapor grafikleri dahil et", value=True)
    include_trades = st.checkbox("Trade listesi dahil et (ilk 25)", value=True)

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
        "market": market, "ticker": ticker, "interval": interval, "period": period,
        "preset": preset_name, "ema_fast": ema_fast, "ema_slow": ema_slow, "rsi_period": rsi_period,
        "bb_period": bb_period, "bb_std": bb_std, "atr_period": atr_period, "vol_sma": vol_sma,
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
    
    figs_for_report = {
        "Price + EMA + Bollinger + Signals": fig_price,
        "RSI": fig_rsi,
        "MACD": fig_macd,
        "ATR %": fig_atr,
        "Equity Curve": fig_eq,
    }

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
        st.warning("Doğrudan PDF için 'reportlab' gerekli.")
    else:
        if st.button("🧾 PDF Oluştur (reportlab)", use_container_width=True):
            with st.spinner("PDF oluşturuluyor..."):
                pdf_bytes = generate_pdf_report(
                    title=f"FA→TA Trading Report - {ticker}",
                    subtitle="Educational analysis (not investment advice).",
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
