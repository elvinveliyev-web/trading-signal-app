import os
import re
import json
import time
import base64
import datetime
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests

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
# BASE DIR
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def pjoin(*parts) -> str:
    return os.path.join(BASE_DIR, *parts)


# =============================
# Universe Loader
# =============================
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_universe_file(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        toks = re.split(r"[\s,;]+", raw.strip())
        tickers = [t.strip().upper() for t in toks if t.strip()]
        tickers = list(dict.fromkeys(tickers))
        return sorted(tickers)
    except Exception:
        return []


# =============================
# Helpers
# =============================
def normalize_ticker(raw: str, market: str) -> str:
    t = (raw or "").strip().upper()
    if not t:
        return t
    if market == "BIST" and not t.endswith(".IS"):
        t = f"{t}.IS"
    return t


def naked_ticker(raw: str) -> str:
    return (raw or "").strip().upper().replace(".IS", "")


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

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        if len(out.columns.levels) == 2:
            out.columns = [c[0] for c in out.columns]

    required = [c for c in ["Open", "High", "Low", "Close"] if c in out.columns]
    if required:
        out = out.dropna(subset=required)

    return out


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
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.replace([np.inf, -np.inf], np.nan).fillna(50)


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
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
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
# YENİ EKLENEN İNDİKATÖRLER (TRIPLE SCREEN İÇİN)
# =============================
def force_index(close: pd.Series, volume: pd.Series) -> pd.Series:
    return volume * (close - close.shift(1))

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 5, d_period: int = 3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k.fillna(50), d.fillna(50)

def elder_ray(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 13):
    e = ema(close, period)
    bull_power = high - e
    bear_power = low - e
    return e, bull_power, bear_power

def check_bullish_divergence(close: pd.Series, indicator: pd.Series, lookback: int = 30) -> bool:
    if len(close) < lookback: return False
    c = close.tail(lookback)
    ind = indicator.tail(lookback)
    try:
        min_idx = c.values.argmin()
        if min_idx < 5: return False
        
        prev_c = c.iloc[:min_idx-2]
        if len(prev_c) < 3: return False
        prev_min_idx = prev_c.values.argmin()
        
        p1, p2 = prev_c.iloc[prev_min_idx], c.iloc[min_idx]
        i1, i2 = ind.iloc[prev_min_idx], ind.iloc[min_idx]
        
        if p2 < p1 and i2 > i1:
            return True
    except Exception:
        pass
    return False

def adx_indicator(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up = high - high.shift(1)
    down = low.shift(1) - low
    
    # DÜZELTME: pandas indeksleri (tarihler) eşleştirildi
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    
    tr = true_range(high, low, close)
    
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    pdm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    mdm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    
    pdi = 100 * (pdm_smooth / tr_smooth.replace(0, np.nan))
    mdi = 100 * (mdm_smooth / tr_smooth.replace(0, np.nan))
    
    dx = 100 * (abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx.fillna(0), pdi.fillna(0), mdi.fillna(0)


# =============================
# KANGAROO TAIL (KANGURU KUYRUĞU)
# =============================
def add_kangaroo_tails(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["KANGAROO_BULL"] = 0
    df["KANGAROO_BEAR"] = 0
    
    body = (df["Close"] - df["Open"]).abs()
    trange = df["High"] - df["Low"]
    lower_wick = df[["Open", "Close"]].min(axis=1) - df["Low"]
    upper_wick = df["High"] - df[["Open", "Close"]].max(axis=1)
    
    rolling_min = df["Low"].rolling(window=lookback).min()
    rolling_max = df["High"].rolling(window=lookback).max()
    
    atr_approx = trange.rolling(10).mean()
    valid_trange = trange > 0
    
    # Bullish: Low is lowest of N days, body is small, lower wick is huge
    bull_cond = valid_trange & (df["Low"] == rolling_min) & ((body / trange) <= 0.3) & ((lower_wick / trange) >= 0.6) & (trange >= atr_approx * 0.8)
    
    # Bearish: High is highest of N days, body is small, upper wick is huge
    bear_cond = valid_trange & (df["High"] == rolling_max) & ((body / trange) <= 0.3) & ((upper_wick / trange) >= 0.6) & (trange >= atr_approx * 0.8)
    
    df.loc[bull_cond, "KANGAROO_BULL"] = 1
    df.loc[bear_cond, "KANGAROO_BEAR"] = 1
    return df


# =============================
# Overbought / Speculation Indicators
# =============================
def add_overbought_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["RSI_OVERBOUGHT"] = (df["RSI"] > 70).astype(int)
    df["RSI_OVERSOLD"] = (df["RSI"] < 30).astype(int)

    bb_den = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
    df["BB_PERCENT_B"] = ((df["Close"] - df["BB_lower"]) / bb_den).replace([np.inf, -np.inf], np.nan)
    df["BB_OVERBOUGHT"] = (df["Close"] > df["BB_upper"]).astype(int)
    df["BB_OVERSOLD"] = (df["Close"] < df["BB_lower"]).astype(int)

    df["VOLUME_SMA20"] = df["Volume"].rolling(20).mean()
    df["VOLUME_SPIKE"] = (df["Volume"] > df["VOLUME_SMA20"] * 1.5).astype(int)

    df["PRICE_TO_EMA50"] = (df["Close"] / df["EMA50"] - 1) * 100
    df["PRICE_TO_EMA200"] = (df["Close"] / df["EMA200"] - 1) * 100
    df["PRICE_EXTREME"] = ((df["PRICE_TO_EMA50"] > 20) | (df["PRICE_TO_EMA200"] > 30)).astype(int)

    def stoch_rsi(series, period=14, smooth_k=3, smooth_d=3):
        rsi_vals = series
        min_rsi = rsi_vals.rolling(period).min()
        max_rsi = rsi_vals.rolling(period).max()
        den = (max_rsi - min_rsi).replace(0, np.nan)
        stoch = 100 * (rsi_vals - min_rsi) / den
        stoch = stoch.replace([np.inf, -np.inf], np.nan).fillna(50)
        k = stoch.rolling(smooth_k).mean()
        d = k.rolling(smooth_d).mean()
        return k, d

    df["STOCH_RSI_K"], df["STOCH_RSI_D"] = stoch_rsi(df["RSI"])
    df["STOCH_OVERBOUGHT"] = (df["STOCH_RSI_K"] > 80).astype(int)

    df["VOLUME_DIR"] = np.sign(df["Volume"].diff()).fillna(0)
    df["PRICE_DIR"] = np.sign(df["Close"].diff()).fillna(0)
    df["WEAK_UPTREND"] = ((df["PRICE_DIR"] > 0) & (df["VOLUME_DIR"] < 0)).astype(int)

    return df


def detect_speculation(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    result = {
        "overbought_score": 0,
        "oversold_score": 0,
        "speculation_score": 0,
        "details": {},
    }

    if last["RSI"] > 70:
        result["overbought_score"] += 40
        result["details"]["rsi"] = f"Aşırı alım (RSI: {last['RSI']:.1f})"
    elif last["RSI"] < 30:
        result["oversold_score"] += 50
        result["details"]["rsi"] = f"Aşırı satım (RSI: {last['RSI']:.1f})"

    if bool(last["BB_OVERBOUGHT"]):
        result["overbought_score"] += 20
        result["details"]["bb"] = "Fiyat Bollinger üst bandında"
    elif bool(last["BB_OVERSOLD"]):
        result["oversold_score"] += 50
        result["details"]["bb"] = "Fiyat Bollinger alt bandında"

    if bool(last["STOCH_OVERBOUGHT"]):
        result["overbought_score"] += 20
        result["details"]["stoch"] = "Stokastik RSI aşırı alımda"

    if bool(last["VOLUME_SPIKE"]):
        result["speculation_score"] += 60
        result["details"]["volume"] = "Ani hacim artışı (spekülasyon)"

    if bool(last["PRICE_EXTREME"]):
        result["overbought_score"] += 20
        result["details"]["price_extreme"] = f"Fiyat EMA'dan çok uzak (EMA50: %{last['PRICE_TO_EMA50']:.1f})"

    if bool(last["WEAK_UPTREND"]):
        result["speculation_score"] += 40
        result["details"]["weak_trend"] = "Fiyat yükselirken hacim düşüyor (zayıflama)"

    result["overbought_score"] = min(100, result["overbought_score"])
    result["oversold_score"] = min(100, result["oversold_score"])
    result["speculation_score"] = min(100, result["speculation_score"])

    if result["overbought_score"] >= 60:
        result["verdict"] = "AŞIRI DEĞERLİ (SAT bölgesi)"
    elif result["oversold_score"] >= 60:
        result["verdict"] = "AŞIRI DEĞERSİZ (AL bölgesi)"
    elif result["speculation_score"] >= 60:
        result["verdict"] = "SPEKÜLATİF HAREKET (dikkatli olunmalı)"
    else:
        result["verdict"] = "NÖTR (normal değer aralığı)"

    return result


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

    bb_mid_safe = df["BB_mid"].replace(0, np.nan)
    df["BB_WIDTH"] = ((df["BB_upper"] - df["BB_lower"]) / bb_mid_safe).replace([np.inf, -np.inf], np.nan)

    vol_sma_safe = df["VOL_SMA"].replace(0, np.nan)
    df["VOL_RATIO"] = (df["Volume"] / vol_sma_safe).replace([np.inf, -np.inf], np.nan)

    df = add_overbought_indicators(df)
    df = add_kangaroo_tails(df)
    return df


# =============================
# Market regime filters
# =============================
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_spy_regime_ok() -> bool:
    spy = yf.download("SPY", period="2y", interval="1d", auto_adjust=True, progress=False)
    spy = _flatten_yf(spy)
    if spy.empty or len(spy) < 260:
        return True
    spy["EMA200"] = ema(spy["Close"], 200)
    last = spy.iloc[-1]
    return bool(last["Close"] > last["EMA200"])


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_bist_regime_ok() -> bool:
    xu100 = yf.download("XU100.IS", period="2y", interval="1d", auto_adjust=True, progress=False)
    xu100 = _flatten_yf(xu100)
    if xu100.empty or len(xu100) < 200:
        return True
    xu100["EMA200"] = ema(xu100["Close"], 200)
    last = xu100.iloc[-1]
    return bool(last["Close"] > last["EMA200"])


# =============================
# Higher timeframe trend filter
# =============================
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_higher_tf_trend(ticker: str, higher_tf_interval: str = "1wk", ema_period: int = 200) -> bool:
    try:
        df = yf.download(ticker, period="5y", interval=higher_tf_interval, auto_adjust=True, progress=False)
        df = _flatten_yf(df)
        if df.empty or len(df) < min(ema_period, 100):
            return True
        df["EMA"] = ema(df["Close"], ema_period)
        last = df.iloc[-1]
        return bool(last["Close"] > last["EMA"])
    except Exception:
        return True


# =============================
# Strategy: scoring + checkpoints
# =============================
def signal_with_checkpoints(
    df: pd.DataFrame,
    cfg: dict,
    market_filter_ok: bool = True,
    higher_tf_filter_ok: bool = True,
):
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
    entry = trend_ok & vol_ok & liq_ok & entry_triggers & market_filter_ok & higher_tf_filter_ok

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
        "Higher TF Filter OK": bool(higher_tf_filter_ok),
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
# Backtest (long-only) + advanced exits
# =============================
def backtest_long_only(
    df: pd.DataFrame,
    cfg: dict,
    risk_free_annual: float,
    benchmark_returns: Optional[pd.Series] = None,
):
    df = df.copy()
    entry_sig = df["ENTRY"].shift(1).fillna(0).astype(int)
    exit_sig = df["EXIT"].shift(1).fillna(0).astype(int)

    cash = float(cfg["initial_capital"])
    shares = 0.0
    stop = np.nan
    entry_price = np.nan
    target_price = np.nan
    bars_held = 0
    half_sold = False

    trades = []
    equity_curve = []

    commission = cfg["commission_bps"] / 10000.0
    slippage = cfg["slippage_bps"] / 10000.0
    time_stop_bars = cfg.get("time_stop_bars", 10)
    tp_mult = cfg.get("take_profit_mult", 2.0)

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        price = float(row["Close"])

        if shares > 0 and pd.notna(row["ATR"]) and row["ATR"] > 0:
            new_stop = price - cfg["atr_stop_mult"] * float(row["ATR"])
            stop = max(stop, new_stop) if pd.notna(stop) else new_stop

        position_value = shares * price * (1 - slippage)
        equity = cash + position_value

        if shares > 0:
            bars_held += 1
            stop_hit = pd.notna(stop) and (price <= stop)
            target_hit = (not half_sold) and pd.notna(target_price) and (price >= target_price)
            time_stop_hit = (bars_held >= time_stop_bars) and (price < entry_price)

            if target_hit:
                sell_shares = shares * 0.5
                sell_price = price * (1 - slippage)
                gross = sell_shares * sell_price
                fee = gross * commission
                cash += (gross - fee)
                shares -= sell_shares
                half_sold = True
                stop = max(stop, entry_price)

                if len(trades) > 0:
                    trades[-1]["pnl"] = cash + (shares * price * (1 - slippage)) - trades[-1]["equity_before"]

            if exit_sig.iloc[i] == 1 or stop_hit or time_stop_hit:
                sell_price = price * (1 - slippage)
                gross = shares * sell_price
                fee = gross * commission
                cash += (gross - fee)

                trades[-1]["exit_date"] = date
                trades[-1]["exit_price"] = sell_price

                if stop_hit:
                    reason = "STOP"
                elif time_stop_hit:
                    reason = "TIME_STOP"
                else:
                    reason = "RULE_EXIT"

                trades[-1]["exit_reason"] = reason
                trades[-1]["pnl"] = cash - trades[-1]["equity_before"]

                shares = 0.0
                stop = np.nan
                entry_price = np.nan
                target_price = np.nan
                bars_held = 0
                half_sold = False

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

    if benchmark_returns is not None:
        common_dates = ret.index.intersection(benchmark_returns.index)
        if len(common_dates) > 5:
            r_aligned = ret.loc[common_dates]
            b_aligned = benchmark_returns.loc[common_dates]
            cov = np.cov(r_aligned, b_aligned)[0, 1]
            var_b = np.var(b_aligned)
            beta = cov / var_b if var_b != 0 else 1.0
            mean_r = r_aligned.mean() * 252
            mean_b = b_aligned.mean() * 252
            alpha = (mean_r - risk_free_annual) - beta * (mean_b - risk_free_annual)
            diff = r_aligned - b_aligned
            info_ratio = (diff.mean() * 252) / (diff.std() * np.sqrt(252)) if diff.std() > 0 else 0.0
        else:
            beta = 1.0
            alpha = 0.0
            info_ratio = 0.0
    else:
        beta = 1.0
        alpha = 0.0
        info_ratio = 0.0

    peak = eq.cummax()
    drawdown_pct = (eq - peak) / peak
    ulcer_index = np.sqrt((drawdown_pct**2).mean()) if len(drawdown_pct) > 0 else 0.0

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

    if not tdf.empty and len(tdf) > 5:
        win_rate = (tdf["pnl"] > 0).mean()
        avg_win = tdf.loc[tdf["pnl"] > 0, "pnl"].mean() if win_rate > 0 else 0
        avg_loss = -tdf.loc[tdf["pnl"] < 0, "pnl"].mean() if win_rate < 1 else 0
        if avg_loss > 0 and win_rate > 0 and win_rate < 1:
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            kelly = (p * b - q) / b
            kelly = max(0, min(kelly, 0.10))
        else:
            kelly = 0.0
    else:
        kelly = 0.0

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
        "Beta": float(beta),
        "Alpha": float(alpha),
        "Information Ratio": float(info_ratio),
        "Ulcer Index": float(ulcer_index),
        "Kelly % (öneri)": float(kelly * 100),
    }
    return eq, tdf, metrics


# =============================
# Fundamentals
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

    FUND_KEYS = [
        "marketCap", "trailingPE", "forwardPE", "pegRatio",
        "priceToSalesTrailing12Months", "priceToBook", "returnOnEquity",
        "profitMargins", "operatingMargins", "debtToEquity",
        "revenueGrowth", "earningsGrowth", "freeCashflow", "currentPrice"
    ]
    
    out = {k: safe_float(info.get(k)) for k in FUND_KEYS}

    out["ticker"] = ticker
    out["market"] = market
    out["sector"] = info.get("sector", "")
    out["industry"] = info.get("industry", "")
    out["longName"] = info.get("longName", "") or info.get("shortName", "")
    
    out["debtToEquity"] = _fix_debt_to_equity(out["debtToEquity"])
    return out


def fundamental_score_row(row: dict, mode: str, thresholds: dict) -> Tuple[float, dict, bool]:
    b = {}

    def ok(name, cond, weight, available: bool):
        b[name] = {
            "ok": bool(cond) if available else False,
            "weight": weight,
            "available": bool(available),
        }
        return (weight if (available and cond) else 0.0), (weight if available else 0.0), (1 if available else 0)

    score = 0.0
    total_w = 0.0
    avail_cnt = 0
    ok_cnt = 0

    def A(x):
        return pd.notna(x)

    if mode == "Quality":
        s, tw, ac = ok("ROE", A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"], 20, A(row["returnOnEquity"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"]) else 0)

        s, tw, ac = ok("Op Margin", A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"], 15, A(row["operatingMargins"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"]) else 0)

        s, tw, ac = ok("Debt/Equity", A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"], 20, A(row["debtToEquity"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"]) else 0)

        s, tw, ac = ok("Profit Margin", A(row["profitMargins"]) and row["profitMargins"] >= thresholds["profit_margin"], 15, A(row["profitMargins"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["profitMargins"]) and row["profitMargins"] >= thresholds["profit_margin"]) else 0)

        s, tw, ac = ok("FCF", A(row["freeCashflow"]) and row["freeCashflow"] > 0, 30, A(row["freeCashflow"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["freeCashflow"]) and row["freeCashflow"] > 0) else 0)

    elif mode == "Value":
        s, tw, ac = ok("Forward P/E", A(row["forwardPE"]) and row["forwardPE"] <= thresholds["fpe"], 30, A(row["forwardPE"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["forwardPE"]) and row["forwardPE"] <= thresholds["fpe"]) else 0)

        s, tw, ac = ok("PEG", A(row["pegRatio"]) and row["pegRatio"] <= thresholds["peg"], 20, A(row["pegRatio"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["pegRatio"]) and row["pegRatio"] <= thresholds["peg"]) else 0)

        s, tw, ac = ok(
            "P/S",
            A(row["priceToSalesTrailing12Months"]) and row["priceToSalesTrailing12Months"] <= thresholds["ps"],
            20,
            A(row["priceToSalesTrailing12Months"]),
        )
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["priceToSalesTrailing12Months"]) and row["priceToSalesTrailing12Months"] <= thresholds["ps"]) else 0)

        s, tw, ac = ok("P/B", A(row["priceToBook"]) and row["priceToBook"] <= thresholds["pb"], 15, A(row["priceToBook"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["priceToBook"]) and row["priceToBook"] <= thresholds["pb"]) else 0)

        s, tw, ac = ok("ROE", A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"], 15, A(row["returnOnEquity"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"]) else 0)

    else:  # Growth
        s, tw, ac = ok("Revenue Growth", A(row["revenueGrowth"]) and row["revenueGrowth"] >= thresholds["rev_g"], 35, A(row["revenueGrowth"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["revenueGrowth"]) and row["revenueGrowth"] >= thresholds["rev_g"]) else 0)

        s, tw, ac = ok("Earnings Growth", A(row["earningsGrowth"]) and row["earningsGrowth"] >= thresholds["earn_g"], 35, A(row["earningsGrowth"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["earningsGrowth"]) and row["earningsGrowth"] >= thresholds["earn_g"]) else 0)

        s, tw, ac = ok("Op Margin", A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"], 15, A(row["operatingMargins"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"]) else 0)

        s, tw, ac = ok("Debt/Equity", A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"], 15, A(row["debtToEquity"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"]) else 0)

    score_pct = (score / total_w) * 100 if total_w > 0 else 0.0
    min_coverage = int(thresholds.get("min_coverage", 3))
    min_ok = int(thresholds["min_ok"])
    pass_bool = (score_pct >= thresholds["min_score"]) and (ok_cnt >= min_ok) and (avail_cnt >= min_coverage)
    return float(score_pct), b, bool(pass_bool)


# =============================
# Target price band / SR Levels (YENİ GÜÇ-HACİM-UZUNLUK EKLENTİSİ)
# =============================
def _swing_points(high: pd.Series, low: pd.Series, left: int = 2, right: int = 2):
    hs = []
    ls = []
    n = len(high)
    for i in range(left, n - right):
        hwin = high.iloc[i - left : i + right + 1]
        lwin = low.iloc[i - left : i + right + 1]
        if high.iloc[i] == hwin.max():
            hs.append((high.index[i], float(high.iloc[i])))
        if low.iloc[i] == lwin.min():
            ls.append((low.index[i], float(low.iloc[i])))
    return hs, ls

def analyze_sr_levels(df: pd.DataFrame, lookback: int = 120, tol=0.015) -> List[dict]:
    """Destek/Direnç seviyelerinin güç, hacim ve uzunluğunu analiz eder."""
    h = df["High"].tail(lookback).dropna()
    l = df["Low"].tail(lookback).dropna()
    c = df["Close"].tail(lookback).dropna()
    if len(c) < 10:
        return []
        
    v = df["Volume"].tail(lookback) if "Volume" in df.columns else pd.Series(dtype=float)

    hs, ls = _swing_points(h, l, left=3, right=3)
    raw_levels = [val for _, val in hs] + [val for _, val in ls]
    raw_levels += [float(c.tail(20).max()), float(c.tail(20).min())]
    raw_levels = sorted(list(set([round(float(x), 2) for x in raw_levels if np.isfinite(x)])))

    if not raw_levels:
        return []

    # Yakın seviyeleri grupla (Cluster)
    clusters = []
    for rl in raw_levels:
        placed = False
        for cl in clusters:
            if abs(rl - cl['center']) / cl['center'] <= tol:
                cl['points'].append(rl)
                cl['center'] = sum(cl['points'])/len(cl['points'])
                placed = True
                break
        if not placed:
            clusters.append({'center': rl, 'points': [rl]})

    avg_vol_normal = float(v.mean()) if not v.empty else 1.0
    if avg_vol_normal <= 0: avg_vol_normal = 1.0

    details = []
    df_lookback = df.tail(lookback)
    
    for cl in clusters:
        level_px = cl['center']
        lower_bound = level_px * (1 - tol/2)
        upper_bound = level_px * (1 + tol/2)

        touches = df_lookback[(df_lookback["High"] >= lower_bound) & (df_lookback["Low"] <= upper_bound)]

        num_touches = len(touches)
        if num_touches == 0:
            continue

        # 1. Uzunluk (Bar sayısı)
        first_touch_idx = touches.index[0]
        first_idx_num = df_lookback.index.get_loc(first_touch_idx)
        duration_bars = len(df_lookback) - first_idx_num

        # 2. Hacim Oranı (%)
        if "Volume" in df_lookback.columns and not touches.empty:
            vol_at_level = float(touches["Volume"].mean())
        else:
            vol_at_level = avg_vol_normal

        vol_diff_pct = (vol_at_level / avg_vol_normal - 1.0) * 100.0

        # 3. Güç Skoru Hesaplama (Temas Sayısı + Hacim Etkisi + Yaş)
        score_touches = min(num_touches * 10, 40) # Maksimum 40 puan (4+ temas)
        score_vol = min(max(vol_diff_pct / 2.0, 0), 35) # Maksimum 35 puan (+70% hacimde tam puan)
        score_dur = min(duration_bars / 2.0, 25) # Maksimum 25 puan (50 bar eski ise tam puan)

        strength_pct = min(score_touches + score_vol + score_dur, 99.0)

        details.append({
            "price": round(level_px, 2),
            "duration_bars": int(duration_bars),
            "vol_at_level": float(vol_at_level),
            "vol_diff_pct": float(vol_diff_pct),
            "strength_pct": float(strength_pct),
            "touches": int(num_touches)
        })

    return sorted(details, key=lambda x: x["price"])


def target_price_band(df: pd.DataFrame):
    last = df.iloc[-1]
    px_close = float(last["Close"])
    atrv = float(last["ATR"]) if pd.notna(last.get("ATR", np.nan)) else np.nan

    # Detaylı S/R hesaplamasını al (Güç, Hacim, Uzunluk vb.)
    lv_details = analyze_sr_levels(df)

    if not np.isfinite(atrv) or atrv <= 0:
        return {"base": px_close, "bull": None, "bear": None, "levels": lv_details, "r1_dict": None, "s1_dict": None}

    bull1 = px_close + 1.5 * atrv
    bull2 = px_close + 3.0 * atrv
    bear1 = px_close - 1.5 * atrv
    bear2 = px_close - 3.0 * atrv

    above = [x for x in lv_details if x["price"] >= px_close]
    below = [x for x in lv_details if x["price"] <= px_close]
    
    r1_dict = min(above, key=lambda x: x["price"]) if above else None
    s1_dict = max(below, key=lambda x: x["price"]) if below else None

    r1 = r1_dict["price"] if r1_dict else None
    s1 = s1_dict["price"] if s1_dict else None

    return {
        "base": px_close,
        "bull": (bull1, bull2, r1),
        "bear": (bear1, bear2, s1),
        "levels": lv_details,
        "r1_dict": r1_dict,
        "s1_dict": s1_dict
    }


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
# Gemini helpers
# =============================
def _get_secret(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name, "")
        if v is None:
            return default
        return str(v).strip()
    except Exception:
        return default


def _http_post_json(url: str, payload: dict, headers: dict = None, timeout: int = 60) -> dict:
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"error": {"message": f"Non-JSON response (status={r.status_code})", "text": r.text[:500]}}
    if r.status_code >= 400:
        if "error" not in data:
            data["error"] = {"message": f"HTTP {r.status_code}", "text": str(data)[:500]}
    return data


def _extract_gemini_text(resp: dict) -> str:
    if not isinstance(resp, dict):
        return str(resp)
    if resp.get("error"):
        return f"Gemini API error: {resp['error'].get('message','')}"
    cands = resp.get("candidates") or []
    if not cands:
        return "Gemini: boş cevap döndü (candidates yok)."
    parts = (cands[0].get("content") or {}).get("parts") or []
    if not parts:
        return "Gemini: boş cevap döndü (parts yok)."
    texts = []
    for p in parts:
        if isinstance(p, dict) and "text" in p:
            texts.append(p["text"])
    return "\n".join(texts).strip() if texts else "Gemini: metin üretmedi."


def gemini_generate_text(
    *,
    prompt: str,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
    image_bytes: Optional[bytes] = None,
) -> str:
    api_key = _get_secret("GEMINI_API_KEY", "")
    if not api_key:
        return "GEMINI_API_KEY bulunamadı. Streamlit Cloud > Settings > Secrets içine GEMINI_API_KEY=... ekleyin."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key}

    parts = [{"text": prompt}]
    if image_bytes:
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        parts.append(
            {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": b64_img,
                }
            }
        )

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }
    resp = _http_post_json(url, payload, headers=headers, timeout=90)
    return _extract_gemini_text(resp)


# =============================
# Sentiment Analysis via Google News RSS + Gemini
# =============================
@st.cache_data(ttl=30 * 60, show_spinner=False)
def get_news_sentiment(
    ticker: str,
    company_name: str = "",
    gemini_model: str = "gemini-1.5-flash",
    gemini_temp: float = 0.2,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    
    try:
        if company_name and company_name != "":
            query = f"{company_name} stock"
        else:
            query = f"{ticker} stock"

        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return {"error": f"Haberler çekilemedi (HTTP {resp.status_code})", "sentiment": None, "summary": ""}
            
        root = ET.fromstring(resp.content)
        
        # SADECE BAŞLIK DEĞİL, LİNKLERİ DE ÇEKİYORUZ
        news_items = []
        for item in root.findall(".//item")[:10]:
            title_node = item.find("title")
            link_node = item.find("link")
            if title_node is not None and title_node.text:
                t = title_node.text
                l = link_node.text if (link_node is not None and link_node.text) else ""
                news_items.append({"title": t, "link": l})

        if not news_items:
            return {"error": "Haber bulunamadı", "sentiment": None, "summary": ""}

        prompt_titles = [item["title"] for item in news_items]
        prompt = f"""Aşağıdaki haber başlıklarının duygu analizini yap (pozitif, negatif, nötr).
Sonuçları şu formatta ver:
Pozitif: [sayı]
Negatif: [sayı]
Nötr: [sayı]
- Bileşik skor: (pozitif - negatif) / toplam (örneğin 0.35)
- Kısa bir özet (2 cümle)

Haber Başlıkları:
{chr(10).join([f"- {t}" for t in prompt_titles])}
"""

        response = gemini_generate_text(
            prompt=prompt,
            model=gemini_model,
            temperature=gemini_temp,
            max_output_tokens=max_tokens,
            image_bytes=None,
        )

        pos_match = re.search(r"Pozitif:?\s*(\d+)", response, re.IGNORECASE)
        neg_match = re.search(r"Negatif:?\s*(\d+)", response, re.IGNORECASE)
        neu_match = re.search(r"Nötr:?\s*(\d+)", response, re.IGNORECASE)

        pos = int(pos_match.group(1)) if pos_match else 0
        neg = int(neg_match.group(1)) if neg_match else 0
        neu = int(neu_match.group(1)) if neu_match else 0
        total = pos + neg + neu
        compound = (pos - neg) / total if total > 0 else 0

        return {
            "error": None,
            "sentiment": compound,
            "summary": response,
            "pos": pos / total if total > 0 else 0,
            "neg": neg / total if total > 0 else 0,
            "neu": neu / total if total > 0 else 0,
            "news_items": news_items[:5], # SADECE İLK 5 HABERİ LİNK OLARAK DÖNDÜR
        }
    except Exception as e:
        return {"error": str(e), "sentiment": None, "summary": ""}


# =============================
# Price Action
# =============================
def price_action_pack(df: pd.DataFrame, last_n: int = 20) -> dict:
    use = df.tail(last_n).copy()
    if use.empty or len(use) < 10:
        return {"note": "insufficient_bars", "last_n": int(len(use))}

    o = use["Open"].astype(float)
    h = use["High"].astype(float)
    l = use["Low"].astype(float)
    c = use["Close"].astype(float)

    swing_highs, swing_lows = _swing_points(h, l, left=2, right=2)

    q20 = float(np.quantile(c.values, 0.20))
    q50 = float(np.quantile(c.values, 0.50))
    q80 = float(np.quantile(c.values, 0.80))

    recent_highs = [v for _, v in swing_highs[-5:]] if swing_highs else []
    recent_lows = [v for _, v in swing_lows[-5:]] if swing_lows else []
    res = max(recent_highs) if recent_highs else float(h.max())
    sup = min(recent_lows) if recent_lows else float(l.min())

    last_close = float(c.iloc[-1])
    prev_close = float(c.iloc[-2]) if len(c) >= 2 else last_close
    last_high = float(h.iloc[-1])
    last_low = float(l.iloc[-1])

    bull_break = (last_close > res) and (prev_close <= res)
    bear_break = (last_close < sup) and (prev_close >= sup)

    vol_ok = None
    if "Volume" in use.columns:
        vol = use["Volume"].astype(float)
        vol_sma = float(vol.rolling(10).mean().iloc[-1]) if len(vol) >= 10 else float(vol.mean())
        vol_ok = float(vol.iloc[-1]) > vol_sma if np.isfinite(vol_sma) else None

    impulse_up = (c.diff().tail(3) > 0).all() and (last_close >= q80)
    impulse_dn = (c.diff().tail(3) < 0).all() and (last_close <= q20)

    ob = None
    if impulse_up:
        for i in range(len(use) - 4, -1, -1):
            if c.iloc[i] < o.iloc[i]:
                ob = {
                    "type": "bullish_order_block_proxy",
                    "index": str(use.index[i]),
                    "open": float(o.iloc[i]),
                    "high": float(h.iloc[i]),
                    "low": float(l.iloc[i]),
                    "close": float(c.iloc[i]),
                }
                break
    elif impulse_dn:
        for i in range(len(use) - 4, -1, -1):
            if c.iloc[i] > o.iloc[i]:
                ob = {
                    "type": "bearish_order_block_proxy",
                    "index": str(use.index[i]),
                    "open": float(o.iloc[i]),
                    "high": float(h.iloc[i]),
                    "low": float(l.iloc[i]),
                    "close": float(c.iloc[i]),
                }
                break

    pack = {
        "last_n": int(len(use)),
        "q20": q20,
        "q50": q50,
        "q80": q80,
        "support": sup,
        "resistance": res,
        "bull_breakout": bool(bull_break),
        "bear_breakout": bool(bear_break),
        "vol_confirm": (None if vol_ok is None else bool(vol_ok)),
        "last_bar": {
            "t": str(use.index[-1]),
            "open": float(o.iloc[-1]),
            "high": last_high,
            "low": last_low,
            "close": last_close,
        },
        "swing_highs": [{"t": str(t), "p": float(p)} for t, p in swing_highs[-6:]],
        "swing_lows": [{"t": str(t), "p": float(p)} for t, p in swing_lows[-6:]],
        "order_block_proxy": ob,
    }
    return pack


def df_snapshot_for_llm(df: pd.DataFrame, n: int = 25) -> dict:
    use_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "EMA50",
        "EMA200",
        "RSI",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        "BB_mid",
        "BB_upper",
        "BB_lower",
        "ATR",
        "ATR_PCT",
        "VOL_SMA",
        "VOL_RATIO",
        "BB_WIDTH",
        "SCORE",
        "ENTRY",
        "EXIT",
        "RSI_OVERBOUGHT",
        "BB_OVERBOUGHT",
        "VOLUME_SPIKE",
        "PRICE_EXTREME",
        "STOCH_OVERBOUGHT",
        "WEAK_UPTREND",
        "KANGAROO_BULL",
        "KANGAROO_BEAR"
    ]
    cols = [c for c in use_cols if c in df.columns]
    tail = df[cols].tail(n).copy()
    tail.index = tail.index.astype(str)
    
    summary = {}
    if not df.empty:
        summary["rsi_last"] = float(df["RSI"].iloc[-1]) if "RSI" in df else None
        summary["rsi_5d_avg"] = float(df["RSI"].tail(5).mean()) if "RSI" in df else None
        if "EMA50" in df and "EMA200" in df:
            summary["trend"] = "up" if df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1] else "down"

    return {
        "cols": cols,
        "n": int(len(tail)),
        "last_index": str(tail.index[-1]) if len(tail) else None,
        "rows": tail.to_dict(orient="records"),
        "summary": summary
    }


# =============================
# Presets
# =============================
PRESETS = {
    "Defansif": {
        "rsi_entry_level": 52,
        "rsi_exit_level": 46,
        "atr_pct_max": 0.06,
        "atr_stop_mult": 2.0,
        "time_stop_bars": 15,
        "take_profit_mult": 2.5,
    },
    "Dengeli": {
        "rsi_entry_level": 50,
        "rsi_exit_level": 45,
        "atr_pct_max": 0.08,
        "atr_stop_mult": 1.5,
        "time_stop_bars": 10,
        "take_profit_mult": 2.0,
    },
    "Agresif": {
        "rsi_entry_level": 48,
        "rsi_exit_level": 43,
        "atr_pct_max": 0.10,
        "atr_stop_mult": 1.2,
        "time_stop_bars": 7,
        "take_profit_mult": 1.5,
    },
}


# =============================
# Screener row finder
# =============================
def find_screener_row(sdf: pd.DataFrame, ticker: str) -> Optional[Dict[str, Any]]:
    if sdf is None or sdf.empty or "ticker" not in sdf.columns:
        return None

    t = (ticker or "").upper().strip()
    t_naked = naked_ticker(t)

    tmp = sdf.copy()
    tmp["_tk"] = tmp["ticker"].astype(str).str.upper().str.strip()
    tmp["_tk_naked"] = tmp["_tk"].str.replace(".IS", "", regex=False)

    m = tmp[(tmp["_tk"] == t) | (tmp["_tk"] == f"{t_naked}.IS") | (tmp["_tk_naked"] == t_naked)]
    if m.empty:
        return None

    row = m.iloc[0].drop(labels=["_tk", "_tk_naked"], errors="ignore").to_dict()
    return row


def merge_fa_row(
    screener_row: Optional[Dict[str, Any]],
    fundamentals: Optional[Dict[str, Any]],
    fa_eval: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
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
# REPORT EXPORT
# =============================
def build_html_report(
    title: str,
    meta: dict,
    checkpoints: dict,
    metrics: dict,
    tp: dict,
    rr_info: dict,
    figs: Dict[str, go.Figure],
    fa_row: Optional[Dict[str, Any]] = None,
    gemini_insight: Optional[str] = None,
    pa_pack: Optional[Dict[str, Any]] = None,
    sentiment_summary: Optional[str] = None,
    sentiment_items: Optional[List[dict]] = None,
    overbought_result: Optional[Dict[str, Any]] = None,
) -> bytes:
    def esc(x):
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

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
    
    levels_txt = "<br>".join([
        f"{x['price']:.2f} (Güç: %{x['strength_pct']:.0f}, Uzunluk: {x['duration_bars']} Bar, Hacim: %{x['vol_diff_pct']:+.1f})"
        for x in levels[:120]
    ]) if levels else "N/A"

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

    overbought_html = ""
    if overbought_result:
        ob_details = "<ul>"
        for _, v in overbought_result.get("details", {}).items():
            ob_details += f"<li>{esc(v)}</li>"
        ob_details += "</ul>"
        overbought_html = f"""
        <div class="card" style="margin-top:16px;">
            <h2>📊 Aşırı Alım / Spekülasyon Analizi</h2>
            <div><b>Karar:</b> {esc(overbought_result['verdict'])}</div>
            <div><b>Aşırı Alım Skoru:</b> {overbought_result['overbought_score']}/100</div>
            <div><b>Aşırı Satım Skoru:</b> {overbought_result['oversold_score']}/100</div>
            <div><b>Spekülasyon Skoru:</b> {overbought_result['speculation_score']}/100</div>
            <div><b>Detaylar:</b> {ob_details}</div>
        </div>
        """

    gemini_block = ""
    if gemini_insight:
        gemini_block = f"""
        <div class="card" style="margin-top:16px;">
            <h2>Gemini — Chart & Price Action Insight</h2>
            <pre style="white-space:pre-wrap; font-family:inherit;">{esc(gemini_insight)}</pre>
        </div>
        """

    pa_block = ""
    if pa_pack:
        pa_block = f"""
        <div class="card" style="margin-top:16px;">
            <h2>Price Action Pack (Last {esc(pa_pack.get('last_n',''))} Bars)</h2>
            <pre style="white-space:pre-wrap; font-family:monospace; font-size:12px;">{esc(json.dumps(pa_pack, ensure_ascii=False, indent=2))}</pre>
        </div>
        """

    sentiment_block = ""
    if sentiment_summary:
        links_html = ""
        if sentiment_items:
            links_html = "<br><br><b>Kaynak Haberler:</b><ul>"
            for item in sentiment_items:
                links_html += f"<li><a href='{esc(item['link'])}' target='_blank'>{esc(item['title'])}</a></li>"
            links_html += "</ul>"

        sentiment_block = f"""
        <div class="card" style="margin-top:16px;">
            <h2>Haber Duygu Analizi (Google News + Gemini)</h2>
            <pre style="white-space:pre-wrap; font-family:inherit;">{esc(sentiment_summary)}</pre>
            {links_html}
        </div>
        """

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
      <div>Beta: {metrics.get('Beta',0):.2f}</div>
      <div>Alpha: {metrics.get('Alpha',0):.2f}</div>
      <div>Info Ratio: {metrics.get('Information Ratio',0):.2f}</div>
      <div>Ulcer Index: {metrics.get('Ulcer Index',0):.4f}</div>
      <div>Kelly Önerisi: {metrics.get('Kelly % (öneri)',0):.1f}%</div>
    </div>
  </div>
  {overbought_html}
  <div class="card" style="margin-top:16px;">
    <h2>Target Band</h2>
    <div>Base: {tp.get('base',0):.2f}</div>
    <div>Bull: {(bull[0] if bull else 0):.2f} → {(bull[1] if bull else 0):.2f} | R1: {(bull[2] if bull else 'N/A')}</div>
    <div>Bear: {(bear[0] if bear else 0):.2f} → {(bear[1] if bear else 0):.2f} | S1: {(bear[2] if bear else 'N/A')}</div>
    <div>RR: {('N/A' if rr_info.get('rr') is None else f"1:{rr_info.get('rr'):.2f}")}</div>
    <div class="muted"><br>Seviyeler ve Güçleri:<br>{levels_txt}</div>
  </div>
  {gemini_block}
  {sentiment_block}
  {pa_block}
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
    levels: Optional[List[dict]],
    trades_df: Optional[pd.DataFrame],
    figs: Optional[Dict[str, go.Figure]],
    include_charts: bool = True,
    gemini_insight: Optional[str] = None,
    pa_pack: Optional[Dict[str, Any]] = None,
    sentiment_summary: Optional[str] = None,
    sentiment_items: Optional[List[dict]] = None,
    overbought_result: Optional[Dict[str, Any]] = None,
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

    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, y, title[:90])
    y -= 18

    c.setFont("Helvetica", 10)
    c.drawString(left, y, subtitle[:140])
    y -= 14

    c.setFont("Helvetica", 9)
    y = _pdf_write_lines(
        c,
        [
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Market: {meta.get('market','')} | Ticker: {meta.get('ticker','')} | Interval: {meta.get('interval','')} | Period: {meta.get('period','')}",
            f"Preset: {meta.get('preset','')} | EMA: {meta.get('ema_fast','')}/{meta.get('ema_slow','')} | RSI: {meta.get('rsi_period','')} | BB: {meta.get('bb_period','')}/{meta.get('bb_std','')} | ATR: {meta.get('atr_period','')} | VolSMA: {meta.get('vol_sma','')}",
        ],
        left,
        y,
        12,
        bottom,
    )
    y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Technical Summary")
    y -= 14

    c.setFont("Helvetica", 9)
    y = _pdf_write_lines(
        c,
        [
            f"Recommendation: {ta_summary.get('rec','')}",
            f"Last Close (bar): {ta_summary.get('close','N/A')} | Live/Last: {ta_summary.get('live','N/A')}",
            f"Score: {ta_summary.get('score','N/A')} | RSI: {ta_summary.get('rsi','N/A')} | EMA50: {ta_summary.get('ema50','N/A')} | EMA200: {ta_summary.get('ema200','N/A')} | ATR%: {ta_summary.get('atr_pct','N/A')}",
        ],
        left,
        y,
        12,
        bottom,
    )
    y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Checkpoints (Last Bar)")
    y -= 14

    c.setFont("Helvetica", 9)
    cp_lines = [f"[{'OK' if v else 'NO'}] {k}" for k, v in checkpoints.items()]
    y = _pdf_write_lines(c, cp_lines, left, y, 11, bottom)
    y -= 6

    if overbought_result:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Aşırı Alım / Spekülasyon")
        y -= 14
        c.setFont("Helvetica", 9)
        ob_lines = [
            f"Karar: {overbought_result['verdict']}",
            f"Aşırı Alım Skoru: {overbought_result['overbought_score']}/100",
            f"Aşırı Satım Skoru: {overbought_result['oversold_score']}/100",
            f"Spekülasyon Skoru: {overbought_result['speculation_score']}/100",
            "Detaylar:",
        ]
        for _, v in overbought_result.get("details", {}).items():
            ob_lines.append(f"  - {v}")
        y = _pdf_write_lines(c, ob_lines, left, y, 11, bottom)
        y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Target Price Band (Scenario)")
    y -= 14
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

    band_lines.append(f"RR (Backtest first target vs stop): {'N/A' if rr is None else f'1:{rr:.2f}'} | Stop(ATR): {fmt_num(stop)}")
    y = _pdf_write_lines(c, band_lines, left, y, 12, bottom)
    y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Levels (Güç, Uzunluk, Hacim)")
    y -= 14

    c.setFont("Helvetica", 9)
    if levels:
        lv_lines = []
        for i in range(0, min(len(levels), 20), 2):
            chunk = levels[i:i+2]
            line = " | ".join([f"{x['price']:.2f} (G:%{x.get('strength_pct',0):.0f}, {x.get('duration_bars',0)}B, V:%{x.get('vol_diff_pct',0):+.1f})" for x in chunk])
            lv_lines.append(line)
    else:
        lv_lines = ["N/A"]
    y = _pdf_write_lines(c, lv_lines, left, y, 11, bottom)
    y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Backtest Summary (Long-only)")
    y -= 14

    c.setFont("Helvetica", 9)
    bm = backtest_metrics or {}
    y = _pdf_write_lines(
        c,
        [
            f"Total Return: {fmt_pct(bm.get('Total Return'))} | Ann Return: {fmt_pct(bm.get('Annualized Return'))} | Ann Vol: {fmt_pct(bm.get('Annualized Volatility'))}",
            f"Sharpe: {fmt_num(bm.get('Sharpe'), 2)} | Sortino: {fmt_num(bm.get('Sortino'), 2)} | Calmar: {fmt_num(bm.get('Calmar'), 2)}",
            f"Max DD: {fmt_pct(bm.get('Max Drawdown'))} | Trades: {bm.get('Trades','')} | Win Rate: {fmt_pct(bm.get('Win Rate'))} | Profit Factor: {fmt_num(bm.get('Profit Factor'), 2)}",
            f"Beta: {fmt_num(bm.get('Beta'),2)} | Alpha: {fmt_num(bm.get('Alpha'),2)} | Info Ratio: {fmt_num(bm.get('Information Ratio'),2)} | Ulcer Index: {fmt_num(bm.get('Ulcer Index'),4)} | Kelly: {fmt_num(bm.get('Kelly % (öneri)'),1)}%",
        ],
        left,
        y,
        12,
        bottom,
    )
    y -= 6

    if sentiment_summary:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Haber Duygu Analizi (Google News)")
        y -= 14
        c.setFont("Helvetica", 9)
        y = _pdf_write_lines(c, sentiment_summary.splitlines(), left, y, 11, bottom)
        y -= 6
        
        if sentiment_items:
            c.setFont("Helvetica-Bold", 10)
            y = _pdf_write_lines(c, ["Kaynak Haberler:"], left, y, 11, bottom)
            c.setFont("Helvetica", 8)
            for item in sentiment_items:
                y = _pdf_write_lines(c, [f"- {item['title'][:110]}"], left, y, 10, bottom)
                c.setFillColorRGB(0, 0, 1)
                y = _pdf_write_lines(c, [f"  {item['link'][:115]}"], left, y, 10, bottom)
                c.setFillColorRGB(0, 0, 0)
            y -= 6

    if pa_pack:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Price Action Pack (Last Bars)")
        y -= 14
        c.setFont("Helvetica", 8)
        pa_txt = json.dumps(pa_pack, ensure_ascii=False, indent=2).splitlines()
        y = _pdf_write_lines(c, pa_txt, left, y, 9, bottom)
        y -= 6

    if gemini_insight:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Gemini Insight")
        y -= 14
        c.setFont("Helvetica", 9)
        gi_lines = (gemini_insight or "").splitlines()
        y = _pdf_write_lines(c, gi_lines, left, y, 11, bottom)
        y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Fundamental Screener Snapshot (Selected Ticker)")
    y -= 14

    c.setFont("Helvetica", 9)
    if fa_row:
        keys = [
            "ticker",
            "longName",
            "FA_pass",
            "FA_score",
            "FA_ok_count",
            "FA_coverage",
            "sector",
            "industry",
            "trailingPE",
            "forwardPE",
            "pegRatio",
            "priceToSalesTrailing12Months",
            "priceToBook",
            "returnOnEquity",
            "operatingMargins",
            "profitMargins",
            "debtToEquity",
            "revenueGrowth",
            "earningsGrowth",
            "marketCap",
        ]
        lines = [f"{k}: {fa_row.get(k)}" for k in keys if k in fa_row]
        if not lines:
            lines = ["(No fields)"]
    else:
        lines = ["Screener satırı bulunamadı (screener çalıştırılmamış olabilir)."]

    y = _pdf_write_lines(c, lines, left, y, 11, bottom)
    y -= 6

    if trades_df is not None and not trades_df.empty:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Trades (first 25 rows)")
        y -= 14
        c.setFont("Helvetica", 8)

        td = trades_df.copy().head(25)
        cols = [cc for cc in ["entry_date", "entry_price", "exit_date", "exit_price", "exit_reason", "pnl", "return_%", "holding_days"] if cc in td.columns]
        header = " | ".join(cols)
        y = _pdf_write_lines(c, [header], left, y, 10, bottom)

        for _, r in td.iterrows():
            row_txt = " | ".join([str(r.get(k, ""))[:18] for k in cols])
            y = _pdf_write_lines(c, [row_txt], left, y, 10, bottom)

        y -= 6

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
            usable_h = (H - 3.2 * cm - 2.0 * cm)
            c.drawImage(img_reader, left, 2.0 * cm, width=usable_w, height=usable_h, preserveAspectRatio=True, anchor="c")

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
# RR helper (DİNAMİK PRICE ACTION GÜNCELLEMESİ)
# =============================
def rr_from_atr_stop(latest_row: pd.Series, tp_dict: dict, cfg: dict):
    close = float(latest_row["Close"])
    atrv = float(latest_row.get("ATR", np.nan)) if pd.notna(latest_row.get("ATR", np.nan)) else np.nan
    
    if not np.isfinite(atrv) or atrv <= 0:
        return {"rr": None, "stop": None, "risk": None, "reward": None}

    # KANGURU KUYRUĞU STOP-LOSS ENTEGRASYONU
    if latest_row.get("KANGAROO_BULL", 0) == 1:
        stop = float(latest_row["Low"]) - (0.1 * atrv) # Kuyruk ucunun çok az altı
    else:
        stop = close - (float(cfg["atr_stop_mult"]) * atrv)
        
    risk = close - stop

    r1 = None
    if tp_dict and tp_dict.get("bull"):
        r1 = tp_dict["bull"][2] 
        
    if r1 is not None and np.isfinite(r1) and r1 > close:
        target = float(r1)
        target_type = "Resistance (R1)"
    else:
        tp_mult = cfg.get("take_profit_mult", 2.0)
        target = close + (tp_mult * cfg["atr_stop_mult"] * atrv)
        target_type = f"ATR-based Target ({tp_mult}x)"

    reward = target - close

    if risk <= 0 or reward <= 0:
        return {"rr": None, "stop": stop, "risk": risk, "reward": reward, "target_type": target_type}

    rr_val = float(reward / risk) if reward is not None else None
    
    return {
        "rr": rr_val, 
        "stop": float(stop), 
        "risk": float(risk), 
        "reward": reward, 
        "target_type": target_type
    }


def fmt_rr(rr):
    if rr is None or (isinstance(rr, float) and (not np.isfinite(rr))):
        return "N/A"
    return f"1:{rr:.2f}"


def pct_dist(level: float, base: float):
    if level is None or not np.isfinite(level) or base == 0:
        return None
    return (level / base - 1.0) * 100.0


# =============================
# Cached data loader (HACK EKLENDİ)
# =============================
@st.cache_data(ttl=300, show_spinner=False)
def load_data_cached(ticker: str, period: str, interval: str, end_date=None, force_latest: bool = False) -> pd.DataFrame:
    if end_date is not None:
        import datetime
        bitis_obj = end_date + datetime.timedelta(days=1)
        bitis_str = bitis_obj.strftime('%Y-%m-%d')
        
        if period == "45d":
            baslangic_obj = end_date - datetime.timedelta(days=45)
        elif period == "3mo":
            baslangic_obj = end_date - datetime.timedelta(days=90)
        elif period == "6mo":
            baslangic_obj = end_date - datetime.timedelta(days=180)
        elif period == "1y":
            baslangic_obj = end_date - datetime.timedelta(days=365)
        elif period == "2y":
            baslangic_obj = end_date - datetime.timedelta(days=730)
        else:
            baslangic_obj = end_date - datetime.timedelta(days=730)
            
        baslangic_str = baslangic_obj.strftime('%Y-%m-%d')
        
        df = yf.download(ticker, start=baslangic_str, end=bitis_str, interval=interval, auto_adjust=True, progress=False)
    else:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        
    df = _flatten_yf(df)

    if force_latest and end_date is None and interval == "1d" and not df.empty:
        try:
            today_data = yf.download(ticker, period="1d", interval="1m", progress=False)
            today_data = _flatten_yf(today_data)
            
            if not today_data.empty:
                today_date = today_data.index[-1].date()
                last_df_date = df.index[-1].date()
                
                if today_date > last_df_date:
                    o = float(today_data["Open"].iloc[0])
                    h = float(today_data["High"].max())
                    l = float(today_data["Low"].min())
                    c = float(today_data["Close"].iloc[-1])
                    v = float(today_data["Volume"].sum())
                    
                    new_idx = pd.to_datetime(str(today_date))
                    if df.index.tz is not None:
                        new_idx = new_idx.tz_localize(df.index.tz)
                        
                    new_row = pd.DataFrame({
                        "Open": [o],
                        "High": [h],
                        "Low": [l],
                        "Close": [c],
                        "Volume": [v]
                    }, index=[new_idx])
                    
                    df = pd.concat([df, new_row])
        except Exception:
            pass
            
    return df


# =============================
# UI STATE
# =============================
st.title("📈 FA→TA Trading Uygulaması + 🤖 AI Analiz")
st.caption("Önce fundamental ile evreni daralt, sonra teknik analizle giriş/çıkış zamanla. Otomatik emir göndermez.")

if "screener_df" not in st.session_state:
    st.session_state.screener_df = pd.DataFrame()
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None
if "ta_ran" not in st.session_state:
    st.session_state.ta_ran = False
if "gemini_text" not in st.session_state:
    st.session_state.gemini_text = ""
if "pa_pack" not in st.session_state:
    st.session_state.pa_pack = {}
if "sentiment_summary" not in st.session_state:
    st.session_state.sentiment_summary = ""
if "sentiment_items" not in st.session_state:
    st.session_state.sentiment_items = []

if "ai_messages" in st.session_state:
    del st.session_state.ai_messages


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("Piyasa")
    market = st.selectbox(
        "Market",
        ["USA", "BIST"],
        index=0,
        help="Analiz edilecek borsayı seçin. USA için ABD hisseleri, BIST için Borsa İstanbul hisseleri.",
    )

    if "last_market" not in st.session_state:
        st.session_state.last_market = market
    elif st.session_state.last_market != market:
        st.session_state.screener_df = pd.DataFrame()
        st.session_state.selected_ticker = None
        st.session_state.last_market = market

    usa_bucket = None
    if market == "USA":
        usa_bucket = st.selectbox(
            "USA Universe",
            ["S&P 500", "Nasdaq 100"],
            index=0,
            help="Hangi endeksteki hisseleri tarayacağınızı seçin. S&P 500 daha geniş, Nasdaq 100 teknoloji ağırlıklı.",
        )

    st.header("1) Fundamental Screener")
    use_fa = st.checkbox(
        "Fundamental filtreyi kullan",
        value=True,
        help="Temel analiz (FA) kurallarını aktifleştirir. Şirketlerin mali tablolarını değerlendirerek puanlar.",
    )
    fa_mode = st.selectbox(
        "Fundamental Mod",
        ["Quality", "Value", "Growth"],
        index=0,
        disabled=(not use_fa),
        help="Quality: Karlılık ve düşük borç arar. Value: Ucuz kalmış hisseleri bulur. Growth: Yüksek büyüme oranlarına odaklanır.",
    )

    st.caption("Eşikler (Genel) — BIST'te coverage düşük olabilir")
    roe = st.slider(
        "ROE min",
        0.0,
        0.40,
        0.15,
        0.01,
        disabled=(not use_fa),
        help="Özkaynak Karlılığı (Return on Equity). Şirketin öz sermayesini ne kadar verimli kullandığını gösterir. Yüksek ROE genellikle iyidir.",
    )
    op_margin = st.slider(
        "Operating Margin min",
        0.0,
        0.40,
        0.10,
        0.01,
        disabled=(not use_fa),
        help="Faaliyet Kar Marjı. Şirketin ana faaliyetlerinden elde ettiği kârlılık. Sektör ortalamasıyla karşılaştırın.",
    )
    profit_margin = st.slider(
        "Profit Margin min",
        0.0,
        0.40,
        0.08,
        0.01,
        disabled=(not use_fa),
        help="Net Kar Marjı. Tüm giderler ve vergiler düşüldükten sonra kalan net kârın satışlara oranı.",
    )
    dte = st.slider(
        "Debt/Equity max",
        0.0,
        3.0,
        1.0,
        0.05,
        disabled=(not use_fa),
        help="Borç/Özkaynak oranı. Şirketin ne kadar borçlu olduğunu gösterir. 1.0 altı genellikle güvenli kabul edilir.",
    )
    fpe = st.slider(
        "Forward P/E max",
        0.0,
        60.0,
        20.0,
        1.0,
        disabled=(not use_fa),
        help="İleri F/K (Fiyat/Kazanç) oranı. Gelecek yıl beklenen kazançlara göre hissenin ucuz/pahalı olduğunu gösterir.",
    )
    peg = st.slider(
        "PEG max",
        0.0,
        5.0,
        1.5,
        0.1,
        disabled=(not use_fa),
        help="F/K / Büyüme oranı. 1.0 civarı hissenin büyüme potansiyeline göre adil fiyatlandığını gösterir.",
    )
    ps = st.slider(
        "P/S max",
        0.0,
        30.0,
        6.0,
        0.5,
        disabled=(not use_fa),
        help="Fiyat/Satış oranı. Henüz kâr etmeyen ancak satışları büyük şirketler için kullanılır.",
    )
    pb = st.slider(
        "P/B max",
        0.0,
        30.0,
        6.0,
        0.5,
        disabled=(not use_fa),
        help="Piyasa Değeri / Defter Değeri. Şirketin net varlıklarına göre kaç katından işlem gördüğü.",
    )
    rev_g = st.slider(
        "Revenue Growth min",
        0.0,
        0.50,
        0.10,
        0.01,
        disabled=(not use_fa),
        help="Yıllık ciro büyümesi. Şirketin satışlarının ne kadar arttığını gösterir.",
    )
    earn_g = st.slider(
        "Earnings Growth min",
        0.0,
        0.50,
        0.10,
        0.01,
        disabled=(not use_fa),
        help="Yıllık kâr büyümesi. Net kârdaki artış oranı.",
    )

    min_score = st.slider(
        "Min Fundamental Score",
        0,
        100,
        60,
        1,
        disabled=(not use_fa),
        help="Ağırlıklı puanın alt limiti. Bu puanın üzerindeki hisseler 'PASS' olarak işaretlenir.",
    )
    min_ok = st.slider(
        "Min OK sayısı",
        1,
        5,
        3,
        1,
        disabled=(not use_fa),
        help="Başarılı kriter sayısı. En az bu kadar kriteri sağlamalı.",
    )
    min_coverage = st.slider(
        "Min Coverage (NaN olmayan)",
        1,
        5,
        3,
        1,
        disabled=(not use_fa),
        help="Verisi olan kriter sayısı. BIST'te veri eksikliği olabileceği için bu sayı düşük tutulabilir.",
    )

    thresholds = {
        "roe": roe,
        "op_margin": op_margin,
        "profit_margin": profit_margin,
        "dte": dte,
        "fpe": fpe,
        "peg": peg,
        "ps": ps,
        "pb": pb,
        "rev_g": rev_g,
        "earn_g": earn_g,
        "min_score": min_score,
        "min_ok": min_ok,
        "min_coverage": min_coverage,
    }

    if market == "USA":
        if usa_bucket == "S&P 500":
            universe = load_universe_file(pjoin("universes", "sp500.txt"))
        else:
            universe = load_universe_file(pjoin("universes", "nasdaq100.txt"))
        st.caption(f"Universe: {usa_bucket} (count: {len(universe)})")
    else:
        universe = load_universe_file(pjoin("universes", "bist100.txt"))
        st.caption(f"Universe: BIST100 (count: {len(universe)})")

    if not universe:
        st.error("Universe listesi boş!")
        st.stop()

    run_screener = st.button("🔎 Screener Çalıştır", type="secondary", disabled=(not use_fa))

    st.divider()
    st.header("2) Teknik Analiz + Backtest")
    preset_name = st.selectbox(
        "Teknik Mod",
        list(PRESETS.keys()),
        index=1,
        help="Önceden tanımlı risk profilleri. Defansif: düşük risk, Agresif: yüksek risk.",
    )

    st.subheader("Sembol (TA)")
    if st.session_state.selected_ticker:
        st.caption(f"Screener seçimi: **{st.session_state.selected_ticker}**")
        raw_ticker = st.text_input("Sembol", value=st.session_state.selected_ticker)
    else:
        raw_ticker = st.text_input(
            "Sembol (USA: AAPL, SPY) / BIST: THYAO",
            value="AAPL" if market == "USA" else "THYAO",
            help="Analiz etmek istediğiniz hisse senedinin sembolü. BIST için otomatik .IS eklenir.",
        )

    ticker = normalize_ticker(raw_ticker, market)

    st.subheader("Zaman Aralığı")
    interval = st.selectbox(
        "Interval",
        ["1d", "1wk", "4h", "1h"],
        index=0,
        help="Mum zaman dilimi. 1d günlük, 1wk haftalık, 4h 4 saatlik, 1h saatlik analiz için. Backtest için 1d önerilir.",
    )
    period = st.selectbox(
        "Periyot",
        ["45d", "3mo", "6mo", "1y", "2y"],
        index=3,
        help="Verinin ne kadar geriye gidileceği. Daha uzun periyot daha sağlıklı backtest sağlar.",
    )

    use_custom_end_date = st.checkbox("Geçmiş Bir Tarihe Göre Analiz Yap (Repaint Önleme)", value=False)
    if use_custom_end_date:
        bugun = datetime.date.today()
        gecen_cuma = bugun - datetime.timedelta(days=bugun.weekday() + 3)
        
        custom_end_date = st.date_input(
            "Bitiş Tarihi Seçin", 
            value=gecen_cuma,
            help="Seçtiğiniz tarihe kadar olan veriler çekilir. Haftalık kapanışlar için Cuma gününü seçin."
        )
    else:
        custom_end_date = None

    force_latest_candle = st.checkbox(
        "Eksik Güncel Mumu Zorla Ekle (Live Candle Hack)", 
        value=False, 
        disabled=(use_custom_end_date or interval != "1d"),
        help="Yahoo Finance günlük mumu henüz vermediyse gün içi dakikalık verilerden o mumu inşa eder. (Sadece günlük periyotta ve güncel analizde çalışır)"
    )

    st.divider()
    st.subheader("Teknik Parametreler")
    ema_fast = st.number_input(
        "EMA Fast",
        min_value=5,
        max_value=100,
        value=50,
        step=1,
        help="Kısa vadeli üstel hareketli ortalama. Fiyatın bu ortalamanın üstünde olması kısa vadeli yükseliş trendini gösterir.",
    )
    ema_slow = st.number_input(
        "EMA Slow",
        min_value=50,
        max_value=400,
        value=200,
        step=1,
        help="Uzun vadeli üstel hareketli ortalama. Fiyat bu ortalamanın üstündeyse ana trend yükseliş, altındaysa düşüş trendi.",
    )
    rsi_period = st.number_input(
        "RSI Period",
        min_value=5,
        max_value=30,
        value=14,
        step=1,
        help="RSI hesaplama periyodu. 14 gün standarttır. 70 üstü aşırı alım, 30 altı aşırı satım.",
    )
    bb_period = st.number_input(
        "Bollinger Period",
        min_value=10,
        max_value=50,
        value=20,
        step=1,
        help="Bollinger bandı ortalama periyodu. Fiyatın üst banda yaklaşması aşırı alım, alt banda yaklaşması aşırı satım.",
    )
    bb_std = st.number_input(
        "Bollinger Std",
        min_value=1.0,
        max_value=3.5,
        value=2.0,
        step=0.1,
        help="Bollinger bandı standart sapma katsayısı. 2 standart sapma %95 güven aralığı verir.",
    )
    atr_period = st.number_input(
        "ATR Period",
        min_value=5,
        max_value=30,
        value=14,
        step=1,
        help="Ortalama Gerçek Aralık (Average True Range) periyodu. Volatilitenin ölçüsü, stop seviyesi belirlemede kullanılır.",
    )
    vol_sma = st.number_input(
        "Volume SMA",
        min_value=5,
        max_value=60,
        value=20,
        step=1,
        help="Hacim basit hareketli ortalaması. Hacim bu ortalamanın üzerindeyse likidite yüksek, işlem anlamlıdır.",
    )

    st.subheader("Market Filtreleri")
    use_spy_filter = st.checkbox(
        "SPY > EMA200 filtresi (Sadece USA)",
        value=True,
        disabled=(market != "USA"),
        help="S&P 500 endeksi 200 günlük ortalamanın altındaysa (ayı piyasası) alım sinyallerini engeller.",
    )
    use_bist_filter = st.checkbox(
        "XU100 > EMA200 filtresi (Sadece BIST)",
        value=True,
        disabled=(market != "BIST"),
        help="BIST 100 endeksi 200 günlük ortalamanın altındaysa alım sinyallerini engeller.",
    )
    use_higher_tf_filter = st.checkbox(
        "Haftalık trend filtresi (Fiyat > EMA200)",
        value=True,
        help="Haftalık grafikte fiyatın 200 haftalık ortalamanın üzerinde olması gerekir. Ana trendin yükseliş olduğunu onaylar.",
    )

    st.subheader("Risk / Backtest Ayarları")
    initial_capital = st.number_input(
        "Başlangıç Sermayesi",
        min_value=100.0,
        value=10000.0,
        step=500.0,
        help="Backtest için simüle edilecek başlangıç parası.",
    )
    risk_per_trade = st.slider(
        "Trade başı risk (equity %)",
        min_value=0.002,
        max_value=0.05,
        value=0.01,
        step=0.001,
        help="Her işlemde kasanın yüzde kaçını riske edeceğiniz. Stop loss ile kaybedilecek maksimum miktar.",
    )
    commission_bps = st.number_input(
        "Komisyon (bps)",
        min_value=0.0,
        value=5.0,
        step=1.0,
        help="İşlem başına komisyon (baz puan). 1 bps = %0.01.",
    )
    slippage_bps = st.number_input(
        "Slippage (bps)",
        min_value=0.0,
        value=2.0,
        step=1.0,
        help="Kayma maliyeti. Sinyal fiyatından daha kötü fiyattan işlem gerçekleşme riski.",
    )
    risk_free_annual = st.number_input(
        "Risk-Free (yıllık)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help="Risksiz faiz oranı (örnek: 0.05 = %5). Sharpe ve Sortino hesaplamalarında kullanılır.",
    )

    st.divider()
    st.header("3) AI Ayarları (Gemini)")
    ai_on = st.checkbox(
        "Gemini AI aktif",
        value=True,
        help="Google Gemini AI ile grafik ve veri analizi yapılır.",
    )
    gemini_model = st.text_input(
        "Gemini Model",
        value="gemini-1.5-flash",
        help="Kullanılacak Gemini modeli. 1.5-flash hızlı ve yeterlidir.",
    )
    gemini_temp = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.2,
        0.05,
        help="Modelin yaratıcılığı. Düşük değerler daha tutarlı, yüksek değerler daha yaratıcı cevaplar üretir.",
    )
    gemini_max_tokens = st.slider(
        "Max Output Tokens",
        256,
        8192,
        2048,
        128,
        help="Modelin üreteceği maksimum token sayısı. Daha uzun cevaplar için artırın.",
    )

    st.divider()
    st.header("4) Haber Duygu Analizi (Google News + Gemini)")
    use_sentiment = st.checkbox(
        "Haber duygu analizini aktifleştir",
        value=True,
        help="Google News'ten haber başlıklarını çeker, Gemini ile duygu analizi yapar.",
    )

    run_btn = st.button("🚀 Teknik Analizi Çalıştır", type="primary")
    if run_btn:
        st.session_state.ta_ran = True

# -----------------------------
# Config
# -----------------------------
cfg = {
    "ema_fast": ema_fast,
    "ema_slow": ema_slow,
    "rsi_period": rsi_period,
    "bb_period": bb_period,
    "bb_std": bb_std,
    "atr_period": atr_period,
    "vol_sma": vol_sma,
    "initial_capital": initial_capital,
    "risk_per_trade": risk_per_trade,
    "commission_bps": commission_bps,
    "slippage_bps": slippage_bps,
}
cfg.update(PRESETS[preset_name])


# -----------------------------
# Fundamental screener action
# -----------------------------
if run_screener and use_fa:
    with st.spinner(f"Fundamental veriler çekiliyor ({market})... (Bu işlem çoklu iş parçacığıyla hızlandırılmıştır)"):
        rows = []
        
        def fetch_one(tk):
            tk_norm = normalize_ticker(tk, market)
            f = fetch_fundamentals_generic(tk_norm, market=market)
            score, breakdown, passed = fundamental_score_row(f, fa_mode, thresholds)
            f["FA_score"] = score
            f["FA_pass"] = passed
            f["FA_ok_count"] = sum(1 for v in breakdown.values() if v.get("available") and v.get("ok"))
            f["FA_coverage"] = sum(1 for v in breakdown.values() if v.get("available"))
            return f

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(fetch_one, tk): tk for tk in universe}
            for future in as_completed(futures):
                try:
                    rows.append(future.result())
                except Exception:
                    pass

        sdf = pd.DataFrame(rows)
        if not sdf.empty:
            sdf["FA_pass_int"] = sdf["FA_pass"].astype(int)
            sdf = sdf.sort_values(["FA_pass_int", "FA_score", "FA_coverage"], ascending=[False, False, False]).drop(columns=["FA_pass_int"])
        st.session_state.screener_df = sdf.copy()


# -----------------------------
# If TA not ran yet: show screener and stop
# -----------------------------
if not st.session_state.ta_ran:
    if use_fa and not st.session_state.screener_df.empty:
        st.subheader(f"🧾 Fundamental Screener Sonuçları ({market})")
        sdf = st.session_state.screener_df.copy()

        show_cols = [
            "ticker",
            "longName",
            "FA_pass",
            "FA_score",
            "FA_ok_count",
            "FA_coverage",
            "sector",
            "industry",
            "trailingPE",
            "forwardPE",
            "pegRatio",
            "priceToSalesTrailing12Months",
            "priceToBook",
            "returnOnEquity",
            "operatingMargins",
            "profitMargins",
            "debtToEquity",
            "revenueGrowth",
            "earningsGrowth",
            "marketCap",
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

    st.info("Sol menüden ayarları yapıp **Teknik Analizi Çalıştır**’a basın.")
    st.stop()


# =============================
# Run TA pipeline
# =============================
market_filter_ok = True
if market == "USA" and use_spy_filter:
    with st.spinner("SPY rejimi kontrol ediliyor..."):
        market_filter_ok = get_spy_regime_ok()
elif market == "BIST" and use_bist_filter:
    with st.spinner("XU100 rejimi kontrol ediliyor..."):
        market_filter_ok = get_bist_regime_ok()

higher_tf_filter_ok = True
if use_higher_tf_filter:
    with st.spinner("Haftalık trend kontrol ediliyor..."):
        higher_tf_filter_ok = get_higher_tf_trend(ticker, higher_tf_interval="1wk", ema_period=200)

sentiment_summary = ""
if use_sentiment and ai_on:
    company_name = ""
    if not st.session_state.screener_df.empty:
        row = find_screener_row(st.session_state.screener_df, ticker)
        if row and row.get("longName"):
            company_name = row["longName"]

    with st.spinner("Google News'ten haberler çekiliyor ve Gemini ile analiz ediliyor..."):
        sent = get_news_sentiment(ticker, company_name, gemini_model, gemini_temp, gemini_max_tokens)
        if sent.get("error") is None:
            sentiment_summary = sent["summary"]
            st.session_state.sentiment_summary = sentiment_summary
            st.session_state.sentiment_items = sent.get("news_items", [])
        else:
            sentiment_summary = f"Haber analizi başarısız: {sent['error']}"
            st.session_state.sentiment_summary = sentiment_summary
            st.session_state.sentiment_items = []
elif use_sentiment and not ai_on:
    st.warning("Haber duygu analizi için Gemini'nin açık olması gerekir.")


with st.spinner(f"Veri indiriliyor: {ticker}"):
    df_raw = load_data_cached(ticker, period, interval, end_date=custom_end_date, force_latest=force_latest_candle)

if df_raw.empty:
    st.error(f"Veri gelmedi: {ticker}")
    st.stop()

required_cols = {"Open", "High", "Low", "Close", "Volume"}
if not required_cols.issubset(set(df_raw.columns)):
    st.error("Veri setinde gerekli OHLCV kolonları eksik.")
    st.stop()

if len(df_raw) < 260 and interval == "1d":
    st.warning("Günlükte 260 bar altı: metrikler daha oynak olabilir.")

df = build_features(df_raw, cfg)

benchmark_ticker = "SPY" if market == "USA" else "XU100.IS"
benchmark_df = load_data_cached(benchmark_ticker, period, interval, end_date=custom_end_date, force_latest=False)
benchmark_returns = benchmark_df["Close"].pct_change().dropna() if not benchmark_df.empty else None

df, checkpoints = signal_with_checkpoints(
    df,
    cfg,
    market_filter_ok=market_filter_ok,
    higher_tf_filter_ok=higher_tf_filter_ok,
)
latest = df.iloc[-1]

live = get_live_price(ticker)
live_price = live.get("last_price", np.nan)

if int(latest["ENTRY"]) == 1:
    rec = "AL"
elif int(latest["EXIT"]) == 1:
    rec = "SAT"
else:
    rec = "AL (Güçlü Trend)" if latest["SCORE"] >= 80 else ("İZLE (Orta)" if latest["SCORE"] >= 60 else "UZAK DUR")

eq, tdf, metrics = backtest_long_only(df, cfg, risk_free_annual=risk_free_annual, benchmark_returns=benchmark_returns)
tp = target_price_band(df)
rr_info = rr_from_atr_stop(latest, tp, cfg)
overbought_result = detect_speculation(df)


# =============================
# Build figures
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

bull_tails = df[df["KANGAROO_BULL"] == 1]
bear_tails = df[df["KANGAROO_BEAR"] == 1]
fig_price.add_trace(go.Scatter(x=bull_tails.index, y=bull_tails["Low"], mode="markers+text", name="Kanguru (Boğa)", text="🦘", textposition="bottom center", marker=dict(symbol="circle", size=8, color="purple")))
fig_price.add_trace(go.Scatter(x=bear_tails.index, y=bear_tails["High"], mode="markers+text", name="Kanguru (Ayı)", text="🦘", textposition="top center", marker=dict(symbol="circle", size=8, color="purple")))

fig_price.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    title="Fiyat Grafiği + EMA + Bollinger + Sinyaller & Kanguru",
    yaxis_title="Fiyat",
    xaxis_title="Tarih",
)

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Aşırı Alım")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Aşırı Satım")
fig_rsi.update_layout(
    height=260,
    title="RSI (Göreceli Güç Endeksi)",
    yaxis_title="RSI",
    xaxis_title="Tarih",
)

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"))
fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
fig_macd.update_layout(
    height=260,
    title="MACD (Moving Average Convergence Divergence)",
    yaxis_title="MACD",
    xaxis_title="Tarih",
)

fig_atr = go.Figure()
fig_atr.add_trace(go.Scatter(x=df.index, y=df["ATR_PCT"] * 100, name="ATR%"))
fig_atr.update_layout(
    height=260,
    title="ATR % (Ortalama Gerçek Aralık / Fiyat)",
    yaxis_title="%",
    xaxis_title="Tarih",
)

fig_stoch = go.Figure()
if "STOCH_RSI_K" in df.columns and "STOCH_RSI_D" in df.columns:
    fig_stoch.add_trace(go.Scatter(x=df.index, y=df["STOCH_RSI_K"], name="Stochastic RSI K"))
    fig_stoch.add_trace(go.Scatter(x=df.index, y=df["STOCH_RSI_D"], name="Stochastic RSI D"))
    fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Aşırı Alım")
    fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Aşırı Satım")
fig_stoch.update_layout(
    height=260,
    title="Stochastic RSI (K & D)",
    yaxis_title="Değer",
    xaxis_title="Tarih",
)

fig_bbwidth = go.Figure()
if "BB_WIDTH" in df.columns:
    fig_bbwidth.add_trace(go.Scatter(x=df.index, y=df["BB_WIDTH"] * 100, name="BB % Genişlik"))
    fig_bbwidth.add_hline(y=2, line_dash="dash", line_color="orange", annotation_text="Sıkışma Bölgesi")
fig_bbwidth.update_layout(
    height=260,
    title="Bollinger Bandı Genişliği %",
    yaxis_title="Genişlik %",
    xaxis_title="Tarih",
)

fig_volratio = go.Figure()
if "VOL_RATIO" in df.columns:
    fig_volratio.add_trace(go.Bar(x=df.index, y=df["VOL_RATIO"], name="Hacim Oranı"))
    fig_volratio.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Anormal Hacim")
fig_volratio.update_layout(
    height=260,
    title="Hacim Oranı (Son Hacim / SMA)",
    yaxis_title="Oran",
    xaxis_title="Tarih",
)

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity"))
fig_eq.update_layout(
    height=320,
    title="Backtest Sermaye Eğrisi",
    yaxis_title="Sermaye",
    xaxis_title="Tarih",
)

# YENİ EKLENEN GRAFİKLER (Hacim Kıyaslamaları)
df["VOL_SMA_10"] = df["Volume"].rolling(10).mean()
if benchmark_df is not None and not benchmark_df.empty:
    bench_vol = benchmark_df["Volume"].reindex(df.index).fillna(0)
else:
    bench_vol = pd.Series(0, index=df.index)

fig_vol_market = make_subplots(specs=[[{"secondary_y": True}]])
fig_vol_market.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Hisse Hacmi", marker_color='lightblue', opacity=0.7), secondary_y=False)
fig_vol_market.add_trace(go.Scatter(x=df.index, y=bench_vol, name=f"Endeks ({benchmark_ticker})", line=dict(color='orange', width=2)), secondary_y=True)
fig_vol_market.update_layout(height=260, title=f"Hisse vs Endeks Hacmi", yaxis_title="Hisse", yaxis2_title="Endeks", xaxis_title="Tarih", margin=dict(l=0, r=0, t=40, b=0))

fig_vol_2wk = go.Figure()
fig_vol_2wk.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Hacim", marker_color='cadetblue', opacity=0.7))
fig_vol_2wk.add_trace(go.Scatter(x=df.index, y=df["VOL_SMA_10"], name="2 Haftalık Ort. (10 Bar)", line=dict(color='red', width=2)))
fig_vol_2wk.update_layout(height=260, title="Hisse Hacmi vs 2 Haftalık Ortalama", yaxis_title="Hacim", xaxis_title="Tarih", margin=dict(l=0, r=0, t=40, b=0))


figs_for_report = {
    "Price + EMA + Bollinger + Signals": fig_price,
    "RSI": fig_rsi,
    "MACD": fig_macd,
    "ATR%": fig_atr,
    "Stochastic RSI": fig_stoch,
    "Bollinger Band Width": fig_bbwidth,
    "Volume Ratio": fig_volratio,
    "Volume vs Market": fig_vol_market,
    "Volume vs 2W Avg": fig_vol_2wk,
    "Equity Curve": fig_eq,
}


# =============================
# Tabs
# =============================
tab_dash, tab_export, tab_heatmap, tab_triple = st.tabs(["📊 Dashboard", "📄 Rapor (PDF/HTML)", "🔥 Sektörel Heatmap", "📺 3 Ekranlı Sistem"])


with tab_dash:
    if use_fa and not st.session_state.screener_df.empty:
        st.subheader(f"🧾 Fundamental Screener Sonuçları ({market})")
        sdf = st.session_state.screener_df.copy()
        show_cols = [
            "ticker",
            "longName",
            "FA_pass",
            "FA_score",
            "FA_ok_count",
            "FA_coverage",
            "sector",
            "industry",
            "trailingPE",
            "forwardPE",
            "pegRatio",
            "priceToSalesTrailing12Months",
            "priceToBook",
            "returnOnEquity",
            "operatingMargins",
            "profitMargins",
            "debtToEquity",
            "revenueGrowth",
            "earningsGrowth",
            "marketCap",
        ]
        sdf_show = sdf[[c for c in show_cols if c in sdf.columns]].copy()
        st.dataframe(sdf_show, use_container_width=True, height=360)

    st.subheader("📊 Aşırı Alım / Spekülasyon Göstergeleri")
    col_ob1, col_ob2, col_ob3, col_ob4 = st.columns(4)
    col_ob1.metric("Aşırı Alım Skoru", f"{overbought_result['overbought_score']}/100", help="Yüksek skor aşırı değerli olduğunu gösterir (RSI, Bollinger, Stokastik RSI, EMA uzaklığı).")
    col_ob2.metric("Aşırı Satım Skoru", f"{overbought_result['oversold_score']}/100", help="Yüksek skor aşırı değersiz olduğunu gösterir.")
    col_ob3.metric("Spekülasyon Skoru", f"{overbought_result['speculation_score']}/100", help="Ani hacim artışı ve zayıf trend (fiyat yükselirken hacim düşüyor).")
    col_ob4.metric("Genel Karar", overbought_result["verdict"])

    with st.expander("Detaylı Aşırı Alım/Spekülasyon Analizi"):
        for _, v in overbought_result["details"].items():
            st.write(f"• {v}")

    c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9)
    c1.metric("Market", market, help="Seçili piyasa.")
    c2.metric("Sembol", ticker, help="Analiz edilen hisse senedi.")
    c3.metric("Daily Close", f"{latest['Close']:.2f}", help="Son mumun kapanış fiyatı (düzeltilmiş).")
    c4.metric("Live/Last", f"{live_price:.2f}" if np.isfinite(live_price) else "N/A", help="Anlık piyasa fiyatı (çekilebildiyse).")
    c5.metric("Skor", f"{latest['SCORE']:.0f}/100", help="Teknik strateji skoru. 0-100 arası, yüksek skor daha güçlü sinyal.")
    c6.metric("Sinyal", rec, help="Algoritmanın son durumu: AL, SAT, İZLE.")
    c7.metric("Piyasa Filtresi", "BULL ✅" if market_filter_ok else "BEAR ❌", help="SPY veya XU100 endeksi > 200 EMA")
    c8.metric("Haftalık Trend", "BULL ✅" if higher_tf_filter_ok else "BEAR ❌", help="Haftalık fiyat > 200 EMA")
    
    is_bull_tail = latest.get("KANGAROO_BULL", 0) == 1
    is_bear_tail = latest.get("KANGAROO_BEAR", 0) == 1
    tail_val = "BOĞA 🦘" if is_bull_tail else ("AYI 🦘" if is_bear_tail else "YOK")
    tail_delta = "AL Yönlü" if is_bull_tail else ("-SAT Yönlü" if is_bear_tail else None)
    c9.metric("Kanguru", tail_val, delta=tail_delta, help="Kanguru Kuyruğu (Kangaroo Tail) formasyonu tespiti.")

    st.subheader("✅ Kontrol Noktaları (Son Bar)")
    cp_cols = st.columns(3)
    cp_items = list(checkpoints.items())
    for i, (k, v) in enumerate(cp_items):
        with cp_cols[i % 3]:
            if k == "Market Filter OK":
                st.metric(k, "✅" if v else "❌", help="Piyasa endeksi 200 günlük ortalamanın üstünde mi?")
            elif k == "Higher TF Filter OK":
                st.metric(k, "✅" if v else "❌", help="Haftalık grafikte fiyat 200 haftalık EMA'nın üstünde mi?")
            elif k == "Liquidity (Volume > VolSMA)":
                st.metric(k, "✅" if v else "❌", help="Hacim, hareketli ortalamanın üstünde mi? (Likidite kontrolü)")
            elif k == "Trend (Close>EMA200 & EMA50>EMA200)":
                st.metric(k, "✅" if v else "❌", help="Fiyat EMA200 üstünde ve EMA50 EMA200 üstünde mi? (Trend yönü)")
            elif k.startswith("RSI >"):
                st.metric(k, "✅" if v else "❌", help="RSI değeri giriş eşiğinin üstünde mi? (Momentum)")
            elif k == "MACD Hist > 0":
                st.metric(k, "✅" if v else "❌", help="MACD histogramı pozitif mi? (Momentum yönü)")
            elif k.startswith("ATR% <"):
                st.metric(k, "✅" if v else "❌", help="ATR'nin fiyata oranı belirlenen maksimumun altında mı? (Volatilite kontrolü)")
            elif k == "Bollinger (Close>BB_mid or Breakout)":
                st.metric(k, "✅" if v else "❌", help="Fiyat Bollinger orta bandının üstünde veya üst banda kırılmış mı?")
            elif k == "OBV > OBV_EMA":
                st.metric(k, "✅" if v else "❌", help="On-Balance Volume kendi EMA'sının üstünde mi? (Hacim trendi)")
            else:
                st.metric(k, "✅" if v else "❌")

    st.subheader("🎯 Hedef Fiyat Bandı (Senaryo)")
    base_px = float(tp["base"])
    rr_str = fmt_rr(rr_info.get("rr"))

    bcol1, bcol2, bcol3 = st.columns(3)
    bcol1.metric("Base", f"{base_px:.2f}", help="Referans alınan anlık/kapanış fiyat.")

    if tp.get("bull"):
        bull1, bull2, r1 = tp["bull"]
        bcol2.metric("Bull Band", f"{bull1:.2f} → {bull2:.2f}", help="ATR bazlı yukarı yönlü dinamik hedef bölgesi. İlk hedef, ikinci hedef.")
        if r1 is not None and np.isfinite(r1):
            r1_info = tp.get("r1_dict") or {}
            dur = r1_info.get("duration_bars", 0)
            vol_pct = r1_info.get("vol_diff_pct", 0)
            str_pct = r1_info.get("strength_pct", 0)
            bcol2.caption(f"Yakın direnç: {r1:.2f} ({pct_dist(r1, base_px):+.2f}%)\n\n**Güç:** %{str_pct:.0f} | **Uzunluk:** {dur} Bar | **Hacim:** %{vol_pct:+.1f} (Ort.)")
    else:
        bcol2.metric("Bull Band", "N/A")
        r1 = None

    if tp.get("bear"):
        bear1, bear2, s1 = tp["bear"]
        target_info = f" | Hedef: {rr_info.get('target_type','')}" if rr_info.get('target_type') else ""
        bcol3.metric("Bear Band", f"{bear1:.2f} → {bear2:.2f}  |  RR {rr_str}{target_info}", help="ATR bazlı aşağı yönlü dinamik stop bölgesi ve Risk/Ödül oranı.")
        if s1 is not None and np.isfinite(s1):
            s1_info = tp.get("s1_dict") or {}
            dur = s1_info.get("duration_bars", 0)
            vol_pct = s1_info.get("vol_diff_pct", 0)
            str_pct = s1_info.get("strength_pct", 0)
            bcol3.caption(f"Yakın destek: {s1:.2f} ({pct_dist(s1, base_px):+.2f}%)\n\n**Güç:** %{str_pct:.0f} | **Uzunluk:** {dur} Bar | **Hacim:** %{vol_pct:+.1f} (Ort.)")
    else:
        bcol3.metric("Bear Band", f"N/A  |  RR {rr_str}")
        s1 = None

    def render_levels_marked(levels: List[dict], base: float, s1, r1):
        lines = []
        for lv_dict in (levels or []):
            lv = float(lv_dict["price"])
            dur = lv_dict["duration_bars"]
            vol_pct = lv_dict["vol_diff_pct"]
            str_pct = lv_dict["strength_pct"]

            tag = ""
            if s1 is not None and np.isfinite(s1) and abs(lv - float(s1)) < 1e-9:
                tag = " 🟩 Yakın Destek"
            if r1 is not None and np.isfinite(r1) and abs(lv - float(r1)) < 1e-9:
                tag = " 🟥 Yakın Direnç"

            dist = pct_dist(lv, base)
            dist_txt = f"{dist:+.2f}%" if dist is not None else ""
            lines.append(f"- **{lv:.2f}** ({dist_txt}) | Güç: %{str_pct:.0f} | Uzunluk: {dur} Bar | Hacim: %{vol_pct:+.1f} {tag}")
        return "\n".join(lines) if lines else "_Seviye yok_"

    with st.expander("Seviye listesi (yaklaşık) — işaretli + fiyata uzaklık %", expanded=False):
        st.markdown(render_levels_marked(tp.get("levels", []), base_px, s1, r1))

    st.subheader("📊 Fiyat + EMA + Bollinger + Sinyaller")
    st.plotly_chart(fig_price, use_container_width=True)
    st.caption("**Yorum:** Mum grafiği, EMA50 (kısa vade), EMA200 (uzun vade) ve Bollinger bantları. Üçgenler alım (yeşil) ve satım (kırmızı) sinyallerini gösterir. Fiyat EMA200 üzerinde trend yükseliş, altında düşüş. Kanguru ikonları tuzak alanlarını belirtir.")

    st.subheader("📉 RSI / MACD / ATR%")
    colA, colB, colC = st.columns(3)
    with colA:
        st.plotly_chart(fig_rsi, use_container_width=True)
        st.caption("**RSI:** 70 üstü aşırı alım (düşüş beklenebilir), 30 altı aşırı satım (yükseliş beklenebilir).")
    with colB:
        st.plotly_chart(fig_macd, use_container_width=True)
        st.caption("**MACD:** MACD çizgisi sinyal çizgisini yukarı keserse alım, aşağı keserse satım sinyali. Histogram pozitifse momentum yukarı.")
    with colC:
        st.plotly_chart(fig_atr, use_container_width=True)
        st.caption("**ATR%:** Volatilitenin fiyata oranı. Yüksek değerler yüksek oynaklık, düşük değerler sakin piyasa.")

    st.subheader("📊 Stochastic RSI / Bollinger Genişliği / Hacim Oranı")
    colD, colE, colF = st.columns(3)
    with colD:
        st.plotly_chart(fig_stoch, use_container_width=True)
        st.caption("**Stochastic RSI:** 80 üstü aşırı alım, 20 altı aşırı satım. K ve D'nin kesişmeleri sinyal verir.")
    with colE:
        st.plotly_chart(fig_bbwidth, use_container_width=True)
        st.caption("**Bollinger Genişliği %:** Düşük değerler (< %2) sıkışma (düşük volatilite), yüksek değerler genişleme (yüksek volatilite).")
    with colF:
        st.plotly_chart(fig_volratio, use_container_width=True)
        st.caption("**Hacim Oranı:** 1.5 üstü anormal hacim (spekülasyon), 0.5 altı düşük hacim (ilgisizlik).")

    # YENİ EKLENEN HACİM GRAFİKLERİ BÖLÜMÜ
    st.subheader("📊 Hacim Karşılaştırmaları (Endeks ve 2 Haftalık Ort.)")
    colV1, colV2 = st.columns(2)
    with colV1:
        st.plotly_chart(fig_vol_market, use_container_width=True)
        st.caption("**Hisse vs Endeks Hacmi:** Hissenin hacim eğilimi endeksle uyumlu mu? (Sağ eksen Endeks Hacmi)")
    with colV2:
        st.plotly_chart(fig_vol_2wk, use_container_width=True)
        st.caption("**Hisse vs 2 Haftalık Ortalama:** Son 10 barlık ortalamaya göre güncel hacim ne durumda?")

    st.subheader("🧪 Backtest Özeti (Long-only + Scale Out + Time Stop)")
    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    m1.metric("Total Return", f"{metrics['Total Return']*100:.1f}%", help="Backtest boyunca toplam getiri yüzdesi.")
    m2.metric("Ann Return", f"{metrics['Annualized Return']*100:.1f}%", help="Yıllıklandırılmış ortalama getiri.")
    m3.metric("Sharpe", f"{metrics['Sharpe']:.2f}", help="Sharpe oranı: getiri / risk. 1.0 üzeri iyi, 2.0 üzeri mükemmel.")
    m4.metric("Max DD", f"{metrics['Max Drawdown']*100:.1f}%", help="Maksimum düşüş: kasanın tepe noktasından en büyük kaybı.")
    m5.metric("Trades", f"{metrics['Trades']}", help="Toplam işlem sayısı.")
    m6.metric("Win Rate", f"{metrics['Win Rate']*100:.1f}%", help="Kazanan işlemlerin yüzdesi.")
    m7.metric("Beta", f"{metrics['Beta']:.2f}", help="Piyasaya duyarlılık. 1'den büyükse piyasadan daha oynak, küçükse daha az oynak.")
    m8.metric("Info Ratio", f"{metrics['Information Ratio']:.2f}", help="Aktif getirinin takip hatasına oranı. Yüksek değerler iyi.")

    with st.expander("Trade listesi (Detaylı Kâr/Zarar ve Çıkış Nedenleri)", expanded=False):
        st.dataframe(tdf, use_container_width=True, height=240)

    with st.expander("Equity curve (Sermaye Eğrisi)", expanded=False):
        st.plotly_chart(fig_eq, use_container_width=True)
        st.caption("**Sermaye Eğrisi:** Zaman içinde kasanın değişimi. Düşüşler (drawdown) riski gösterir.")

    if sentiment_summary:
        st.subheader("📰 Haber Duygu Analizi (Google News + Gemini)")
        st.info(sentiment_summary)
        
        if st.session_state.sentiment_items:
            st.markdown("**Kaynak Haberler:**")
            for item in st.session_state.sentiment_items:
                st.markdown(f"- [{item['title']}]({item['link']})")

    st.subheader("🤖 Gemini Multimodal AI — Grafik + Price Action + Spekülasyon Analizi")
    if not ai_on:
        st.info("Gemini kapalı (sol menüden açabilirsiniz).")
    else:
        pa = price_action_pack(df, last_n=20)
        st.session_state.pa_pack = pa

        user_msg = st.text_area(
            "Gemini'ye sor/talimat ver (spekülasyon sorusu eklendi):",
            value="Ekteki fiyat grafiği resmini ve aşağıdaki JSON'da bulunan son 20 barlık price-action verilerini incele. Ayrıca aşırı alım/spekülasyon göstergelerini de değerlendir (RSI, Bollinger, hacim sıçraması, fiyatın EMA'dan uzaklığı, Stochastic RSI). Bu hisse aşırı değerli mi, spekülatif bir hareket mi var? AL/SAT/İZLE önerisi ve stratejinin bozulacağı şartları yaz. Analizin sonunda aşağıdaki formatta bir tablo ekle:\n\n| Hedef | Fiyat |\n|-------|-------|\n| Alış Fiyatı (önerilen giriş) | ... |\n| Hedef Satış Fiyatı (ilk hedef) | ... |\n| Stop Loss (ATR bazlı) | ... |\n\n",
            height=150,
        )

        col_g1, col_g2 = st.columns([1, 1])

        with col_g1:
            if st.button("🖼️ Gemini'ye Sor (Görsel + Tüm Veriler)", use_container_width=True):
                snap20 = df_snapshot_for_llm(df, n=25)

                fa_row_local = None
                if use_fa and not st.session_state.screener_df.empty:
                    screener_row = find_screener_row(st.session_state.screener_df, ticker)
                    f_single = fetch_fundamentals_generic(ticker, market=market)
                    f_score, f_breakdown, f_pass = fundamental_score_row(f_single, fa_mode, thresholds)
                    fa_eval = {
                        "mode": fa_mode,
                        "score": f_score,
                        "passed": f_pass,
                        "ok_cnt": sum(1 for v in f_breakdown.values() if v.get("available") and v.get("ok")),
                        "coverage": sum(1 for v in f_breakdown.values() if v.get("available")),
                    }
                    fa_row_local = merge_fa_row(screener_row, f_single, fa_eval)

                sector_comp = ""
                if fa_row_local and fa_row_local.get("trailingPE") and fa_row_local.get("sector"):
                    sector_comp = f"Sektör: {fa_row_local['sector']}, F/K: {fa_row_local['trailingPE']:.2f}"

                sentiment_info = st.session_state.get("sentiment_summary", "")

                prompt = f"""
Sen bir price-action, formasyon okuma, aşırı alım/spekülasyon tespiti ve risk yönetimi odaklı kıdemli finansal analiz asistanısın. Kesin yatırım tavsiyesi verme, sadece objektif ve eğitim amaçlı analiz yap. Lütfen aşağıdaki adımları takip ederek analiz yap:
1. Genel trendi değerlendir (EMA50, EMA200, fiyatın bu ortalamalara göre konumu).
2. Temel destek/direnç seviyelerini belirle (price action paketindeki swing high/low, order block).
3. Aşırı alım/spekülasyon göstergelerini incele:
   - RSI (>70 aşırı alım, <30 aşırı satım)
   - Bollinger Bandı (fiyat üst bandın üstünde mi?)
   - Hacim sıçraması (son hacim normalin 1.5 katından fazla mı?)
   - Fiyatın EMA50'den uzaklığı (%20'den fazla mı?)
   - Stokastik RSI (>80 aşırı alım)
   - Fiyat yükselirken hacim düşüyor mu? (zayıflama)
4. Volatiliteyi (ATR), Kanguru Kuyruğu (Kangaroo Tail) formasyonu olup olmadığını ve stop seviyesi için fikir ver.
5. Temel analiz skorunu (FA) değerlendir (eğer varsa).
6. Haber duyarlılığını dikkate al (eğer varsa).
7. Tüm bu bilgileri sentezleyerek:
   - Hisse aşırı değerli mi, aşırı değersiz mi, yoksa normal bölgede mi?
   - Spekülatif bir hareket var mı? (ani hacim, zayıf trend)
   - AL/SAT/İZLE önerisi, hedef bant ve stratejinin bozulacağı şartlar.
Ekte sana analiz edilen hissenin grafiğinin GÖRSELİNİ (image) gönderdim. Görseli detaylıca incele. Ek olarak algoritmamızın ürettiği aşağıdaki JSON verilerini de referans al:

JSON:
{json.dumps({
    "ticker": ticker,
    "market": market,
    "algo_signal": rec,
    "latest_close": float(latest["Close"]),
    "target_band": tp,
    "rr_info": rr_info,
    "pa_pack": pa,
    "data_snapshot": snap20,
    "fundamental_score": fa_row_local.get("FA_score") if fa_row_local else None,
    "fundamental_pass": fa_row_local.get("FA_pass") if fa_row_local else None,
    "sector_info": sector_comp,
    "sentiment_analysis": sentiment_info,
    "overbought_analysis": overbought_result
}, ensure_ascii=False, default=str)}

Kullanıcının Sorusu: {user_msg}

Analizin sonunda aşağıdaki gibi bir tablo ile hedef alış ve satış fiyatlarını göster (fiyatları kendi hesapladığın seviyelerle doldur):

| Hedef | Fiyat |
|-------|-------|
| Alış Fiyatı (önerilen giriş) | ... |
| Hedef Satış Fiyatı (ilk hedef) | ... |
| Stop Loss (ATR bazlı) | ... |
Not: Eğer verdiğim JSON'da "rr_info" altında "target_type" varsa, hedef fiyatın neye dayandığını (direnç veya ATR) tabloda belirtebilirsin.
"""

                image_bytes = _plotly_fig_to_png_bytes(fig_price)

                text = gemini_generate_text(
                    prompt=prompt,
                    model=gemini_model,
                    temperature=gemini_temp,
                    max_output_tokens=gemini_max_tokens,
                    image_bytes=image_bytes,
                )
                st.session_state.gemini_text = text

        with col_g2:
            if st.button("Temizle", use_container_width=True):
                st.session_state.gemini_text = ""

        if st.session_state.gemini_text:
            st.markdown(st.session_state.gemini_text)


# =============================
# HEATMAP TAB
# =============================
with tab_heatmap:
    st.header("🔥 Sektörel Treemap (Heatmap)")
    st.write("Belirtilen pazar veya Screener sonuçları üzerinden şirketlerin günlük, haftalık ve aylık performanslarını görselleştirir.")

    if st.button("Heatmap Verilerini Getir ve Oluştur (1D, 1W, 1M)", type="primary"):
        with st.spinner("Toplu veri çekiliyor ve hesaplanıyor..."):
            if not st.session_state.screener_df.empty:
                hm_tickers = st.session_state.screener_df["ticker"].tolist()
            else:
                hm_tickers = universe[:100]

            use_tickers = [normalize_ticker(t, market) for t in hm_tickers]

            try:
                df_all = yf.download(use_tickers, period="1mo", interval="1d", auto_adjust=True, group_by="ticker", progress=False)
            except Exception:
                df_all = pd.DataFrame()

            hm_data = []
            for t in use_tickers:
                try:
                    if len(use_tickers) == 1:
                        df_t = df_all.copy()
                    else:
                        df_t = df_all[t].copy()

                    df_t = df_t.dropna()
                    if len(df_t) >= 2:
                        c_last = float(df_t["Close"].iloc[-1])
                        c_prev_1d = float(df_t["Close"].iloc[-2])
                        c_prev_1wk = float(df_t["Close"].iloc[-6]) if len(df_t) >= 6 else float(df_t["Close"].iloc[0])
                        c_prev_1mo = float(df_t["Close"].iloc[0])

                        ret_1d = (c_last / c_prev_1d - 1) * 100
                        ret_1wk = (c_last / c_prev_1wk - 1) * 100
                        ret_1mo = (c_last / c_prev_1mo - 1) * 100

                        sector = "Genel"
                        if not st.session_state.screener_df.empty:
                            row_match = find_screener_row(st.session_state.screener_df, t)
                            if row_match and pd.notna(row_match.get("sector")) and str(row_match.get("sector")).strip():
                                sector = str(row_match.get("sector"))

                        hm_data.append(
                            {
                                "Ticker": t,
                                "Sector": sector,
                                "1 Günlük %": ret_1d,
                                "1 Haftalık %": ret_1wk,
                                "1 Aylık %": ret_1mo,
                            }
                        )
                except Exception:
                    pass

            df_hm = pd.DataFrame(hm_data)

        if not df_hm.empty:
            df_hm["Abs_1D"] = df_hm["1 Günlük %"].abs()

            st.subheader("GÜNLÜK Performans")
            fig_hm_1d = px.treemap(
                df_hm,
                path=[px.Constant("Tüm Pazar"), "Sector", "Ticker"],
                values="Abs_1D",
                color="1 Günlük %",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                custom_data=["1 Günlük %", "1 Haftalık %", "1 Aylık %"],
            )
            fig_hm_1d.update_traces(
                hovertemplate="<b>%{label}</b><br>1 Günlük: %{customdata[0]:.2f}%<br>1 Haftalık: %{customdata[1]:.2f}%<br>1 Aylık: %{customdata[2]:.2f}%"
            )
            st.plotly_chart(fig_hm_1d, use_container_width=True)

            st.subheader("HAFTALIK Performans")
            df_hm["Abs_1W"] = df_hm["1 Haftalık %"].abs()
            fig_hm_1w = px.treemap(
                df_hm,
                path=[px.Constant("Tüm Pazar"), "Sector", "Ticker"],
                values="Abs_1W",
                color="1 Haftalık %",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
            )
            st.plotly_chart(fig_hm_1w, use_container_width=True)

            st.subheader("AYLIK Performans")
            df_hm["Abs_1M"] = df_hm["1 Aylık %"].abs()
            fig_hm_1m = px.treemap(
                df_hm,
                path=[px.Constant("Tüm Pazar"), "Sector", "Ticker"],
                values="Abs_1M",
                color="1 Aylık %",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
            )
            st.plotly_chart(fig_hm_1m, use_container_width=True)
        else:
            st.error("Heatmap için yeterli veri çekilemedi.")


# =============================
# EXPORT TAB
# =============================
with tab_export:
    st.subheader("📄 Rapor İndir (En sorunsuz: HTML → tarayıcıdan PDF)")
    st.caption("HTML rapor: grafikler %100 gelir. PDF: reportlab + (grafikler için) kaleido varsa grafikleri gömer. Yoksa PDF metin ağırlıklı olur.")

    include_charts = st.checkbox("Rapor grafikleri dahil et", value=True)
    include_trades = st.checkbox("Trade listesi dahil et (ilk 25)", value=True)
    include_gemini = st.checkbox("Gemini çıktısını rapora ekle", value=True)
    include_pa = st.checkbox("Price Action Pack'i rapora ekle", value=True)
    include_sentiment = st.checkbox("Haber duygu analizini rapora ekle", value=True)
    include_overbought = st.checkbox("Aşırı alım/spekülasyon analizini rapora ekle", value=True)

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

    gemini_text = st.session_state.gemini_text if include_gemini else None
    pa_pack_export = st.session_state.pa_pack if include_pa else None
    sentiment_export = st.session_state.sentiment_summary if include_sentiment else None
    sentiment_items_export = st.session_state.sentiment_items if include_sentiment else None
    overbought_export = overbought_result if include_overbought else None

    html_bytes = build_html_report(
        title=f"FA→TA Trading Report - {ticker}",
        meta=meta,
        checkpoints=checkpoints,
        metrics=metrics,
        tp=tp,
        rr_info=rr_info,
        figs=(figs_for_report if include_charts else {}),
        fa_row=fa_row,
        gemini_insight=gemini_text,
        pa_pack=pa_pack_export,
        sentiment_summary=sentiment_export,
        sentiment_items=sentiment_items_export,
        overbought_result=overbought_export,
    )
    st.download_button(
        "⬇️ HTML İndir (Önerilen)",
        data=html_bytes,
        file_name=f"{ticker}_FA_TA_report.html",
        mime="text/html",
        use_container_width=True,
    )

    st.divider()

    if not REPORTLAB_OK:
        st.warning("Doğrudan PDF için 'reportlab' gerekli. requirements.txt içine `reportlab` ekleyip redeploy edersen PDF butonu da aktif olur.")
    else:
        if st.button("🧾 PDF Oluştur (reportlab)", use_container_width=True):
            with st.spinner("PDF oluşturuluyor..."):
                pdf_bytes = generate_pdf_report(
                    title=f"FA→TA Trading Report - {ticker}",
                    subtitle="Educational analysis.",
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
                    include_charts=include_charts,
                    gemini_insight=gemini_text,
                    pa_pack=pa_pack_export,
                    sentiment_summary=sentiment_export,
                    sentiment_items=sentiment_items_export,
                    overbought_result=overbought_export,
                )

            if pdf_bytes:
                st.success("PDF hazır ✅")
                st.download_button(
                    "⬇️ PDF İndir",
                    data=pdf_bytes,
                    file_name=f"{ticker}_FA_TA_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.info("Grafikler gelmiyorsa `kaleido` kütüphanesini requirements'a eklemelisin.")
            else:
                st.error("PDF üretilemedi.")

# =============================
# TRIPLE SCREEN TAB
# =============================
with tab_triple:
    st.header("📺 Üçlü Ekran Trading Sistemi (Triple Screen)")
    st.caption("Dr. Alexander Elder'in 3 Ekranlı sistemine dayanan, trend, osilatör ve giriş seviyesi analizleri.")
    
    if not st.session_state.ta_ran:
        st.info("Sol menüden 'Teknik Analizi Çalıştır' butonuna basarak sistemi aktifleştirmelisin.")
    else:
        if st.button("Üçlü Ekran Verilerini Getir ve Analiz Et", key="run_triple"):
            with st.spinner("3 Ekran verileri hesaplanıyor (1W, 1D, 4H)..."):
                
                df_1w = load_data_cached(ticker, "2y", "1wk")
                df_1d = load_data_cached(ticker, "1y", "1d", force_latest=force_latest_candle)
                df_4h = load_data_cached(ticker, "60d", "4h")
                
                if df_1w.empty or df_1d.empty or df_4h.empty:
                    st.error("Bazı zaman dilimleri için veri çekilemedi (Hisse senedi yeni olabilir veya API gecikmesi var).")
                else:
                    t_screen1, t_screen2, t_screen3 = st.tabs(["1. Ekran (Haftalık)", "2. Ekran (Günlük)", "3. Ekran (4 Saatlik)"])
                    
                    with t_screen1:
                        st.subheader("1. Ekran: Haftalık (Ana Trend)")
                        m_line, m_sig, m_hist = macd(df_1w["Close"])
                        
                        ema_1w_13 = ema(df_1w["Close"], 13)
                        ema_1w_26 = ema(df_1w["Close"], 26)
                        
                        last_close_1w = df_1w["Close"].iloc[-1]
                        if ema_1w_13.iloc[-1] > ema_1w_26.iloc[-1] and last_close_1w > ema_1w_13.iloc[-1]:
                            ema1w_sig = "AL"
                        elif ema_1w_13.iloc[-1] < ema_1w_26.iloc[-1] and last_close_1w < ema_1w_13.iloc[-1]:
                            ema1w_sig = "SAT"
                        else:
                            ema1w_sig = "BEKLE"
                        
                        last_hist = float(m_hist.iloc[-1])
                        prev_hist = float(m_hist.iloc[-2])
                        slope_up = last_hist > prev_hist
                        
                        div_macd = check_bullish_divergence(df_1w["Close"], m_hist)

                        adx_1w, pdi_1w, mdi_1w = adx_indicator(df_1w["High"], df_1w["Low"], df_1w["Close"])
                        adx_val_1w = adx_1w.iloc[-1]
                        pdi_val_1w = pdi_1w.iloc[-1]
                        mdi_val_1w = mdi_1w.iloc[-1]
                        
                        if adx_val_1w >= 25 and pdi_val_1w > mdi_val_1w:
                            adx_sig_1w = "AL (Güçlü Trend)"
                        elif adx_val_1w >= 25 and mdi_val_1w > pdi_val_1w:
                            adx_sig_1w = "SAT (Güçlü Trend)"
                        else:
                            adx_sig_1w = "BEKLE (Zayıf Trend)"
                        
                        c1w_1, c1w_2, c1w_3 = st.columns(3)
                        c1w_1.metric(
                            "MACD Histogram Eğimi", 
                            "YUKARI (AL Sinyali)" if slope_up else "AŞAĞI (SAT Sinyali)", 
                            f"{last_hist - prev_hist:.2f}"
                        )
                        c1w_2.metric(
                            "Haftalık EMA (13-26)",
                            ema1w_sig,
                            f"EMA13: {ema_1w_13.iloc[-1]:.2f} | EMA26: {ema_1w_26.iloc[-1]:.2f}"
                        )
                        c1w_3.metric(
                            "ADX (14)", 
                            adx_sig_1w, 
                            f"ADX: {adx_val_1w:.1f} | +DI: {pdi_val_1w:.1f} | -DI: {mdi_val_1w:.1f}"
                        )
                        
                        if div_macd:
                            st.success("🚀 Sistem Haftalık MACD Histogramında **Pozitif Uyumsuzluk** tespit etti!")
                            
                        fig1_price = go.Figure()
                        fig1_price.add_trace(go.Candlestick(x=df_1w.index, open=df_1w["Open"], high=df_1w["High"], low=df_1w["Low"], close=df_1w["Close"], name="Fiyat"))
                        fig1_price.add_trace(go.Scatter(x=df_1w.index, y=ema_1w_13, name="EMA 13", line=dict(color='blue')))
                        fig1_price.add_trace(go.Scatter(x=df_1w.index, y=ema_1w_26, name="EMA 26", line=dict(color='red')))
                        fig1_price.update_layout(title="Haftalık Fiyat ve EMA (13 & 26)", height=350, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig1_price, use_container_width=True)

                        fig1 = go.Figure()
                        colors = ['green' if x > 0 else 'red' for x in m_hist.diff()]
                        fig1.add_trace(go.Bar(x=df_1w.index, y=m_hist, name="MACD Hist", marker_color=colors))
                        fig1.update_layout(title="Haftalık MACD Histogramı", height=250)
                        st.plotly_chart(fig1, use_container_width=True)

                        fig1_adx = go.Figure()
                        fig1_adx.add_trace(go.Scatter(x=df_1w.index, y=adx_1w, name="ADX", line=dict(color='black', width=2.5)))
                        fig1_adx.add_trace(go.Scatter(x=df_1w.index, y=pdi_1w, name="+DI", line=dict(color='green')))
                        fig1_adx.add_trace(go.Scatter(x=df_1w.index, y=mdi_1w, name="-DI", line=dict(color='red')))
                        fig1_adx.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Trend Başlangıcı (25)")
                        fig1_adx.add_hline(y=50, line_dash="dot", line_color="purple", annotation_text="Aşırı Güçlü Trend (50)")
                        fig1_adx.add_hrect(y0=25, y1=100, fillcolor="rgba(0, 255, 0, 0.05)", layer="below", line_width=0)
                        fig1_adx.add_hrect(y0=0, y1=25, fillcolor="rgba(255, 0, 0, 0.05)", layer="below", line_width=0)
                        fig1_adx.update_layout(title="Haftalık ADX ve Yön Göstergeleri (+DI / -DI)", height=250)
                        st.plotly_chart(fig1_adx, use_container_width=True)

                    with t_screen2:
                        st.subheader("2. Ekran: Günlük (Osilatörler ve Sapmalar)")
                        
                        ema_1d_11 = ema(df_1d["Close"], 11)
                        ema_1d_22 = ema(df_1d["Close"], 22)
                        
                        last_close_1d = df_1d["Close"].iloc[-1]
                        if ema_1d_11.iloc[-1] > ema_1d_22.iloc[-1] and last_close_1d > ema_1d_11.iloc[-1]:
                            ema1d_sig = "AL"
                        elif ema_1d_11.iloc[-1] < ema_1d_22.iloc[-1] and last_close_1d < ema_1d_11.iloc[-1]:
                            ema1d_sig = "SAT"
                        else:
                            ema1d_sig = "BEKLE"
                        
                        st.metric("Günlük EMA (11-22)", ema1d_sig, f"EMA11: {ema_1d_11.iloc[-1]:.2f} | EMA22: {ema_1d_22.iloc[-1]:.2f}")

                        fi = force_index(df_1d["Close"], df_1d["Volume"])
                        fi_ema13 = ema(fi, 13)
                        fi_ema2 = ema(fi, 2)
                        
                        rsi13 = rsi(df_1d["Close"], 13)
                        
                        stoch_k, stoch_d = stochastic(df_1d["High"], df_1d["Low"], df_1d["Close"], k_period=5, d_period=3)
                        
                        er_ema, bull_p, bear_p = elder_ray(df_1d["High"], df_1d["Low"], df_1d["Close"], 13)
                        
                        fi_al = (fi.iloc[-1] > fi_ema13.iloc[-1]) and (fi_ema2.iloc[-1] < 0)
                        rsi_al = (rsi13.iloc[-1] < 30)
                        stoch_al = (stoch_k.iloc[-1] < 20)
                        
                        er_ema_up = (er_ema.iloc[-1] > er_ema.iloc[-2])
                        bp_neg_but_rising = (bear_p.iloc[-1] < 0) and (bear_p.iloc[-1] > bear_p.iloc[-2])
                        er_al = er_ema_up and bp_neg_but_rising
                        
                        div_rsi = check_bullish_divergence(df_1d["Close"], rsi13)
                        div_stoch = check_bullish_divergence(df_1d["Close"], stoch_k)
                        div_er = check_bullish_divergence(df_1d["Close"], bear_p)
                        
                        adx_1d, pdi_1d, mdi_1d = adx_indicator(df_1d["High"], df_1d["Low"], df_1d["Close"])
                        adx_val_1d = adx_1d.iloc[-1]
                        pdi_val_1d = pdi_1d.iloc[-1]
                        mdi_val_1d = mdi_1d.iloc[-1]
                        
                        if adx_val_1d >= 25 and pdi_val_1d > mdi_val_1d:
                            adx_sig_1d = "AL (Güçlü Trend)"
                        elif adx_val_1d >= 25 and mdi_val_1d > pdi_val_1d:
                            adx_sig_1d = "SAT (Güçlü Trend)"
                        else:
                            adx_sig_1d = "BEKLE (Zayıf Trend)"

                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Kuvvet Endeksi (FI)", "AL" if fi_al else "BEKLE", "13 EMA Üstü & 2 EMA Negatif" if fi_al else "")
                        c2.metric("RSI (13)", "AL" if rsi_al else "BEKLE", f"{rsi13.iloc[-1]:.1f}")
                        c3.metric("Stokastik (5)", "AL" if stoch_al else "BEKLE", f"{stoch_k.iloc[-1]:.1f}")
                        c4.metric("Elder-Ray", "AL" if er_al else "BEKLE")
                        c5.metric("ADX (14)", adx_sig_1d, f"ADX: {adx_val_1d:.1f} | +DI: {pdi_val_1d:.1f}")
                        
                        if div_rsi: st.success("🚀 RSI(13)'te **Pozitif Uyumsuzluk** tespit edildi!")
                        if div_stoch: st.success("🚀 Stokastik(5)'te **Pozitif Uyumsuzluk** tespit edildi!")
                        if div_er: st.success("🚀 Elder-Ray Bear Power'da **Pozitif Uyumsuzluk (Boğa Uyumsuzluğu)** tespit edildi!")
                        
                        if st.session_state.sentiment_summary:
                            st.info(f"**Haber Etkisi Modülü:** {st.session_state.sentiment_summary}")
                            
                        fig2_price = go.Figure()
                        fig2_price.add_trace(go.Candlestick(x=df_1d.index, open=df_1d["Open"], high=df_1d["High"], low=df_1d["Low"], close=df_1d["Close"], name="Fiyat"))
                        fig2_price.add_trace(go.Scatter(x=df_1d.index, y=ema_1d_11, name="EMA 11", line=dict(color='blue')))
                        fig2_price.add_trace(go.Scatter(x=df_1d.index, y=ema_1d_22, name="EMA 22", line=dict(color='red')))
                        fig2_price.update_layout(title="Günlük Fiyat ve EMA (11 & 22)", height=350, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig2_price, use_container_width=True)
                        
                        fig2_fi = go.Figure()
                        fig2_fi.add_trace(go.Scatter(x=df_1d.index, y=fi_ema13, name="FI 13 EMA", line=dict(color='orange')))
                        fig2_fi.add_trace(go.Bar(x=df_1d.index, y=fi_ema2, name="FI 2 EMA", marker_color='gray'))
                        fig2_fi.update_layout(title="Kuvvet Endeksi (Force Index)", height=250)
                        st.plotly_chart(fig2_fi, use_container_width=True)
                        
                        fig2_er = go.Figure()
                        fig2_er.add_trace(go.Bar(x=df_1d.index, y=bull_p, name="Bull Power", marker_color='green'))
                        fig2_er.add_trace(go.Bar(x=df_1d.index, y=bear_p, name="Bear Power", marker_color='red'))
                        fig2_er.update_layout(title="Elder-Ray (Bull & Bear Power)", height=250)
                        st.plotly_chart(fig2_er, use_container_width=True)

                        fig2_adx = go.Figure()
                        fig2_adx.add_trace(go.Scatter(x=df_1d.index, y=adx_1d, name="ADX", line=dict(color='black', width=2.5)))
                        fig2_adx.add_trace(go.Scatter(x=df_1d.index, y=pdi_1d, name="+DI", line=dict(color='green')))
                        fig2_adx.add_trace(go.Scatter(x=df_1d.index, y=mdi_1d, name="-DI", line=dict(color='red')))
                        
                        fig2_adx.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Trend Başlangıcı (25)")
                        fig2_adx.add_hline(y=50, line_dash="dot", line_color="purple", annotation_text="Aşırı Güçlü Trend (50)")
                        fig2_adx.add_hrect(y0=25, y1=100, fillcolor="rgba(0, 255, 0, 0.05)", layer="below", line_width=0)
                        fig2_adx.add_hrect(y0=0, y1=25, fillcolor="rgba(255, 0, 0, 0.05)", layer="below", line_width=0)
                        
                        fig2_adx.update_layout(title="Günlük ADX ve Yön Göstergeleri (+DI / -DI)", height=250)
                        st.plotly_chart(fig2_adx, use_container_width=True)

                    with t_screen3:
                        st.subheader("3. Ekran: 4 Saatlik (Giriş / Çıkış ve Hedefler)")
                        
                        adx_4h, pdi_4h, mdi_4h = adx_indicator(df_4h["High"], df_4h["Low"], df_4h["Close"])
                        adx_val_4h = adx_4h.iloc[-1]
                        pdi_val_4h = pdi_4h.iloc[-1]
                        mdi_val_4h = mdi_4h.iloc[-1]
                        
                        if adx_val_4h >= 25 and pdi_val_4h > mdi_val_4h:
                            adx_sig_4h = "AL (Güçlü Trend)"
                        elif adx_val_4h >= 25 and mdi_val_4h > pdi_val_4h:
                            adx_sig_4h = "SAT (Güçlü Trend)"
                        else:
                            adx_sig_4h = "BEKLE (Zayıf Trend)"
                        
                        st.metric("4 Saatlik ADX (14)", adx_sig_4h, f"ADX: {adx_val_4h:.1f} | +DI: {pdi_val_4h:.1f} | -DI: {mdi_val_4h:.1f}")

                        ema_4h = ema(df_4h["Close"], 13)
                        atr_4h = atr(df_4h["High"], df_4h["Low"], df_4h["Close"], 14)
                        last_atr_4h = float(atr_4h.iloc[-1]) if not pd.isna(atr_4h.iloc[-1]) else 0.0
                        
                        pens = ema_4h - df_4h["Low"]
                        pens_positive = pens[pens > 0]
                        avg_pen = float(pens_positive.mean()) if not pens_positive.empty else 0.0
                        
                        up_pens = df_4h["High"] - ema_4h
                        up_pens_positive = up_pens[up_pens > 0]
                        avg_up_pen = float(up_pens_positive.mean()) if not up_pens_positive.empty else 0.0
                        
                        ema_today = float(ema_4h.iloc[-1])
                        ema_yest = float(ema_4h.iloc[-2])
                        ema_delta = ema_today - ema_yest
                        ema_est_tmrw = ema_today + ema_delta
                        
                        buy_level = ema_est_tmrw - avg_pen
                        
                        stop_loss = buy_level - (1.5 * last_atr_4h) if last_atr_4h > 0 else buy_level * 0.98
                        risk = buy_level - stop_loss
                        
                        target_1 = ema_est_tmrw + avg_up_pen
                        target_2 = buy_level + (risk * 2)
                        
                        st.markdown(f"""
                        **Hesaplamalar ve Strateji (Buy Limit & Hedefler):**
                        * 📌 **Güncel EMA (13):** {ema_today:.2f} | **Yarınki Tahmini EMA:** {ema_est_tmrw:.2f}
                        * 🟢 **Önerilen Alış Seviyesi (Buy Limit): {buy_level:.2f}** *(Ortalama {avg_pen:.2f} düşüş penetrasyonu ile)*
                        * 🔴 **Zarar Kes (Stop-Loss): {stop_loss:.2f}** *(Alışın 1.5 ATR altı. Risk: {risk:.2f})*
                        * 🎯 **Hedef 1 (Kısa Vade): {target_1:.2f}** *(Simetrik Yükseliş Penetrasyonu)*
                        * 🚀 **Hedef 2 (Trend - 1:2 RR): {target_2:.2f}** *(Riske edilen tutarın 2 katı kazanç)*
                        """)
                        
                        fig3 = go.Figure()
                        fig3.add_trace(go.Candlestick(x=df_4h.index, open=df_4h["Open"], high=df_4h["High"], low=df_4h["Low"], close=df_4h["Close"], name="Price"))
                        fig3.add_trace(go.Scatter(x=df_4h.index, y=ema_4h, name="EMA 13", line=dict(color='blue')))
                        
                        last_time = df_4h.index[-1]
                        next_time = last_time + pd.Timedelta(hours=4)
                        fig3.add_trace(go.Scatter(x=[next_time], y=[ema_est_tmrw], mode='markers', marker=dict(size=10, color='orange'), name="Tahmini EMA"))
                        
                        fig3.add_hline(y=target_2, line_dash="dash", line_color="darkgreen", annotation_text="Hedef 2 (1:2 RR)", annotation_position="top left")
                        fig3.add_hline(y=target_1, line_dash="dashdot", line_color="cyan", annotation_text="Hedef 1 (Simetrik)", annotation_position="top left")
                        fig3.add_hline(y=buy_level, line_dash="dash", line_color="lime", annotation_text="Limit Alış Seviyesi", annotation_position="bottom left")
                        fig3.add_hline(y=stop_loss, line_dash="dot", line_color="red", annotation_text="Stop-Loss (1.5 ATR)", annotation_position="bottom left")
                        
                        fig3.update_layout(title="4 Saatlik Giriş/Çıkış Stratejisi (Alış, Hedef ve Stop)", height=450, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig3, use_container_width=True)

                        fig3_adx = go.Figure()
                        fig3_adx.add_trace(go.Scatter(x=df_4h.index, y=adx_4h, name="ADX", line=dict(color='black', width=2.5)))
                        fig3_adx.add_trace(go.Scatter(x=df_4h.index, y=pdi_4h, name="+DI", line=dict(color='green')))
                        fig3_adx.add_trace(go.Scatter(x=df_4h.index, y=mdi_4h, name="-DI", line=dict(color='red')))
                        
                        fig3_adx.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Trend Başlangıcı (25)")
                        fig3_adx.add_hline(y=50, line_dash="dot", line_color="purple", annotation_text="Aşırı Güçlü Trend (50)")
                        fig3_adx.add_hrect(y0=25, y1=100, fillcolor="rgba(0, 255, 0, 0.05)", layer="below", line_width=0)
                        fig3_adx.add_hrect(y0=0, y1=25, fillcolor="rgba(255, 0, 0, 0.05)", layer="below", line_width=0)
                        
                        fig3_adx.update_layout(title="4 Saatlik ADX ve Yön Göstergeleri (+DI / -DI)", height=250)
                        st.plotly_chart(fig3_adx, use_container_width=True)
