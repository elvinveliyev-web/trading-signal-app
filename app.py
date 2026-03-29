import os
import re
import json
import time
import base64
import importlib
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests

# ============================================================
# OPSİYONEL BAĞIMLILIKLAR
# ============================================================
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

try:
    import feedparser
    FEEDPARSER_OK = True
except ImportError:
    FEEDPARSER_OK = False

try:
    import kaleido  # noqa: F401
    KALEIDO_OK = True
except ImportError:
    KALEIDO_OK = False

# ============================================================
# SAYFA YAPIKLANDIRMASI
# ============================================================
st.set_page_config(
    page_title="FA→TA Trading + AI",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "FA→TA Trading Uygulaması — Temel + Teknik Analiz + Gemini AI",
    },
)

# ============================================================
# ÖZEL CSS — Geliştirilmiş UI/UX
# ============================================================
st.markdown("""
<style>
/* ── Ana Yazı Tipi ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Arkaplan ── */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ── Kenar Çubuğu ── */
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #58a6ff;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 1.2rem;
}

/* ── Metrik Kartları ── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 16px;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: #58a6ff;
}
[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-size: 0.72rem !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    color: #e6edf3 !important;
    font-size: 1.35rem !important;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Sekmeler ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #21262d;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.875rem;
    padding: 8px 18px;
    transition: all 0.2s;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #21262d !important;
    color: #58a6ff !important;
}

/* ── Butonlar ── */
.stButton > button {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.875rem;
    padding: 10px 20px;
    transition: all 0.2s;
    font-family: 'IBM Plex Sans', sans-serif;
}
.stButton > button:hover {
    background: #388bfd;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(31,111,235,0.4);
}
.stButton > button[kind="secondary"] {
    background: #21262d;
    color: #58a6ff;
    border: 1px solid #30363d;
}
.stButton > button[kind="secondary"]:hover {
    background: #30363d;
    border-color: #58a6ff;
}

/* ── Veri Tablosu ── */
[data-testid="stDataFrame"] {
    border: 1px solid #21262d;
    border-radius: 10px;
    overflow: hidden;
}

/* ── Genişletilebilir Bölümler ── */
[data-testid="stExpander"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
}
[data-testid="stExpander"] summary {
    color: #8b949e;
    font-weight: 500;
}
[data-testid="stExpander"] summary:hover {
    color: #58a6ff;
}

/* ── Bilgi Kutuları ── */
.stAlert {
    border-radius: 10px;
    border-left-width: 4px;
}

/* ── Kart Bileşeni ── */
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 8px;
}
.metric-card:hover {
    border-color: #58a6ff;
}

/* ── Sinyal Rozeti ── */
.badge-al {
    background: #1a7f37;
    color: #3fb950;
    border: 1px solid #2ea043;
    border-radius: 20px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 0.85rem;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-sat {
    background: #6e1a1a;
    color: #f85149;
    border: 1px solid #da3633;
    border-radius: 20px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 0.85rem;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-izle {
    background: #3d2e00;
    color: #e3b341;
    border: 1px solid #9e6a03;
    border-radius: 20px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 0.85rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── İlerleme Çubuğu ── */
.score-bar-container {
    background: #21262d;
    border-radius: 6px;
    height: 8px;
    margin-top: 4px;
    overflow: hidden;
}
.score-bar-fill-green {
    background: linear-gradient(90deg, #238636, #3fb950);
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}
.score-bar-fill-red {
    background: linear-gradient(90deg, #b91c1c, #f85149);
    height: 100%;
    border-radius: 6px;
}
.score-bar-fill-yellow {
    background: linear-gradient(90deg, #9e6a03, #e3b341);
    height: 100%;
    border-radius: 6px;
}
.score-bar-fill-blue {
    background: linear-gradient(90deg, #1f6feb, #58a6ff);
    height: 100%;
    border-radius: 6px;
}

/* ── Bölüm Başlıkları ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #21262d;
}
.section-header h2 {
    font-size: 1rem;
    font-weight: 600;
    color: #e6edf3;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Sohbet Arayüzü ── */
.chat-message-user {
    background: #1f6feb22;
    border: 1px solid #1f6feb44;
    border-radius: 12px 12px 2px 12px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
    color: #cdd9e5;
}
.chat-message-ai {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px 12px 12px 2px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
    color: #e6edf3;
}

/* ── Kod Blokları ── */
code {
    font-family: 'IBM Plex Mono', monospace;
    background: #161b22;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85em;
    color: #79c0ff;
}

/* ── Divider ── */
hr {
    border-color: #21262d;
    margin: 12px 0;
}

/* ── Input alanları ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div,
.stNumberInput > div > div > input {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: #1f6feb !important;
}

/* ── Checkbox ── */
.stCheckbox > label {
    color: #cdd9e5;
    font-size: 0.875rem;
}

/* ── Başlık ── */
h1.app-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.02em;
}
.app-subtitle {
    color: #8b949e;
    font-size: 0.875rem;
    margin-top: -8px;
}

/* ── Plotly grafik arka planı ── */
.js-plotly-plot .plotly .main-svg {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# TEMEL DİZİN
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

def pjoin(*parts) -> str:
    return os.path.join(BASE_DIR, *parts)

# ============================================================
# YARDIMCI ARAÇLAR
# ============================================================
FUND_KEYS_FLOAT = [
    "marketCap", "trailingPE", "forwardPE", "pegRatio",
    "priceToSalesTrailing12Months", "priceToBook",
    "returnOnEquity", "profitMargins", "operatingMargins",
    "debtToEquity", "revenueGrowth", "earningsGrowth",
    "freeCashflow", "currentPrice",
]

def safe_float(x) -> float:
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            v = float(x)
            return np.nan if not np.isfinite(v) else v
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan

def normalize_ticker(raw: str, market: str) -> str:
    t = (raw or "").strip().upper()
    if not t:
        return t
    if market == "BIST" and not t.endswith(".IS"):
        t = f"{t}.IS"
    return t

def naked_ticker(raw: str) -> str:
    return (raw or "").strip().upper().replace(".IS", "")

def fmt_pct(x: float) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "N/A"
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "N/A"

def fmt_num(x, nd: int = 2) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return "N/A"
        return f"{v:.{nd}f}"
    except Exception:
        return "N/A"

def fmt_rr(rr) -> str:
    try:
        v = float(rr)
        if not np.isfinite(v):
            return "N/A"
        return f"1:{v:.2f}"
    except Exception:
        return "N/A"

def pct_dist(level: float, base: float) -> Optional[float]:
    try:
        if not np.isfinite(level) or base == 0:
            return None
        return (level / base - 1.0) * 100.0
    except Exception:
        return None

def _fix_debt_to_equity(x: float) -> float:
    if pd.notna(x) and x > 10:
        return x / 100.0
    return x

def _flatten_yf(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """MultiIndex sütunlarını güvenli şekilde düzleştirir."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if ticker:
            # Çok hisseli indirme: sadece bu tickerın sütunlarını al
            try:
                out = out.xs(ticker, axis=1, level=1)
            except Exception:
                out.columns = [c[0] for c in out.columns]
        else:
            out.columns = [c[0] for c in out.columns]
    required = [c for c in ["Open", "High", "Low", "Close"] if c in out.columns]
    if required:
        out = out.dropna(subset=required)
    return out

# ============================================================
# TEKNİK GÖSTERGELER
# ============================================================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(alpha=1/period, adjust=False).mean()
    roll_dn = pd.Series(down, index=close.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).replace([np.inf, -np.inf], np.nan).fillna(50)

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig

def bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return mid, mid + std_mult * sd, mid - std_mult * sd

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1/period, adjust=False).mean()

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()

def max_drawdown(eq: pd.Series) -> float:
    if eq is None or len(eq) == 0:
        return 0.0
    peak = eq.cummax()
    return float(((eq / peak) - 1.0).min())

def stoch_rsi(rsi_series: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    min_r = rsi_series.rolling(period).min()
    max_r = rsi_series.rolling(period).max()
    den = (max_r - min_r).replace(0, np.nan)
    stoch = (100 * (rsi_series - min_r) / den).replace([np.inf, -np.inf], np.nan).fillna(50)
    k = stoch.rolling(smooth_k).mean()
    return k, k.rolling(smooth_d).mean()

# ============================================================
# AŞIRI ALIM / SPEKÜLASYON
# ============================================================
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

    df["STOCH_RSI_K"], df["STOCH_RSI_D"] = stoch_rsi(df["RSI"])
    df["STOCH_OVERBOUGHT"] = (df["STOCH_RSI_K"] > 80).astype(int)

    df["VOLUME_DIR"] = np.sign(df["Volume"].diff()).fillna(0)
    df["PRICE_DIR"] = np.sign(df["Close"].diff()).fillna(0)
    df["WEAK_UPTREND"] = ((df["PRICE_DIR"] > 0) & (df["VOLUME_DIR"] < 0)).astype(int)
    return df


def detect_speculation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Korelasyon düzeltmeli ağırlıklandırma:
    RSI, BB ve Stochastic RSI birbirine yüksek koreleli sinyallerdir.
    Bu grup tek bir 'momentum aşırılık' faktörü olarak ele alınır.
    Hacim ve fiyat-EMA uzaklığı bağımsız sinyallerdir.
    """
    last = df.iloc[-1]
    details = {}

    # ── Momentum faktörü (maks 1 tam puan, koreleli 3 sinyalin ortalaması)
    momentum_signals = 0
    momentum_count = 0

    ob_score = 0
    os_score = 0
    spec_score = 0

    # RSI
    if last["RSI"] > 70:
        momentum_signals += 1
        details["rsi"] = f"Aşırı alım — RSI: {last['RSI']:.1f}"
    elif last["RSI"] < 30:
        os_score += 30
        details["rsi"] = f"Aşırı satım — RSI: {last['RSI']:.1f}"
    momentum_count += 1

    # Bollinger
    if bool(last["BB_OVERBOUGHT"]):
        momentum_signals += 1
        details["bb"] = "Fiyat Bollinger üst bandının üzerinde"
    elif bool(last["BB_OVERSOLD"]):
        os_score += 20
        details["bb"] = "Fiyat Bollinger alt bandının altında"
    momentum_count += 1

    # Stochastic RSI
    if bool(last["STOCH_OVERBOUGHT"]):
        momentum_signals += 1
        details["stoch"] = "Stochastic RSI aşırı alım bölgesinde (>80)"
    momentum_count += 1

    # Momentum faktörünü 0-35 puan aralığına normalize et
    if momentum_count > 0:
        ob_score += int((momentum_signals / momentum_count) * 35)

    # ── Bağımsız sinyaller
    if bool(last["VOLUME_SPIKE"]):
        spec_score += 35
        details["volume"] = "Ani hacim artışı (normalin 1.5 katı üzerinde)"

    if bool(last["PRICE_EXTREME"]):
        ob_score += 20
        details["price_extreme"] = (
            f"Fiyat EMA'lardan çok uzak "
            f"(EMA50'den %{last['PRICE_TO_EMA50']:.1f}, EMA200'den %{last['PRICE_TO_EMA200']:.1f})"
        )

    if bool(last["WEAK_UPTREND"]):
        spec_score += 25
        details["weak_trend"] = "Fiyat yükselirken hacim düşüyor (momentum zayıflıyor)"

    # ── Sınırla
    ob_score = min(100, ob_score)
    os_score = min(100, os_score)
    spec_score = min(100, spec_score)

    if ob_score >= 50:
        verdict = "AŞIRI DEĞERLİ — SAT bölgesi ⚠️"
    elif os_score >= 50:
        verdict = "AŞIRI DEĞERSİZ — AL fırsatı 🟢"
    elif spec_score >= 50:
        verdict = "SPEKÜLATİF HAREKET — Dikkatli ol ⚡"
    else:
        verdict = "NÖTR — Normal değer aralığı ✅"

    return {
        "overbought_score": ob_score,
        "oversold_score": os_score,
        "speculation_score": spec_score,
        "verdict": verdict,
        "details": details,
    }

# ============================================================
# GÖSTERGE İNŞA EDİCİ
# ============================================================
def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["EMA50"] = ema(df["Close"], int(cfg["ema_fast"]))
    df["EMA200"] = ema(df["Close"], int(cfg["ema_slow"]))
    df["RSI"] = rsi(df["Close"], int(cfg["rsi_period"]))
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"], 12, 26, 9)
    df["BB_mid"], df["BB_upper"], df["BB_lower"] = bollinger(
        df["Close"], int(cfg["bb_period"]), float(cfg["bb_std"])
    )
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
    return df

# ============================================================
# PİYASA REJİMİ FİLTRELERİ
# ============================================================
@st.cache_data(ttl=6*3600, show_spinner=False)
def get_spy_regime_ok() -> bool:
    spy = yf.download("SPY", period="2y", interval="1d", auto_adjust=True, progress=False)
    spy = _flatten_yf(spy)
    if spy.empty or len(spy) < 200:
        return True
    spy["EMA200"] = ema(spy["Close"], 200)
    last = spy.iloc[-1]
    return bool(last["Close"] > last["EMA200"])

@st.cache_data(ttl=6*3600, show_spinner=False)
def get_bist_regime_ok() -> bool:
    xu = yf.download("XU100.IS", period="2y", interval="1d", auto_adjust=True, progress=False)
    xu = _flatten_yf(xu)
    if xu.empty or len(xu) < 200:
        return True
    xu["EMA200"] = ema(xu["Close"], 200)
    last = xu.iloc[-1]
    return bool(last["Close"] > last["EMA200"])

@st.cache_data(ttl=6*3600, show_spinner=False)
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

# ============================================================
# STRATEJİ: PUANLAMA + KONTROL NOKTALARI
# ============================================================
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
        "Piyasa Rejim Filtresi": bool(market_filter_ok),
        "Üst Zaman Dilimi Trendi": bool(higher_tf_filter_ok),
        "Likidite (Hacim > HacimSMA)": bool(last["Volume"] > last["VOL_SMA"]) if pd.notna(last["VOL_SMA"]) else False,
        "Trend (Kapanış>EMA200 & EMA50>EMA200)": bool((last["Close"] > last["EMA200"]) and (last["EMA50"] > last["EMA200"])) if pd.notna(last["EMA200"]) else False,
        f"RSI > {cfg['rsi_entry_level']}": bool(last["RSI"] > cfg["rsi_entry_level"]) if pd.notna(last["RSI"]) else False,
        "MACD Hist > 0": bool(last["MACD_hist"] > 0) if pd.notna(last["MACD_hist"]) else False,
        f"ATR% < {cfg['atr_pct_max']:.2%}": bool((last["ATR"] / last["Close"]) < cfg["atr_pct_max"]) if pd.notna(last["ATR"]) and pd.notna(last["Close"]) else False,
        "Bollinger (Kapanış>BB_orta veya Kırılım)": bool((last["Close"] > last["BB_mid"]) or (last["Close"] > last["BB_upper"])) if pd.notna(last["BB_mid"]) else False,
        "OBV > OBV_EMA": bool(last["OBV"] > last["OBV_EMA"]) if pd.notna(last["OBV_EMA"]) else False,
    }
    return df, cp

# ============================================================
# GERİ TEST
# ============================================================
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

        # Dinamik trailing stop güncelleme
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

            # İlk hedefte yarı çıkış
            if target_hit:
                sell_shares = shares * 0.5
                sell_price = price * (1 - slippage)
                gross = sell_shares * sell_price
                cash += gross - (gross * commission)
                shares -= sell_shares
                half_sold = True
                stop = max(stop, entry_price)
                if trades:
                    trades[-1]["pnl"] = cash + (shares * price * (1 - slippage)) - trades[-1]["equity_before"]

            # Çıkış koşulları
            if exit_sig.iloc[i] == 1 or stop_hit or time_stop_hit:
                sell_price = price * (1 - slippage)
                gross = shares * sell_price
                cash += gross - (gross * commission)

                if trades:
                    trades[-1]["exit_date"] = date
                    trades[-1]["exit_price"] = sell_price
                    trades[-1]["exit_reason"] = "STOP" if stop_hit else ("ZAMAN_STOPU" if time_stop_hit else "KURAL_ÇIKIŞ")
                    trades[-1]["pnl"] = cash - trades[-1]["equity_before"]

                shares = 0.0
                stop = np.nan
                entry_price = np.nan
                target_price = np.nan
                bars_held = 0
                half_sold = False

        position_value = shares * price * (1 - slippage)
        equity = cash + position_value

        # Giriş koşulları
        if shares == 0 and entry_sig.iloc[i] == 1 and pd.notna(row["ATR"]) and row["ATR"] > 0:
            risk_cash = equity * cfg["risk_per_trade"]
            stop_dist = cfg["atr_stop_mult"] * float(row["ATR"])

            if stop_dist > 0:
                qty = risk_cash / stop_dist
                buy_price = price * (1 + slippage)
                total_cost = qty * buy_price * (1 + commission)

                if total_cost <= cash:
                    cash -= total_cost
                    shares = qty
                    entry_price = buy_price
                    stop = buy_price - cfg["atr_stop_mult"] * float(row["ATR"])
                    target_price = buy_price + (tp_mult * cfg["atr_stop_mult"] * float(row["ATR"]))
                    bars_held = 0
                    half_sold = False
                    trades.append({
                        "entry_date": date,
                        "entry_price": buy_price,
                        "exit_date": None,
                        "exit_price": None,
                        "exit_reason": None,
                        "shares": shares,
                        "equity_before": equity,
                        "pnl": None,
                    })

        equity_curve.append((date, cash + shares * price * (1 - slippage)))

    eq = pd.Series(
        [v for _, v in equity_curve],
        index=[d for d, _ in equity_curve],
        name="equity",
    ).astype(float).replace([np.inf, -np.inf], np.nan).dropna()

    ret = eq.pct_change().dropna()
    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    ann_return = (1 + total_return) ** (252 / max(1, len(ret))) - 1 if len(ret) > 0 else 0.0
    ann_vol = float(ret.std() * np.sqrt(252)) if len(ret) > 1 else 0.0

    rf_daily = (1 + float(risk_free_annual)) ** (1/252) - 1
    excess = ret - rf_daily
    sharpe = float((excess.mean() * 252) / (excess.std() * np.sqrt(252))) if len(ret) > 1 and excess.std() > 0 else 0.0
    downside = excess.copy()
    downside[downside > 0] = 0
    dd_dev = float(np.sqrt((downside**2).mean()) * np.sqrt(252)) if len(downside) > 1 else 0.0
    sortino = float((excess.mean() * 252) / dd_dev) if dd_dev > 0 else 0.0
    mdd = max_drawdown(eq)
    calmar = float(ann_return / abs(mdd)) if mdd < 0 else 0.0

    beta, alpha, info_ratio = 1.0, 0.0, 0.0
    if benchmark_returns is not None:
        common = ret.index.intersection(benchmark_returns.index)
        if len(common) > 5:
            r_a = ret.loc[common]
            b_a = benchmark_returns.loc[common]
            cov = np.cov(r_a, b_a)[0, 1]
            var_b = np.var(b_a)
            beta = cov / var_b if var_b != 0 else 1.0
            alpha = (r_a.mean() * 252 - risk_free_annual) - beta * (b_a.mean() * 252 - risk_free_annual)
            diff = r_a - b_a
            info_ratio = (diff.mean() * 252) / (diff.std() * np.sqrt(252)) if diff.std() > 0 else 0.0

    peak = eq.cummax()
    ulcer_index = float(np.sqrt((((eq - peak) / peak)**2).mean())) if len(eq) > 0 else 0.0

    tdf = pd.DataFrame(trades)
    profit_factor = 0.0
    kelly = 0.0

    if not tdf.empty:
        tdf["pnl"] = tdf["pnl"].astype(float)
        tdf["getiri_%"] = (tdf["pnl"] / tdf["equity_before"]) * 100
        tdf["tutulma_gunu"] = (
            pd.to_datetime(tdf["exit_date"]) - pd.to_datetime(tdf["entry_date"])
        ).dt.days

        gross_profit = float(tdf.loc[tdf["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-tdf.loc[tdf["pnl"] < 0, "pnl"].sum())
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")

        if len(tdf) > 5:
            win_rate = (tdf["pnl"] > 0).mean()
            avg_win = tdf.loc[tdf["pnl"] > 0, "pnl"].mean() if win_rate > 0 else 0
            avg_loss = -tdf.loc[tdf["pnl"] < 0, "pnl"].mean() if win_rate < 1 else 0
            if avg_loss > 0 and 0 < win_rate < 1:
                b = avg_win / avg_loss
                kelly_full = (win_rate * b - (1 - win_rate)) / b
                # Güvenli: tam Kelly'nin yarısı, maks %12.5
                kelly = max(0.0, min(kelly_full * 0.5, 0.125))

    metrics = {
        "Toplam Getiri": float(total_return),
        "Yıllık Getiri": float(ann_return),
        "Yıllık Volatilite": float(ann_vol),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Calmar": float(calmar),
        "Maks Drawdown": float(mdd),
        "İşlem Sayısı": int(len(tdf)) if not tdf.empty else 0,
        "Kazanma Oranı": float((tdf["pnl"] > 0).mean()) if not tdf.empty else 0.0,
        "Kar Faktörü": float(profit_factor) if np.isfinite(profit_factor) else 999.0,
        "Beta": float(beta),
        "Alpha": float(alpha),
        "Bilgi Oranı": float(info_ratio),
        "Ulcer Index": float(ulcer_index),
        "Kelly % (Öneri)": float(kelly * 100),
    }
    return eq, tdf, metrics

# ============================================================
# TEMEL ANALİZ
# ============================================================
@st.cache_data(ttl=12*3600, show_spinner=False)
def fetch_fundamentals_generic(ticker: str, market: str) -> dict:
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    out = {k: safe_float(info.get(k)) for k in FUND_KEYS_FLOAT}
    out.update({
        "ticker": ticker,
        "market": market,
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "longName": info.get("longName", "") or info.get("shortName", ""),
    })
    out["debtToEquity"] = _fix_debt_to_equity(out["debtToEquity"])
    return out


def fundamental_score_row(row: dict, mode: str, thresholds: dict) -> Tuple[float, dict, bool]:
    def _check(name, cond, weight, available):
        return {"ok": bool(cond) if available else False, "weight": weight, "available": bool(available)}

    A = lambda x: pd.notna(x) and np.isfinite(float(x)) if x is not None else False

    checks = {}
    if mode == "Kalite":
        checks = {
            "ROE": _check("ROE", A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"], 20, A(row["returnOnEquity"])),
            "Faaliyet Marjı": _check("Faaliyet Marjı", A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"], 15, A(row["operatingMargins"])),
            "Borç/Özkaynak": _check("Borç/Özkaynak", A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"], 20, A(row["debtToEquity"])),
            "Net Kar Marjı": _check("Net Kar Marjı", A(row["profitMargins"]) and row["profitMargins"] >= thresholds["profit_margin"], 15, A(row["profitMargins"])),
            "Serbest Nakit Akışı": _check("Serbest Nakit Akışı", A(row["freeCashflow"]) and row["freeCashflow"] > 0, 30, A(row["freeCashflow"])),
        }
    elif mode == "Değer":
        checks = {
            "İleri F/K": _check("İleri F/K", A(row["forwardPE"]) and row["forwardPE"] <= thresholds["fpe"], 30, A(row["forwardPE"])),
            "PEG": _check("PEG", A(row["pegRatio"]) and row["pegRatio"] <= thresholds["peg"], 20, A(row["pegRatio"])),
            "F/S": _check("F/S", A(row["priceToSalesTrailing12Months"]) and row["priceToSalesTrailing12Months"] <= thresholds["ps"], 20, A(row["priceToSalesTrailing12Months"])),
            "F/D": _check("F/D", A(row["priceToBook"]) and row["priceToBook"] <= thresholds["pb"], 15, A(row["priceToBook"])),
            "ROE": _check("ROE", A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"], 15, A(row["returnOnEquity"])),
        }
    else:  # Büyüme
        checks = {
            "Gelir Büyümesi": _check("Gelir Büyümesi", A(row["revenueGrowth"]) and row["revenueGrowth"] >= thresholds["rev_g"], 35, A(row["revenueGrowth"])),
            "Kazanç Büyümesi": _check("Kazanç Büyümesi", A(row["earningsGrowth"]) and row["earningsGrowth"] >= thresholds["earn_g"], 35, A(row["earningsGrowth"])),
            "Faaliyet Marjı": _check("Faaliyet Marjı", A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"], 15, A(row["operatingMargins"])),
            "Borç/Özkaynak": _check("Borç/Özkaynak", A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"], 15, A(row["debtToEquity"])),
        }

    total_w = sum(v["weight"] for v in checks.values() if v["available"])
    earned = sum(v["weight"] for v in checks.values() if v["available"] and v["ok"])
    ok_cnt = sum(1 for v in checks.values() if v["available"] and v["ok"])
    avail_cnt = sum(1 for v in checks.values() if v["available"])

    score_pct = (earned / total_w * 100) if total_w > 0 else 0.0
    passed = (
        score_pct >= thresholds["min_score"]
        and ok_cnt >= thresholds["min_ok"]
        and avail_cnt >= thresholds.get("min_coverage", 3)
    )
    return float(score_pct), checks, bool(passed)

# ============================================================
# HEDEF FİYAT BANDI — Yerel Ekstremum Tabanlı Seviyeler
# ============================================================
def _local_extrema_levels(close: pd.Series, high: pd.Series, low: pd.Series, lookback: int = 120) -> List[float]:
    """
    Gerçek destek/direnç seviyeleri için yerel minimum/maksimumları kullanır.
    İstatistiksel yüzdelikler yerine fiyat yapısına dayalıdır.
    """
    n = min(lookback, len(close))
    c = close.tail(n)
    h = high.tail(n)
    l = low.tail(n)

    levels = set()
    window = 5

    # Yerel maksimumlar (dirençler)
    for i in range(window, len(h) - window):
        win = h.iloc[i - window: i + window + 1]
        if h.iloc[i] == win.max():
            levels.add(round(float(h.iloc[i]), 2))

    # Yerel minimumlar (destekler)
    for i in range(window, len(l) - window):
        win = l.iloc[i - window: i + window + 1]
        if l.iloc[i] == win.min():
            levels.add(round(float(l.iloc[i]), 2))

    # Son 20 barlık yüksek/düşük de ekle
    levels.add(round(float(h.tail(20).max()), 2))
    levels.add(round(float(l.tail(20).min()), 2))
    levels.add(round(float(c.tail(20).median()), 2))

    return sorted(lv for lv in levels if np.isfinite(lv))


def target_price_band(df: pd.DataFrame):
    last = df.iloc[-1]
    px_close = float(last["Close"])
    atrv = float(last["ATR"]) if pd.notna(last.get("ATR", np.nan)) else np.nan

    levels = _local_extrema_levels(df["Close"], df["High"], df["Low"])

    if not np.isfinite(atrv) or atrv <= 0:
        return {"base": px_close, "bull": None, "bear": None, "levels": levels}

    bull1 = px_close + 1.5 * atrv
    bull2 = px_close + 3.0 * atrv
    bear1 = px_close - 1.5 * atrv
    bear2 = px_close - 3.0 * atrv

    above = [x for x in levels if x > px_close]
    below = [x for x in levels if x < px_close]
    r1 = min(above) if above else None
    s1 = max(below) if below else None

    return {"base": px_close, "bull": (bull1, bull2, r1), "bear": (bear1, bear2, s1), "levels": levels}

# ============================================================
# CANLI FİYAT
# ============================================================
@st.cache_data(ttl=30, show_spinner=False)
def get_live_price(ticker: str) -> dict:
    out = {"last_price": np.nan, "currency": "", "exchange": "", "asof": ""}
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi:
            out["last_price"] = safe_float(getattr(fi, "last_price", None) or getattr(fi, "lastPrice", None))
            out["currency"] = getattr(fi, "currency", "") or ""
            out["exchange"] = getattr(fi, "exchange", "") or ""
    except Exception:
        pass
    return out

# ============================================================
# GEMİNİ — Header üzerinden güvenli kimlik doğrulama
# ============================================================
def _get_secret(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name, "")
        return str(v).strip() if v else default
    except Exception:
        return default


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
        return "❌ GEMINI_API_KEY bulunamadı. Streamlit Cloud → Settings → Secrets içine GEMINI_API_KEY=... ekleyin."

    # Güvenli: API anahtarını URL yerine header ile gönder
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    parts = [{"text": prompt}]
    if image_bytes:
        parts.append({
            "inlineData": {
                "mimeType": "image/png",
                "data": base64.b64encode(image_bytes).decode("utf-8"),
            }
        })

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=90)
        data = r.json()
    except Exception as e:
        return f"❌ Gemini API bağlantı hatası: {e}"

    if r.status_code >= 400:
        err = data.get("error", {})
        return f"❌ Gemini API hatası {r.status_code}: {err.get('message', str(data)[:300])}"

    cands = data.get("candidates") or []
    if not cands:
        return "⚠️ Gemini boş yanıt döndürdü."
    parts_out = (cands[0].get("content") or {}).get("parts") or []
    return "\n".join(p["text"] for p in parts_out if isinstance(p, dict) and "text" in p).strip() or "⚠️ Gemini metin üretmedi."

# ============================================================
# HABER DUYGU ANALİZİ
# ============================================================
@st.cache_data(ttl=30*60, show_spinner=False)
def get_news_sentiment(
    ticker: str,
    company_name: str = "",
    gemini_model: str = "gemini-1.5-flash",
    gemini_temp: float = 0.2,
) -> Dict[str, Any]:
    if not FEEDPARSER_OK:
        return {"error": "feedparser kütüphanesi eksik. `pip install feedparser` çalıştırın.", "sentiment": None, "summary": ""}

    try:
        query = company_name.strip() if company_name.strip() else ticker
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query + ' stock')}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        if not feed.entries:
            return {"error": "Haber bulunamadı.", "sentiment": None, "summary": ""}

        titles = [e.title for e in feed.entries[:10]]
        prompt = f"""Aşağıdaki haber başlıklarının duygu analizini yap (pozitif, negatif, nötr).
Sonuçları şu formatta ver:
- Pozitif: X haber
- Negatif: Y haber
- Nötr: Z haber
- Bileşik skor: (pozitif - negatif) / toplam
- Kısa özet (2 cümle Türkçe)

Haber Başlıkları:
{chr(10).join(f"- {t}" for t in titles)}"""

        response = gemini_generate_text(prompt=prompt, model=gemini_model, temperature=gemini_temp, max_output_tokens=512)
        pos = int(m.group(1)) if (m := re.search(r"Pozitif:?\s*(\d+)", response, re.I)) else 0
        neg = int(m.group(1)) if (m := re.search(r"Negatif:?\s*(\d+)", response, re.I)) else 0
        neu = int(m.group(1)) if (m := re.search(r"Nötr:?\s*(\d+)", response, re.I)) else 0
        total = pos + neg + neu

        return {
            "error": None,
            "sentiment": (pos - neg) / total if total > 0 else 0,
            "summary": response,
            "pos": pos / total if total > 0 else 0,
            "neg": neg / total if total > 0 else 0,
            "neu": neu / total if total > 0 else 0,
            "titles": titles[:5],
        }
    except Exception as e:
        return {"error": str(e), "sentiment": None, "summary": ""}

# ============================================================
# PRICE ACTION
# ============================================================
def _swing_points(high: pd.Series, low: pd.Series, left: int = 2, right: int = 2):
    hs, ls = [], []
    for i in range(left, len(high) - right):
        h_win = high.iloc[i - left: i + right + 1]
        l_win = low.iloc[i - left: i + right + 1]
        if high.iloc[i] == h_win.max():
            hs.append((high.index[i], float(high.iloc[i])))
        if low.iloc[i] == l_win.min():
            ls.append((low.index[i], float(low.iloc[i])))
    return hs, ls


def price_action_pack(df: pd.DataFrame, last_n: int = 20) -> dict:
    use = df.tail(last_n).copy()
    if len(use) < 10:
        return {"not": "Yetersiz bar", "last_n": int(len(use))}

    o, h, l, c = use["Open"].astype(float), use["High"].astype(float), use["Low"].astype(float), use["Close"].astype(float)
    s_highs, s_lows = _swing_points(h, l)

    recent_highs = [v for _, v in s_highs[-5:]]
    recent_lows = [v for _, v in s_lows[-5:]]
    res = max(recent_highs) if recent_highs else float(h.max())
    sup = min(recent_lows) if recent_lows else float(l.min())

    last_c, prev_c = float(c.iloc[-1]), float(c.iloc[-2]) if len(c) >= 2 else float(c.iloc[-1])
    bull_break = (last_c > res) and (prev_c <= res)
    bear_break = (last_c < sup) and (prev_c >= sup)

    vol_ok = None
    if "Volume" in use.columns:
        vol = use["Volume"].astype(float)
        vol_sma_v = float(vol.rolling(10).mean().iloc[-1]) if len(vol) >= 10 else float(vol.mean())
        vol_ok = float(vol.iloc[-1]) > vol_sma_v if np.isfinite(vol_sma_v) else None

    return {
        "last_n": int(len(use)),
        "destek": sup,
        "direnc": res,
        "yukari_kirilis": bool(bull_break),
        "asagi_kirilis": bool(bear_break),
        "hacim_onay": (None if vol_ok is None else bool(vol_ok)),
        "son_bar": {
            "tarih": str(use.index[-1]),
            "acilis": float(o.iloc[-1]),
            "yuksek": float(h.iloc[-1]),
            "dusuk": float(l.iloc[-1]),
            "kapanis": last_c,
        },
        "swing_yuksekler": [{"t": str(t), "f": float(p)} for t, p in s_highs[-6:]],
        "swing_dusukler": [{"t": str(t), "f": float(p)} for t, p in s_lows[-6:]],
    }


def df_snapshot_for_llm(df: pd.DataFrame, n: int = 25) -> dict:
    """
    LLM'e gönderilecek veri miktarını azalt.
    Tam tablo yerine: son N bar + özet istatistikler.
    """
    use_cols = ["Close", "Volume", "EMA50", "EMA200", "RSI", "MACD_hist",
                "BB_upper", "BB_lower", "ATR_PCT", "VOL_RATIO", "SCORE", "ENTRY", "EXIT"]
    cols = [c for c in use_cols if c in df.columns]
    tail = df[cols].tail(n).copy()
    tail.index = tail.index.astype(str)

    # Özet istatistikler
    summary = {
        "rsi_son": round(float(df["RSI"].iloc[-1]), 2) if "RSI" in df.columns else None,
        "rsi_5bar_ort": round(float(df["RSI"].tail(5).mean()), 2) if "RSI" in df.columns else None,
        "macd_hist_son": round(float(df["MACD_hist"].iloc[-1]), 4) if "MACD_hist" in df.columns else None,
        "trend": "yukari" if (df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1]) else "asagi",
        "atr_pct_son": round(float(df["ATR_PCT"].iloc[-1]) * 100, 2) if "ATR_PCT" in df.columns else None,
        "vol_ratio_son": round(float(df["VOL_RATIO"].iloc[-1]), 2) if "VOL_RATIO" in df.columns else None,
        "skor": round(float(df["SCORE"].iloc[-1]), 0) if "SCORE" in df.columns else None,
    }
    return {"ozet": summary, "son_barlar": tail.tail(n).to_dict(orient="records")}

# ============================================================
# UNIVERSE LOADER
# ============================================================
@st.cache_data(ttl=24*3600, show_spinner=False)
def load_universe_file(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        toks = re.split(r"[\s,;]+", raw.strip())
        tickers = list(dict.fromkeys(t.strip().upper() for t in toks if t.strip()))
        return sorted(tickers)
    except Exception:
        return []


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
    return m.iloc[0].drop(labels=["_tk", "_tk_naked"], errors="ignore").to_dict()


def merge_fa_row(screener_row, fundamentals, fa_eval) -> Dict[str, Any]:
    out = {}
    if fundamentals:
        out.update(fundamentals)
    if screener_row:
        out.update(screener_row)
    if fa_eval:
        out.update({
            "FA_mod": fa_eval.get("mode"),
            "FA_puan": fa_eval.get("score"),
            "FA_gecer": fa_eval.get("passed"),
            "FA_ok_sayi": fa_eval.get("ok_cnt"),
            "FA_kapsam": fa_eval.get("coverage"),
        })
    return out

# ============================================================
# RISK/ÖDÜL
# ============================================================
def rr_from_atr_stop(latest_row: pd.Series, tp_dict: dict, cfg: dict) -> dict:
    close = float(latest_row["Close"])
    atrv = safe_float(latest_row.get("ATR"))
    if not np.isfinite(atrv) or atrv <= 0:
        return {"rr": None, "stop": None, "risk": None, "reward": None}

    stop = close - cfg["atr_stop_mult"] * atrv
    risk = close - stop
    tp_mult = cfg.get("take_profit_mult", 2.0)
    target = close + tp_mult * cfg["atr_stop_mult"] * atrv
    reward = target - close

    if risk <= 0 or reward <= 0:
        return {"rr": None, "stop": stop, "risk": risk, "reward": reward}

    return {
        "rr": float(reward / risk),
        "stop": float(stop),
        "risk": float(risk),
        "reward": float(reward),
        "target_type": f"ATR Tabanlı TP (çarpan: {tp_mult:.1f}x)",
    }

# ============================================================
# VERİ YÜKLEYICI
# ============================================================
@st.cache_data(show_spinner=False)
def load_data_cached(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    return _flatten_yf(df)

# ============================================================
# ÖNCEDEN TANIMLI MODLAR
# ============================================================
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

# ============================================================
# PLOTLY TEMA (Koyu)
# ============================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#8b949e", family="IBM Plex Sans"),
    xaxis=dict(gridcolor="#21262d", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#21262d", showgrid=True, zeroline=False),
    legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1),
    margin=dict(t=40, b=40, l=60, r=20),
)

# ============================================================
# SESSION STATE BAŞLATMA
# ============================================================
if "screener_df" not in st.session_state:
    st.session_state.screener_df = pd.DataFrame()
if "last_market" not in st.session_state:
    st.session_state.last_market = None
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = [
        {"role": "assistant", "content": "Merhaba! Hisse analizi hakkında sorularını yanıtlamaya hazırım. Örneğin: *"Riskler neler?", "Bu seviyede alım mantıklı mı?", "Hedef fiyat nedir?"*"}
    ]
if "ta_ran" not in st.session_state:
    st.session_state.ta_ran = False
if "gemini_text" not in st.session_state:
    st.session_state.gemini_text = ""
if "pa_pack" not in st.session_state:
    st.session_state.pa_pack = {}
if "sentiment_result" not in st.session_state:
    st.session_state.sentiment_result = {}
if "last_ta_ticker" not in st.session_state:
    st.session_state.last_ta_ticker = ""

# ============================================================
# KENAR ÇUBUĞU
# ============================================================
with st.sidebar:
    st.markdown("## 📈 FA→TA Trading")
    st.caption("v2.0 — Temel + Teknik + AI")
    st.divider()

    st.markdown("### 🌍 Piyasa")
    market = st.selectbox(
        "Borsa",
        ["USA", "BIST"],
        index=0,
        help="USA: Amerikan hisse senetleri | BIST: Borsa İstanbul",
    )

    # Piyasa değiştiğinde screener'ı sıfırla
    if st.session_state.last_market != market:
        st.session_state.screener_df = pd.DataFrame()
        st.session_state.last_market = market
        st.session_state.selected_ticker = None

    usa_bucket = None
    if market == "USA":
        usa_bucket = st.selectbox(
            "Endeks",
            ["S&P 500", "Nasdaq 100"],
            index=0,
            help="S&P 500: 500 şirket | Nasdaq 100: Teknoloji ağırlıklı",
        )

    if market == "USA":
        universe = load_universe_file(pjoin("universes", "sp500.txt" if usa_bucket == "S&P 500" else "nasdaq100.txt"))
        st.caption(f"🔹 Evren: {usa_bucket} ({len(universe)} hisse)")
    else:
        universe = load_universe_file(pjoin("universes", "bist100.txt"))
        st.caption(f"🔹 Evren: BIST100 ({len(universe)} hisse)")

    if not universe:
        st.error("Universe listesi boş! `universes/` klasörünü kontrol edin.")
        st.stop()

    st.divider()
    st.markdown("### 🔬 Temel Analiz Filtresi")
    use_fa = st.checkbox("Temel analiz filtresini kullan", value=True)
    fa_mode = st.selectbox(
        "Strateji Modu",
        ["Kalite", "Değer", "Büyüme"],
        index=0,
        disabled=not use_fa,
        help="Kalite: yüksek karlılık & düşük borç | Değer: ucuz hisse | Büyüme: hızlı büyüme",
    )

    with st.expander("Temel Eşikler", expanded=False):
        roe = st.slider("ROE min", 0.0, 0.40, 0.15, 0.01, disabled=not use_fa)
        op_margin = st.slider("Faaliyet Marjı min", 0.0, 0.40, 0.10, 0.01, disabled=not use_fa)
        profit_margin = st.slider("Net Kar Marjı min", 0.0, 0.40, 0.08, 0.01, disabled=not use_fa)
        dte = st.slider("Borç/Özkaynak maks", 0.0, 3.0, 1.0, 0.05, disabled=not use_fa)
        fpe = st.slider("İleri F/K maks", 0.0, 60.0, 20.0, 1.0, disabled=not use_fa)
        peg = st.slider("PEG maks", 0.0, 5.0, 1.5, 0.1, disabled=not use_fa)
        ps = st.slider("F/S maks", 0.0, 30.0, 6.0, 0.5, disabled=not use_fa)
        pb = st.slider("F/D maks", 0.0, 30.0, 6.0, 0.5, disabled=not use_fa)
        rev_g = st.slider("Gelir Büyümesi min", 0.0, 0.50, 0.10, 0.01, disabled=not use_fa)
        earn_g = st.slider("Kazanç Büyümesi min", 0.0, 0.50, 0.10, 0.01, disabled=not use_fa)
        min_score = st.slider("Min Temel Puan", 0, 100, 60, 1, disabled=not use_fa)
        min_ok = st.slider("Min Geçen Kriter", 1, 5, 3, 1, disabled=not use_fa)
        min_coverage = st.slider("Min Veri Kapsamı", 1, 5, 3, 1, disabled=not use_fa)

    thresholds = dict(
        roe=roe, op_margin=op_margin, profit_margin=profit_margin, dte=dte,
        fpe=fpe, peg=peg, ps=ps, pb=pb, rev_g=rev_g, earn_g=earn_g,
        min_score=min_score, min_ok=min_ok, min_coverage=min_coverage,
    )

    run_screener = st.button(
        "🔎 Screener Çalıştır",
        type="secondary",
        disabled=not use_fa,
        use_container_width=True,
    )

    st.divider()
    st.markdown("### 📊 Teknik Analiz")

    if st.session_state.selected_ticker:
        st.info(f"Screener seçimi: **{st.session_state.selected_ticker}**")

    default_ticker = st.session_state.selected_ticker or ("AAPL" if market == "USA" else "THYAO")
    raw_ticker = st.text_input(
        "Sembol",
        value=default_ticker,
        help="USA: AAPL, SPY | BIST: THYAO (otomatik .IS eklenir)",
    )
    ticker = normalize_ticker(raw_ticker, market)

    col_int, col_per = st.columns(2)
    with col_int:
        interval = st.selectbox(
            "Interval",
            ["1d", "1h", "1wk", "4d"],
            index=0,
            format_func=lambda x: {
                "1d": "Günlük", "1h": "Saatlik",
                "1wk": "Haftalık", "4d": "4 Günlük",
            }.get(x, x),
            help="Mum zaman dilimi. Backtest için günlük/haftalık önerilir.",
        )
    with col_per:
        period = st.selectbox(
            "Periyot",
            ["6mo", "1y", "2y", "5y", "10y"],
            index=3,
            format_func=lambda x: {
                "6mo": "6 Ay", "1y": "1 Yıl",
                "2y": "2 Yıl", "5y": "5 Yıl", "10y": "10 Yıl",
            }.get(x, x),
        )

    preset_name = st.selectbox(
        "Risk Profili",
        list(PRESETS.keys()),
        index=1,
        help="Defansif: düşük risk | Dengeli: orta | Agresif: yüksek risk",
    )

    with st.expander("Teknik Parametreler", expanded=False):
        ema_fast = st.number_input("EMA Hızlı", 5, 100, 50, 1)
        ema_slow = st.number_input("EMA Yavaş", 50, 400, 200, 1)
        rsi_period = st.number_input("RSI Periyodu", 5, 30, 14, 1)
        bb_period = st.number_input("Bollinger Periyodu", 10, 50, 20, 1)
        bb_std = st.number_input("Bollinger Std", 1.0, 3.5, 2.0, 0.1)
        atr_period = st.number_input("ATR Periyodu", 5, 30, 14, 1)
        vol_sma = st.number_input("Hacim SMA", 5, 60, 20, 1)

    with st.expander("Piyasa Filtreleri", expanded=False):
        use_spy_filter = st.checkbox("SPY > EMA200 (Sadece USA)", value=True, disabled=market != "USA")
        use_bist_filter = st.checkbox("XU100 > EMA200 (Sadece BIST)", value=True, disabled=market != "BIST")
        use_higher_tf_filter = st.checkbox("Haftalık trend filtresi", value=True)

    with st.expander("Risk & Backtest Ayarları", expanded=False):
        initial_capital = st.number_input("Başlangıç Sermayesi", 100.0, 10_000_000.0, 10_000.0, 500.0)
        risk_per_trade = st.slider("İşlem başı risk (%)", 0.002, 0.05, 0.01, 0.001,
                                   format="%.3f",
                                   help="Stop loss ile en fazla kaybedilecek sermaye yüzdesi")
        commission_bps = st.number_input("Komisyon (bps)", 0.0, 50.0, 5.0, 1.0)
        slippage_bps = st.number_input("Slipaj (bps)", 0.0, 20.0, 2.0, 1.0)
        risk_free_annual = st.number_input("Risksiz Faiz (yıllık)", 0.0, 0.30, 0.0, 0.01)

    st.divider()
    st.markdown("### 🤖 AI Ayarları (Gemini)")
    ai_on = st.checkbox("Gemini AI aktif", value=True)
    gemini_model = st.text_input("Model", value="gemini-1.5-flash")
    gemini_temp = st.slider("Sıcaklık", 0.0, 1.0, 0.2, 0.05)
    gemini_max_tokens = st.slider("Maks Token", 256, 8192, 2048, 128)

    st.divider()
    st.markdown("### 📰 Haber Analizi")
    use_sentiment = st.checkbox("Google News duygu analizi", value=True,
                                 help="feedparser kütüphanesi gerekir")

    st.divider()
    run_btn = st.button("🚀 Analizi Çalıştır", type="primary", use_container_width=True)
    if run_btn:
        st.session_state.ta_ran = True
        st.session_state.last_ta_ticker = ticker
        st.session_state.gemini_text = ""
        st.session_state.ai_messages = [
            {"role": "assistant", "content": f"**{ticker}** analizi hazır. Sorularınızı yazabilirsiniz."}
        ]

# ============================================================
# SCREENER ÇALIŞTIRMA — Paralel
# ============================================================
if run_screener and use_fa:
    def _fetch_and_score(tk):
        tk_norm = normalize_ticker(tk, market)
        f = fetch_fundamentals_generic(tk_norm, market=market)
        score, breakdown, passed = fundamental_score_row(f, fa_mode, thresholds)
        f["FA_puan"] = score
        f["FA_gecer"] = passed
        f["FA_ok_sayi"] = sum(1 for v in breakdown.values() if v.get("available") and v.get("ok"))
        f["FA_kapsam"] = sum(1 for v in breakdown.values() if v.get("available"))
        return f

    progress_bar = st.progress(0, text="Temel veriler paralel olarak çekiliyor...")
    rows = []
    total = len(universe)

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(_fetch_and_score, tk): tk for tk in universe}
        done = 0
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception:
                pass
            done += 1
            progress_bar.progress(done / total, text=f"Çekiliyor... {done}/{total}")

    progress_bar.empty()

    if rows:
        sdf = pd.DataFrame(rows)
        sdf["_sort_pass"] = sdf["FA_gecer"].astype(int)
        sdf = sdf.sort_values(
            ["_sort_pass", "FA_puan", "FA_kapsam"],
            ascending=[False, False, False],
        ).drop(columns=["_sort_pass"])
        st.session_state.screener_df = sdf.copy()
        st.success(f"✅ Screener tamamlandı: {len(sdf)} hisse tarandı.")

# ============================================================
# CFG
# ============================================================
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

# ============================================================
# TA ÇALIŞTIRILMADIYSA: SCREENER GÖSTER
# ============================================================
if not st.session_state.ta_ran:
    st.markdown("""
    <div style='text-align:center; padding: 40px 0 20px 0;'>
        <h1 style='font-size:2rem; color:#e6edf3; font-weight:700; letter-spacing:-0.03em;'>
            📈 FA→TA Trading Uygulaması
        </h1>
        <p style='color:#8b949e; font-size:1rem; margin-top:8px;'>
            Temel analiz ile evreni daralt → Teknik analiz ile zamanla → Gemini AI ile derinleştir
        </p>
    </div>
    """, unsafe_allow_html=True)

    if use_fa and not st.session_state.screener_df.empty:
        sdf = st.session_state.screener_df.copy()
        col_show = ["ticker", "longName", "FA_gecer", "FA_puan", "FA_kapsam",
                    "sector", "trailingPE", "forwardPE", "returnOnEquity",
                    "operatingMargins", "debtToEquity", "revenueGrowth", "marketCap"]
        sdf_show = sdf[[c for c in col_show if c in sdf.columns]].copy()

        pass_count = int(sdf["FA_gecer"].sum()) if "FA_gecer" in sdf.columns else 0
        total_count = len(sdf)

        c1, c2, c3 = st.columns(3)
        c1.metric("Taranan Hisse", total_count)
        c2.metric("GEÇEN", pass_count, delta=f"%{pass_count/total_count*100:.1f}")
        c3.metric("Mod", fa_mode)

        st.dataframe(sdf_show, use_container_width=True, height=380)

        pass_list = sdf.loc[sdf["FA_gecer"] == True, "ticker"].tolist()
        if not pass_list:
            st.warning("⚠️ Bu eşiklerle geçen hisse yok. Eşikleri gevşetmeyi deneyin.")
        else:
            picked = st.selectbox("Geçen listesinden hisse seç", pass_list)
            if st.button("➡️ Teknik Analize Aktar", type="primary"):
                st.session_state.selected_ticker = picked
                st.session_state.ta_ran = True
                st.session_state.last_ta_ticker = normalize_ticker(picked, market)
                st.rerun()

    else:
        st.info("👈 Sol menüden piyasa ve parametreleri seçip **Analizi Çalıştır** butonuna basın.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <div style='font-size:1.5rem; margin-bottom:8px;'>🔬</div>
                <div style='color:#58a6ff; font-weight:600; margin-bottom:4px;'>Temel Analiz</div>
                <div style='color:#8b949e; font-size:0.85rem;'>ROE, Marj, Borç, Büyüme kriterleri ile hisse filtrele</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <div style='font-size:1.5rem; margin-bottom:8px;'>📊</div>
                <div style='color:#58a6ff; font-weight:600; margin-bottom:4px;'>Teknik Analiz</div>
                <div style='color:#8b949e; font-size:0.85rem;'>EMA, RSI, MACD, Bollinger, ATR, OBV sinyalleri</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <div style='font-size:1.5rem; margin-bottom:8px;'>🤖</div>
                <div style='color:#58a6ff; font-weight:600; margin-bottom:4px;'>Gemini AI</div>
                <div style='color:#8b949e; font-size:0.85rem;'>Görsel + veri analizi, hedef fiyat, risk değerlendirmesi</div>
            </div>
            """, unsafe_allow_html=True)

    st.stop()

# ============================================================
# TA PIPELINE — VERİ VE HESAPLAMALAR
# ============================================================
status_container = st.empty()

with status_container.container():
    st.info(f"⚙️ `{ticker}` analizi çalışıyor...")

# Piyasa rejimi
market_filter_ok = True
if market == "USA" and use_spy_filter:
    market_filter_ok = get_spy_regime_ok()
elif market == "BIST" and use_bist_filter:
    market_filter_ok = get_bist_regime_ok()

# Üst zaman dilimi trendi
higher_tf_filter_ok = True
if use_higher_tf_filter:
    higher_tf_filter_ok = get_higher_tf_trend(ticker, "1wk", 200)

# Haber analizi
sentiment_result = {}
if use_sentiment and ai_on and FEEDPARSER_OK:
    company_name = ""
    if not st.session_state.screener_df.empty:
        row = find_screener_row(st.session_state.screener_df, ticker)
        if row and row.get("longName"):
            company_name = row["longName"]
    sentiment_result = get_news_sentiment(ticker, company_name, gemini_model, gemini_temp)
    st.session_state.sentiment_result = sentiment_result

# Fiyat verisi
df_raw = load_data_cached(ticker, period, interval)

if df_raw.empty:
    status_container.empty()
    st.error(f"❌ `{ticker}` için veri alınamadı. Sembolü kontrol edin.")
    st.stop()

required_cols = {"Open", "High", "Low", "Close", "Volume"}
if not required_cols.issubset(set(df_raw.columns)):
    status_container.empty()
    st.error("Veri setinde gerekli OHLCV sütunları eksik.")
    st.stop()

# Göstergeler
df = build_features(df_raw, cfg)

# Benchmark
bm_ticker = "SPY" if market == "USA" else "XU100.IS"
bm_df = load_data_cached(bm_ticker, period, interval)
benchmark_returns = bm_df["Close"].pct_change().dropna() if not bm_df.empty else None

# Sinyal
df, checkpoints = signal_with_checkpoints(df, cfg, market_filter_ok, higher_tf_filter_ok)
latest = df.iloc[-1]
live = get_live_price(ticker)
live_price = live.get("last_price", np.nan)

# Sinyal etiketi
score_val = float(latest["SCORE"])
if int(latest["ENTRY"]) == 1:
    rec = "AL"
    rec_color = "al"
elif int(latest["EXIT"]) == 1:
    rec = "SAT"
    rec_color = "sat"
else:
    if score_val >= 80:
        rec = "AL — Güçlü"
        rec_color = "al"
    elif score_val >= 60:
        rec = "İZLE"
        rec_color = "izle"
    else:
        rec = "UZAK DUR"
        rec_color = "sat"

# Backtest
eq, tdf, metrics = backtest_long_only(df, cfg, risk_free_annual=risk_free_annual, benchmark_returns=benchmark_returns)
tp = target_price_band(df)
rr_info = rr_from_atr_stop(latest, tp, cfg)
overbought_result = detect_speculation(df)
pa = price_action_pack(df, last_n=20)
st.session_state.pa_pack = pa

status_container.empty()

# ============================================================
# GRAFİKLER
# ============================================================
def make_price_fig(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Fiyat",
        increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        increasing_fillcolor="#3fb95044", decreasing_fillcolor="#f8514944",
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA Hızlı",
                             line=dict(color="#58a6ff", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA Yavaş",
                             line=dict(color="#e3b341", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Üst",
                             line=dict(color="#8b949e", dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Alt",
                             line=dict(color="#8b949e", dash="dot", width=1),
                             fill="tonexty", fillcolor="rgba(139,148,158,0.05)"))
    entries = df[df["ENTRY"] == 1]
    exits = df[df["EXIT"] == 1]
    fig.add_trace(go.Scatter(
        x=entries.index, y=entries["Low"] * 0.995,
        mode="markers", name="Giriş",
        marker=dict(symbol="triangle-up", size=10, color="#3fb950"),
    ))
    fig.add_trace(go.Scatter(
        x=exits.index, y=exits["High"] * 1.005,
        mode="markers", name="Çıkış",
        marker=dict(symbol="triangle-down", size=10, color="#f85149"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=520,
        xaxis_rangeslider_visible=False,
        title=dict(text=f"{ticker} — Fiyat + EMA + Bollinger + Sinyaller", font=dict(color="#e6edf3", size=14)),
    )
    return fig


def make_rsi_fig(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#58a6ff", width=1.5)))
    fig.add_hrect(y0=70, y1=100, fillcolor="#f85149", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="#3fb950", opacity=0.08, line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color="#f85149", line_width=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#3fb950", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT, height=200,
                      title=dict(text="RSI", font=dict(color="#e6edf3", size=13)),
                      yaxis=dict(range=[0, 100], gridcolor="#21262d"))
    return fig


def make_macd_fig(df):
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in df["MACD_hist"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogram", marker_color=colors, opacity=0.7))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#58a6ff", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Sinyal", line=dict(color="#e3b341", width=1.5)))
    fig.update_layout(**PLOTLY_LAYOUT, height=200,
                      title=dict(text="MACD", font=dict(color="#e6edf3", size=13)))
    return fig


def make_stoch_fig(df):
    fig = go.Figure()
    if "STOCH_RSI_K" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_RSI_K"], name="K",
                                 line=dict(color="#58a6ff", width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_RSI_D"], name="D",
                                 line=dict(color="#e3b341", width=1.5)))
        fig.add_hrect(y0=80, y1=100, fillcolor="#f85149", opacity=0.08, line_width=0)
        fig.add_hrect(y0=0, y1=20, fillcolor="#3fb950", opacity=0.08, line_width=0)
        fig.add_hline(y=80, line_dash="dash", line_color="#f85149", line_width=1)
        fig.add_hline(y=20, line_dash="dash", line_color="#3fb950", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT, height=200,
                      title=dict(text="Stochastic RSI", font=dict(color="#e6edf3", size=13)),
                      yaxis=dict(range=[0, 100], gridcolor="#21262d"))
    return fig


def make_bbwidth_fig(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_WIDTH"] * 100, name="BB Genişlik %",
                             line=dict(color="#bc8cff", width=1.5), fill="tozeroy",
                             fillcolor="rgba(188,140,255,0.08)"))
    fig.add_hline(y=2, line_dash="dash", line_color="#e3b341", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT, height=200,
                      title=dict(text="Bollinger Genişlik %", font=dict(color="#e6edf3", size=13)))
    return fig


def make_volratio_fig(df):
    colors = ["#f85149" if v >= 1.5 else "#8b949e" for v in df["VOL_RATIO"].fillna(0)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["VOL_RATIO"], name="Hacim Oranı", marker_color=colors))
    fig.add_hline(y=1.5, line_dash="dash", line_color="#f85149", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT, height=200,
                      title=dict(text="Hacim Oranı (Hacim/SMA)", font=dict(color="#e6edf3", size=13)))
    return fig


def make_equity_fig(eq):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values, name="Sermaye",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280,
                      title=dict(text="Backtest — Sermaye Eğrisi", font=dict(color="#e6edf3", size=13)))
    return fig


fig_price = make_price_fig(df)
fig_rsi = make_rsi_fig(df)
fig_macd = make_macd_fig(df)
fig_stoch = make_stoch_fig(df)
fig_bbwidth = make_bbwidth_fig(df)
fig_volratio = make_volratio_fig(df)
fig_equity = make_equity_fig(eq)

figs_for_report = {
    "Fiyat + EMA + Bollinger + Sinyaller": fig_price,
    "RSI": fig_rsi,
    "MACD": fig_macd,
    "Stochastic RSI": fig_stoch,
    "Bollinger Genişlik": fig_bbwidth,
    "Hacim Oranı": fig_volratio,
    "Sermaye Eğrisi": fig_equity,
}

# ============================================================
# UYGULAMA BAŞLIĞI
# ============================================================
st.markdown(f"""
<div style='display:flex; align-items:center; justify-content:space-between; padding: 8px 0 16px 0; border-bottom: 1px solid #21262d; margin-bottom: 20px;'>
    <div>
        <h1 style='font-size:1.5rem; font-weight:700; color:#e6edf3; margin:0; letter-spacing:-0.02em;'>
            📈 {ticker}
            <span style='font-size:1rem; color:#8b949e; font-weight:400; margin-left:8px;'>
                {market} · {interval} · {period}
            </span>
        </h1>
        <p style='color:#8b949e; font-size:0.8rem; margin:4px 0 0 0;'>
            {time.strftime('%Y-%m-%d %H:%M')} · Preset: {preset_name}
        </p>
    </div>
    <div>
        <span class='badge-{rec_color.lower()}'>⬡ {rec}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SEKMELER
# ============================================================
tab_dashboard, tab_chat, tab_screener, tab_heatmap, tab_export = st.tabs([
    "📊 Dashboard",
    "💬 AI Sohbet",
    "🔬 Screener",
    "🔥 Heatmap",
    "📄 Rapor",
])

# ────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ────────────────────────────────────────────────────────────
with tab_dashboard:

    # ── Özet Metrikler ──
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Son Kapanış", f"{latest['Close']:.2f}")
    c2.metric("Anlık Fiyat", f"{live_price:.2f}" if np.isfinite(live_price) else "N/A")
    c3.metric("Teknik Skor", f"{score_val:.0f} / 100")
    c4.metric("RSI", f"{latest['RSI']:.1f}")
    c5.metric("ATR%", f"{float(latest.get('ATR_PCT', 0))*100:.2f}%")
    c6.metric("Piyasa Rejimi", "🐂 Boğa" if market_filter_ok else "🐻 Ayı")

    st.markdown("")

    # ── Kontrol Noktaları + Aşırı Alım Yanyana ──
    col_cp, col_ob = st.columns([1, 1])

    with col_cp:
        st.markdown("""<div class='section-header'><h2>✅ Kontrol Noktaları</h2></div>""", unsafe_allow_html=True)
        for k, v in checkpoints.items():
            icon = "✅" if v else "❌"
            color = "#3fb950" if v else "#f85149"
            bg = "#1a3d1a" if v else "#3d1a1a"
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:10px; padding:8px 12px;
                        background:{bg}22; border-left:3px solid {color};
                        border-radius:6px; margin-bottom:4px; font-size:0.85rem; color:#cdd9e5;'>
                {icon} {k}
            </div>""", unsafe_allow_html=True)

    with col_ob:
        st.markdown("""<div class='section-header'><h2>🎯 Aşırı Alım / Spekülasyon</h2></div>""", unsafe_allow_html=True)

        ob_s = overbought_result["overbought_score"]
        os_s = overbought_result["oversold_score"]
        sp_s = overbought_result["speculation_score"]

        verdict = overbought_result["verdict"]
        if "AŞIRI DEĞERLİ" in verdict:
            v_color, v_bg = "#f85149", "#3d1a1a"
        elif "AŞIRI DEĞERSİZ" in verdict:
            v_color, v_bg = "#3fb950", "#1a3d1a"
        elif "SPEKÜLATİF" in verdict:
            v_color, v_bg = "#e3b341", "#3d2e00"
        else:
            v_color, v_bg = "#58a6ff", "#1a2a3d"

        st.markdown(f"""
        <div style='padding:12px 16px; background:{v_bg}44; border:1px solid {v_color}44;
                    border-radius:10px; margin-bottom:12px;'>
            <div style='color:{v_color}; font-weight:600; font-size:0.9rem;'>{verdict}</div>
        </div>""", unsafe_allow_html=True)

        for label, val, color in [
            ("Aşırı Alım", ob_s, "#f85149"),
            ("Aşırı Satım", os_s, "#3fb950"),
            ("Spekülasyon", sp_s, "#e3b341"),
        ]:
            bar_class = "red" if color == "#f85149" else ("green" if color == "#3fb950" else "yellow")
            st.markdown(f"""
            <div style='margin-bottom:10px;'>
                <div style='display:flex; justify-content:space-between; font-size:0.8rem; color:#8b949e; margin-bottom:3px;'>
                    <span>{label}</span><span style='color:{color};font-weight:600;'>{val}/100</span>
                </div>
                <div class='score-bar-container'>
                    <div class='score-bar-fill-{bar_class}' style='width:{val}%;'></div>
                </div>
            </div>""", unsafe_allow_html=True)

        if overbought_result["details"]:
            with st.expander("Detaylar", expanded=False):
                for v in overbought_result["details"].values():
                    st.markdown(f"- {v}")

    st.markdown("")

    # ── Hedef Fiyat Bandı ──
    st.markdown("""<div class='section-header'><h2>🎯 Hedef Fiyat Bandı</h2></div>""", unsafe_allow_html=True)

    base_px = float(tp["base"])
    bc1, bc2, bc3, bc4 = st.columns(4)

    bc1.metric("Baz Fiyat", f"{base_px:.2f}")

    if tp.get("bull"):
        b1, b2, r1 = tp["bull"]
        bc2.metric("Yükseliş Hedefi 1", f"{b1:.2f}",
                   delta=f"+{pct_dist(b1, base_px):.2f}%" if pct_dist(b1, base_px) else "")
        bc3.metric("Yükseliş Hedefi 2", f"{b2:.2f}",
                   delta=f"+{pct_dist(b2, base_px):.2f}%" if pct_dist(b2, base_px) else "")
        if r1:
            bc3.caption(f"Yakın direnç: **{r1:.2f}**")

    if tp.get("bear"):
        b1, b2, s1 = tp["bear"]
        bc4.metric(
            f"Stop Loss · RR: {fmt_rr(rr_info.get('rr'))}",
            f"{b1:.2f}",
            delta=f"{pct_dist(b1, base_px):.2f}%" if pct_dist(b1, base_px) else "",
            delta_color="inverse",
        )

    with st.expander("Destek/Direnç Seviyeleri (Yerel Ekstremum)", expanded=False):
        levels = tp.get("levels", [])
        s1_val = tp["bear"][2] if tp.get("bear") else None
        r1_val = tp["bull"][2] if tp.get("bull") else None

        rows_lv = []
        for lv in levels:
            dist = pct_dist(float(lv), base_px)
            tag = ""
            if s1_val and abs(float(lv) - float(s1_val)) < 0.01:
                tag = "🟩 Yakın Destek"
            if r1_val and abs(float(lv) - float(r1_val)) < 0.01:
                tag = "🟥 Yakın Direnç"
            yon = "📈" if float(lv) > base_px else "📉"
            rows_lv.append({"Seviye": f"{float(lv):.2f}", "Uzaklık %": f"{dist:+.2f}%" if dist else "N/A", "Yön": yon, "Not": tag})

        if rows_lv:
            st.dataframe(pd.DataFrame(rows_lv), use_container_width=True, hide_index=True)

    st.markdown("")

    # ── Fiyat Grafiği ──
    st.markdown("""<div class='section-header'><h2>📈 Fiyat Grafiği</h2></div>""", unsafe_allow_html=True)
    st.plotly_chart(fig_price, use_container_width=True)

    # ── Alt Göstergeler ──
    st.markdown("""<div class='section-header'><h2>📉 Teknik Göstergeler</h2></div>""", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(fig_rsi, use_container_width=True)
        st.plotly_chart(fig_stoch, use_container_width=True)
    with g2:
        st.plotly_chart(fig_macd, use_container_width=True)
        st.plotly_chart(fig_bbwidth, use_container_width=True)

    st.plotly_chart(fig_volratio, use_container_width=True)

    # ── Backtest ──
    st.markdown("""<div class='section-header'><h2>🧪 Backtest Sonuçları</h2></div>""", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m5, m6, m7, m8 = st.columns(4)

    total_ret = metrics["Toplam Getiri"]
    ann_ret = metrics["Yıllık Getiri"]

    m1.metric("Toplam Getiri", f"{total_ret*100:.1f}%", delta=f"{'↑' if total_ret>0 else '↓'}")
    m2.metric("Yıllık Getiri", f"{ann_ret*100:.1f}%")
    m3.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
    m4.metric("Sortino", f"{metrics['Sortino']:.2f}")
    m5.metric("Maks Drawdown", f"{metrics['Maks Drawdown']*100:.1f}%")
    m6.metric("Kazanma Oranı", f"{metrics['Kazanma Oranı']*100:.1f}%")
    m7.metric("Kar Faktörü", f"{min(metrics['Kar Faktörü'], 99):.2f}")
    m8.metric(
        "Kelly Önerisi",
        f"{metrics['Kelly % (Öneri)']:.1f}%",
        help="Tam Kelly'nin %50'si, maks %12.5. Bu paylaşımın pozisyon büyüklüğüne üst sınırıdır.",
    )

    with st.expander("Ek Metrikler (Beta, Alpha, Ulcer Index)", expanded=False):
        ex1, ex2, ex3, ex4 = st.columns(4)
        ex1.metric("Beta", f"{metrics['Beta']:.2f}")
        ex2.metric("Alpha", f"{metrics['Alpha']:.4f}")
        ex3.metric("Bilgi Oranı", f"{metrics['Bilgi Oranı']:.2f}")
        ex4.metric("Ulcer Index", f"{metrics['Ulcer Index']:.4f}")

    st.plotly_chart(fig_equity, use_container_width=True)

    with st.expander(f"İşlem Listesi ({metrics['İşlem Sayısı']} işlem)", expanded=False):
        if not tdf.empty:
            st.dataframe(tdf, use_container_width=True, height=280)
        else:
            st.info("Bu periyotta sinyal üretilmedi.")

    # ── Haber Duygu Analizi ──
    if sentiment_result and not sentiment_result.get("error"):
        st.markdown("""<div class='section-header'><h2>📰 Haber Duygu Analizi</h2></div>""", unsafe_allow_html=True)

        sentiment_val = sentiment_result.get("sentiment", 0) or 0
        pos_r = sentiment_result.get("pos", 0) or 0
        neg_r = sentiment_result.get("neg", 0) or 0
        neu_r = sentiment_result.get("neu", 0) or 0

        s1c, s2c, s3c, s4c = st.columns(4)
        s1c.metric("Bileşik Skor", f"{sentiment_val:.2f}", help="-1 (çok negatif) → +1 (çok pozitif)")
        s2c.metric("Pozitif", f"{pos_r*100:.0f}%")
        s3c.metric("Negatif", f"{neg_r*100:.0f}%")
        s4c.metric("Nötr", f"{neu_r*100:.0f}%")

        if sentiment_result.get("titles"):
            with st.expander("Son Haberler", expanded=False):
                for t_title in sentiment_result["titles"]:
                    st.markdown(f"- {t_title}")

        with st.expander("Gemini Duygu Analizi Detayı", expanded=False):
            st.markdown(sentiment_result.get("summary", ""))

    # ── Gemini Hızlı Analiz ──
    st.markdown("""<div class='section-header'><h2>🤖 Gemini Hızlı Analiz</h2></div>""", unsafe_allow_html=True)

    if not ai_on:
        st.info("Gemini kapalı — sol menüden etkinleştirin.")
    else:
        quick_prompt = st.text_area(
            "Gemini'ye sor:",
            value=(
                "Bu hissenin mevcut durumunu değerlendir. "
                "Aşırı alım/satım var mı? AL/SAT/İZLE önerisi ve stop/hedef fiyat tablosu ver."
            ),
            height=100,
            key="quick_prompt",
        )

        col_g1, col_g2 = st.columns([3, 1])
        with col_g1:
            analyze_btn = st.button("🔍 Gemini ile Analiz Et", use_container_width=True)
        with col_g2:
            clear_btn = st.button("🗑️ Temizle", use_container_width=True, key="clear_gemini")

        if clear_btn:
            st.session_state.gemini_text = ""

        if analyze_btn:
            with st.spinner("Gemini analiz ediyor..."):
                snap = df_snapshot_for_llm(df, n=25)
                fa_row_local = None
                if use_fa and not st.session_state.screener_df.empty:
                    sr = find_screener_row(st.session_state.screener_df, ticker)
                    f_s = fetch_fundamentals_generic(ticker, market=market)
                    f_score, f_bd, f_pass = fundamental_score_row(f_s, fa_mode, thresholds)
                    fa_row_local = merge_fa_row(sr, f_s, {
                        "mode": fa_mode, "score": f_score, "passed": f_pass,
                        "ok_cnt": sum(1 for v in f_bd.values() if v.get("available") and v.get("ok")),
                        "coverage": sum(1 for v in f_bd.values() if v.get("available")),
                    })

                prompt = f"""Sen kıdemli bir teknik analist ve risk yöneticisisin. Kesin yatırım tavsiyesi verme, eğitim amaçlı objektif analiz yap.

Analiz edilecek hisse: {ticker} ({market})
Algoritmik sinyal: {rec}
Son kapanış: {float(latest['Close']):.2f}

Veri özeti (JSON):
{json.dumps({
    "teknik_ozet": snap["ozet"],
    "hedef_bant": tp,
    "risk_odül": rr_info,
    "aşırı_alım_analizi": overbought_result,
    "price_action": pa,
    "temel_puan": fa_row_local.get("FA_puan") if fa_row_local else None,
    "haber_duygu": sentiment_result.get("sentiment") if sentiment_result else None,
}, ensure_ascii=False, default=str)}

Kullanıcı sorusu: {quick_prompt}

Yanıtının sonunda şu tabloyu doldur:

| Seviye | Fiyat |
|--------|-------|
| Önerilen Giriş | ... |
| İlk Hedef | ... |
| İkinci Hedef | ... |
| Stop Loss | ... |
"""
                image_bytes = None
                try:
                    image_bytes = fig_price.to_image(format="png", scale=2) if KALEIDO_OK else None
                except Exception:
                    pass

                result = gemini_generate_text(
                    prompt=prompt, model=gemini_model,
                    temperature=gemini_temp, max_output_tokens=gemini_max_tokens,
                    image_bytes=image_bytes,
                )
                st.session_state.gemini_text = result

                # Sohbet geçmişine ekle
                st.session_state.ai_messages.append({"role": "user", "content": quick_prompt})
                st.session_state.ai_messages.append({"role": "assistant", "content": result})

        if st.session_state.gemini_text:
            st.markdown(st.session_state.gemini_text)

# ────────────────────────────────────────────────────────────
# TAB 2 — AI SOHBET
# ────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown("""<div class='section-header'><h2>💬 AI Sohbet — Gemini ile Analiz Konuş</h2></div>""", unsafe_allow_html=True)

    if not ai_on:
        st.warning("Gemini kapalı. Sol menüden 'Gemini AI aktif' seçeneğini açın.")
    else:
        # Mesaj geçmişini göster
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.ai_messages:
                if msg["role"] == "assistant":
                    st.markdown(f"<div class='chat-message-ai'>🤖 {msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-message-user'>👤 {msg['content']}</div>", unsafe_allow_html=True)

        st.markdown("")

        # Hızlı soru önerileri
        st.caption("Hızlı sorular:")
        qc1, qc2, qc3, qc4 = st.columns(4)
        quick_questions = [
            "Riskler neler?",
            "Hangi seviyede alırım?",
            "Stop loss nereye koymalıyım?",
            "Bu trend ne kadar sürer?",
        ]
        for i, (qcol, qq) in enumerate(zip([qc1, qc2, qc3, qc4], quick_questions)):
            if qcol.button(qq, key=f"qq_{i}", use_container_width=True):
                st.session_state.pending_chat = qq

        # Mesaj girişi
        user_input = st.chat_input("Sorunuzu yazın...")
        if "pending_chat" in st.session_state:
            user_input = st.session_state.pop("pending_chat")

        if user_input:
            st.session_state.ai_messages.append({"role": "user", "content": user_input})

            with st.spinner("Gemini yanıt hazırlıyor..."):
                # Tüm bağlam
                context = {
                    "ticker": ticker,
                    "market": market,
                    "sinyal": rec,
                    "skor": score_val,
                    "teknik_ozet": df_snapshot_for_llm(df, n=15)["ozet"],
                    "hedef_bant": tp,
                    "rr": rr_info,
                    "asiri_alim": overbought_result,
                    "price_action": pa,
                    "haber_duygu": sentiment_result.get("sentiment") if sentiment_result else None,
                    "backtest": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
                }

                # Sohbet geçmişini kısalt (son 6 mesaj)
                recent_msgs = st.session_state.ai_messages[-7:-1]
                history_txt = "\n".join(
                    f"{'Kullanıcı' if m['role'] == 'user' else 'Asistan'}: {m['content'][:300]}"
                    for m in recent_msgs
                )

                full_prompt = f"""Sen kıdemli bir teknik analist asistanısın.
{ticker} hissesini analiz ediyorsun.

Bağlam verileri:
{json.dumps(context, ensure_ascii=False, default=str)}

Önceki konuşma:
{history_txt}

Kullanıcının yeni sorusu: {user_input}

Türkçe, net ve özlü yanıt ver. Yatırım tavsiyesi değil, eğitim amaçlı analiz sun."""

                response = gemini_generate_text(
                    prompt=full_prompt, model=gemini_model,
                    temperature=gemini_temp, max_output_tokens=gemini_max_tokens,
                )
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                st.rerun()

        # Sohbeti sıfırla
        if st.button("🗑️ Sohbeti Sıfırla", key="reset_chat"):
            st.session_state.ai_messages = [
                {"role": "assistant", "content": f"**{ticker}** analizi için yeni sohbet başladı. Sorularınızı yazabilirsiniz."}
            ]
            st.rerun()

# ────────────────────────────────────────────────────────────
# TAB 3 — SCREENER
# ────────────────────────────────────────────────────────────
with tab_screener:
    st.markdown("""<div class='section-header'><h2>🔬 Temel Analiz Screener</h2></div>""", unsafe_allow_html=True)

    if not use_fa:
        st.info("Temel analiz filtresi kapalı. Sol menüden etkinleştirin.")
    elif st.session_state.screener_df.empty:
        st.info("Henüz screener çalıştırılmadı. Sol menüden **Screener Çalıştır** butonuna basın.")
    else:
        sdf = st.session_state.screener_df.copy()
        pass_count = int(sdf["FA_gecer"].sum()) if "FA_gecer" in sdf.columns else 0
        total_count = len(sdf)

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Taranan", total_count)
        sc2.metric("GEÇEN", pass_count)
        sc3.metric("Başarı Oranı", f"{pass_count/total_count*100:.1f}%")
        sc4.metric("Mod", fa_mode)

        # Filtre
        show_only_pass = st.checkbox("Sadece GEÇEN hisseleri göster", value=True)
        if show_only_pass:
            sdf = sdf[sdf["FA_gecer"] == True]

        col_show = [
            "ticker", "longName", "FA_gecer", "FA_puan", "FA_kapsam",
            "sector", "industry", "trailingPE", "forwardPE", "pegRatio",
            "returnOnEquity", "operatingMargins", "profitMargins",
            "debtToEquity", "revenueGrowth", "earningsGrowth", "marketCap",
        ]
        sdf_show = sdf[[c for c in col_show if c in sdf.columns]].copy()

        st.dataframe(
            sdf_show.style.background_gradient(
                subset=["FA_puan"] if "FA_puan" in sdf_show.columns else [],
                cmap="RdYlGn",
            ),
            use_container_width=True,
            height=460,
        )

        pass_list = sdf.loc[sdf["FA_gecer"] == True, "ticker"].tolist() if "FA_gecer" in sdf.columns else []
        if pass_list:
            st.markdown("")
            picked = st.selectbox("Geçen listesinden hisse seç ve teknik analize gönder:", pass_list)
            if st.button("➡️ Bu Hisseyi Analiz Et", type="primary", use_container_width=True):
                st.session_state.selected_ticker = picked
                st.session_state.ta_ran = True
                st.session_state.last_ta_ticker = normalize_ticker(picked, market)
                st.session_state.gemini_text = ""
                st.rerun()

# ────────────────────────────────────────────────────────────
# TAB 4 — HEATMAP
# ────────────────────────────────────────────────────────────
with tab_heatmap:
    st.markdown("""<div class='section-header'><h2>🔥 Sektörel Heatmap</h2></div>""", unsafe_allow_html=True)
    st.caption("Piyasadaki hisselerin performanslarını sektöre göre görselleştirir.")

    hm_col1, hm_col2 = st.columns([2, 1])
    with hm_col1:
        hm_source = st.radio(
            "Heatmap kaynağı",
            ["Evren (tümü)", "Sadece Screener'dan geçenler"],
            horizontal=True,
        )
    with hm_col2:
        hm_limit = st.number_input("Maks hisse sayısı", 10, 500, 100, 10)

    if st.button("📊 Heatmap Oluştur", type="primary", use_container_width=True):
        if hm_source == "Sadece Screener'dan geçenler" and not st.session_state.screener_df.empty:
            hm_tickers_raw = st.session_state.screener_df[
                st.session_state.screener_df.get("FA_gecer", False) == True
            ]["ticker"].tolist()
        else:
            hm_tickers_raw = universe

        hm_tickers = [normalize_ticker(t, market) for t in hm_tickers_raw[:int(hm_limit)]]

        with st.spinner(f"{len(hm_tickers)} hisse için veri çekiliyor..."):
            try:
                if len(hm_tickers) == 1:
                    df_all = yf.download(hm_tickers[0], period="1mo", interval="1d",
                                         auto_adjust=True, progress=False)
                    df_all = {hm_tickers[0]: _flatten_yf(df_all)}
                else:
                    raw_all = yf.download(hm_tickers, period="1mo", interval="1d",
                                          auto_adjust=True, group_by="ticker", progress=False)
                    df_all = {}
                    for t in hm_tickers:
                        try:
                            df_all[t] = _flatten_yf(raw_all[t].copy())
                        except Exception:
                            pass
            except Exception as e:
                st.error(f"Veri çekme hatası: {e}")
                df_all = {}

            hm_data = []
            for t, df_t in df_all.items():
                try:
                    df_t = df_t.dropna(subset=["Close"])
                    if len(df_t) < 2:
                        continue
                    c_last = float(df_t["Close"].iloc[-1])
                    c_1d = float(df_t["Close"].iloc[-2])
                    c_1wk = float(df_t["Close"].iloc[-6]) if len(df_t) >= 6 else float(df_t["Close"].iloc[0])
                    c_1mo = float(df_t["Close"].iloc[0])

                    sector = "Genel"
                    if not st.session_state.screener_df.empty:
                        row_m = find_screener_row(st.session_state.screener_df, t)
                        if row_m and row_m.get("sector"):
                            sector = str(row_m["sector"])

                    hm_data.append({
                        "Hisse": naked_ticker(t),
                        "Sektör": sector,
                        "1G %": round((c_last / c_1d - 1) * 100, 2),
                        "1H %": round((c_last / c_1wk - 1) * 100, 2),
                        "1A %": round((c_last / c_1mo - 1) * 100, 2),
                    })
                except Exception:
                    pass

        if not hm_data:
            st.error("Heatmap için yeterli veri alınamadı.")
        else:
            df_hm = pd.DataFrame(hm_data)

            for label, col in [("Günlük (1G)", "1G %"), ("Haftalık (1H)", "1H %"), ("Aylık (1A)", "1A %")]:
                st.subheader(label)
                abs_col = f"{col}_abs"
                df_hm[abs_col] = df_hm[col].abs().replace(0, 0.01)
                fig_hm = px.treemap(
                    df_hm,
                    path=[px.Constant("Tüm Pazar"), "Sektör", "Hisse"],
                    values=abs_col,
                    color=col,
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                    custom_data=["1G %", "1H %", "1A %"],
                )
                fig_hm.update_traces(
                    hovertemplate="<b>%{label}</b><br>Günlük: %{customdata[0]:.2f}%<br>Haftalık: %{customdata[1]:.2f}%<br>Aylık: %{customdata[2]:.2f}%",
                    textfont=dict(family="IBM Plex Mono", size=11),
                )
                fig_hm.update_layout(**PLOTLY_LAYOUT, height=420)
                st.plotly_chart(fig_hm, use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 5 — RAPOR
# ────────────────────────────────────────────────────────────
with tab_export:
    st.markdown("""<div class='section-header'><h2>📄 Rapor İndir</h2></div>""", unsafe_allow_html=True)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        inc_charts = st.checkbox("Grafikleri dahil et", value=True)
        inc_trades = st.checkbox("İşlem listesi (ilk 25)", value=True)
        inc_gemini = st.checkbox("Gemini analizini ekle", value=True)
    with col_r2:
        inc_pa = st.checkbox("Price Action verisi", value=True)
        inc_sentiment = st.checkbox("Haber duygu analizi", value=True)
        inc_ob = st.checkbox("Aşırı alım/spekülasyon analizi", value=True)

    if not KALEIDO_OK:
        st.warning(
            "⚠️ PDF'e grafik gömmek için `kaleido` gereklidir. "
            "`requirements.txt` içine `kaleido` ekleyip yeniden dağıtın. "
            "HTML rapor grafikleri her zaman içerir."
        )

    # Temel analiz satırı
    with st.spinner("Temel analiz verisi hazırlanıyor..."):
        f_single = fetch_fundamentals_generic(ticker, market=market)
        f_score_r, f_bd_r, f_pass_r = fundamental_score_row(f_single, fa_mode, thresholds)
        screener_row_r = find_screener_row(st.session_state.get("screener_df", pd.DataFrame()), ticker)
        fa_row_r = merge_fa_row(screener_row_r, f_single, {
            "mode": fa_mode, "score": f_score_r, "passed": f_pass_r,
            "ok_cnt": sum(1 for v in f_bd_r.values() if v.get("available") and v.get("ok")),
            "coverage": sum(1 for v in f_bd_r.values() if v.get("available")),
        })

    meta = dict(
        market=market, ticker=ticker, interval=interval, period=period,
        preset=preset_name, ema_fast=ema_fast, ema_slow=ema_slow,
        rsi_period=rsi_period, bb_period=bb_period, bb_std=bb_std,
        atr_period=atr_period, vol_sma=vol_sma,
    )

    gemini_export = st.session_state.gemini_text if inc_gemini else None
    pa_export = st.session_state.pa_pack if inc_pa else None
    sent_export = sentiment_result.get("summary", "") if inc_sentiment and sentiment_result else None
    ob_export = overbought_result if inc_ob else None

    # ── HTML Rapor ──
    def build_html(title, meta, cp, metrics, tp, rr_info, figs, fa_row, gemini, pa, sent, ob):
        def esc(x):
            return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        fig_html_parts = []
        first = True
        for name, fig in (figs or {}).items():
            fig_html_parts.append(
                f"<h3 style='color:#58a6ff;'>{esc(name)}</h3>"
                + fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False)
            )
            first = False

        cp_html = "".join(
            f"<li style='padding:4px 0; color:{'#3fb950' if v else '#f85149'};'>{'✅' if v else '❌'} {esc(k)}</li>"
            for k, v in cp.items()
        )

        bull = tp.get("bull")
        bear = tp.get("bear")
        levels_txt = ", ".join(f"{x:.2f}" for x in tp.get("levels", [])[:20])

        fa_html = ""
        if fa_row:
            for key, lbl in [
                ("ticker","Ticker"),("longName","Ad"),("FA_gecer","FA Geçti"),
                ("FA_puan","FA Puan"),("sector","Sektör"),("trailingPE","F/K"),
                ("forwardPE","İleri F/K"),("returnOnEquity","ROE"),
                ("operatingMargins","Faaliyet Marjı"),("debtToEquity","Borç/Özkaynak"),
                ("revenueGrowth","Gelir Büyümesi"),("marketCap","Piyasa Değeri"),
            ]:
                fa_html += f"<tr><td><b>{esc(lbl)}</b></td><td>{esc(fa_row.get(key,''))}</td></tr>"

        ob_html = ""
        if ob:
            det = "".join(f"<li>{esc(v)}</li>" for v in ob.get("details", {}).values())
            ob_html = f"""
            <div style="background:#1a1a2e;border:1px solid #30363d;border-radius:10px;padding:16px;margin:16px 0;">
                <h3 style="color:#e3b341;">📊 Aşırı Alım / Spekülasyon Analizi</h3>
                <p><b>Karar:</b> {esc(ob['verdict'])}</p>
                <p>Aşırı Alım: {ob['overbought_score']}/100 | Aşırı Satım: {ob['oversold_score']}/100 | Spekülasyon: {ob['speculation_score']}/100</p>
                <ul>{det}</ul>
            </div>"""

        gemini_html = ""
        if gemini:
            gemini_html = f"""<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:16px;margin:16px 0;">
                <h3 style="color:#58a6ff;">🤖 Gemini Analizi</h3>
                <pre style="white-space:pre-wrap;font-family:inherit;color:#cdd9e5;">{esc(gemini)}</pre>
            </div>"""

        sent_html = ""
        if sent:
            sent_html = f"""<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:16px;margin:16px 0;">
                <h3 style="color:#58a6ff;">📰 Haber Duygu Analizi</h3>
                <pre style="white-space:pre-wrap;font-family:inherit;color:#cdd9e5;">{esc(sent)}</pre>
            </div>"""

        return f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <title>{esc(title)}</title>
  <style>
    body{{font-family:'Segoe UI',Arial,sans-serif;background:#0d1117;color:#e6edf3;margin:24px;}}
    .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
    .card{{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:16px;}}
    h1,h2,h3{{margin:0 0 8px 0;}}
    h1{{color:#58a6ff;}}
    ul{{margin:8px 0 0 18px;}}
    table{{width:100%;border-collapse:collapse;}}
    td{{border-top:1px solid #21262d;padding:6px 8px;}}
    .muted{{color:#8b949e;font-size:12px;}}
    @media print{{body{{margin:10mm;background:#fff;color:#000;}}}}
  </style>
</head>
<body>
  <h1>{esc(title)}</h1>
  <p class="muted">
    {esc(time.strftime('%Y-%m-%d %H:%M:%S'))} |
    {esc(meta.get('market'))} · {esc(meta.get('ticker'))} · {esc(meta.get('interval'))} · {esc(meta.get('period'))} |
    Preset: {esc(meta.get('preset'))}
  </p>
  <div class="grid" style="margin-top:16px;">
    <div class="card">
      <h2>✅ Kontrol Noktaları</h2>
      <ul>{cp_html}</ul>
    </div>
    <div class="card">
      <h2>📊 Backtest</h2>
      <p>Toplam Getiri: {metrics.get('Toplam Getiri',0)*100:.1f}%</p>
      <p>Yıllık Getiri: {metrics.get('Yıllık Getiri',0)*100:.1f}%</p>
      <p>Sharpe: {metrics.get('Sharpe',0):.2f} | Sortino: {metrics.get('Sortino',0):.2f}</p>
      <p>Maks Drawdown: {metrics.get('Maks Drawdown',0)*100:.1f}%</p>
      <p>İşlem: {metrics.get('İşlem Sayısı',0)} | Kazanma: {metrics.get('Kazanma Oranı',0)*100:.1f}%</p>
      <p>Beta: {metrics.get('Beta',0):.2f} | Kelly: {metrics.get('Kelly % (Öneri)',0):.1f}%</p>
    </div>
  </div>
  {ob_html}
  <div class="card" style="margin-top:16px;">
    <h2>🎯 Hedef Fiyat Bandı</h2>
    <p>Baz: {tp.get('base',0):.2f}</p>
    <p>Yükseliş: {(bull[0] if bull else 0):.2f} → {(bull[1] if bull else 0):.2f} | R1: {bull[2] if bull and bull[2] else 'N/A'}</p>
    <p>Düşüş: {(bear[0] if bear else 0):.2f} → {(bear[1] if bear else 0):.2f} | S1: {bear[2] if bear and bear[2] else 'N/A'}</p>
    <p>Risk/Ödül: {fmt_rr(rr_info.get('rr'))}</p>
    <p class="muted">Seviyeler: {esc(levels_txt)}</p>
  </div>
  {gemini_html}
  {sent_html}
  <div class="card" style="margin-top:16px;">
    <h2>🔬 Temel Analiz Özeti</h2>
    <table>{fa_html}</table>
  </div>
  <div style="margin-top:20px;">
    {''.join(fig_html_parts)}
  </div>
</body>
</html>""".encode("utf-8")

    html_bytes = build_html(
        title=f"FA→TA Raporu — {ticker}",
        meta=meta, cp=checkpoints, metrics=metrics,
        tp=tp, rr_info=rr_info,
        figs=(figs_for_report if inc_charts else {}),
        fa_row=fa_row_r, gemini=gemini_export,
        pa=pa_export, sent=sent_export, ob=ob_export,
    )

    st.download_button(
        "⬇️ HTML Rapor İndir (Önerilen — Grafikler Dahil)",
        data=html_bytes,
        file_name=f"{ticker}_FA_TA_rapor.html",
        mime="text/html",
        use_container_width=True,
        type="primary",
    )

    st.divider()

    # ── PDF ──
    if not REPORTLAB_OK:
        st.warning("PDF oluşturmak için `reportlab` kütüphanesi gerekli. `requirements.txt` içine ekleyin.")
    else:
        if st.button("🧾 PDF Oluştur", use_container_width=True):
            with st.spinner("PDF oluşturuluyor..."):
                buf = BytesIO()
                c_pdf = canvas.Canvas(buf, pagesize=A4)
                W, H = A4
                left, top, bottom = 1.6*cm, H - 1.6*cm, 1.6*cm
                y = top

                def pdf_lines(lines, font_size=9, line_h=12):
                    nonlocal y, c_pdf
                    c_pdf.setFont("Helvetica", font_size)
                    for line in lines:
                        if y <= bottom:
                            c_pdf.showPage()
                            y = H - 1.6*cm
                        c_pdf.drawString(left, y, str(line)[:200])
                        y -= line_h

                def pdf_heading(text, size=12):
                    nonlocal y, c_pdf
                    if y < bottom + 30:
                        c_pdf.showPage()
                        y = H - 1.6*cm
                    c_pdf.setFont("Helvetica-Bold", size)
                    c_pdf.drawString(left, y, text[:100])
                    y -= size + 6

                pdf_heading(f"FA→TA Raporu — {ticker}", 16)
                pdf_lines([
                    f"Tarih: {time.strftime('%Y-%m-%d %H:%M')} | Piyasa: {market} | Interval: {interval} | Preset: {preset_name}",
                    f"Sinyal: {rec} | Skor: {score_val:.0f}/100",
                ])
                y -= 8

                pdf_heading("Kontrol Noktaları")
                pdf_lines([f"[{'OK' if v else 'NO'}] {k}" for k, v in checkpoints.items()])
                y -= 8

                if ob_export:
                    pdf_heading("Aşırı Alım / Spekülasyon")
                    pdf_lines([
                        f"Karar: {ob_export['verdict']}",
                        f"Aşırı Alım: {ob_export['overbought_score']}/100 | Spekülasyon: {ob_export['speculation_score']}/100",
                    ] + [f"- {v}" for v in ob_export.get("details", {}).values()])
                    y -= 8

                pdf_heading("Backtest Özeti")
                pdf_lines([
                    f"Toplam Getiri: {fmt_pct(metrics.get('Toplam Getiri'))} | Yıllık: {fmt_pct(metrics.get('Yıllık Getiri'))}",
                    f"Sharpe: {fmt_num(metrics.get('Sharpe'))} | Sortino: {fmt_num(metrics.get('Sortino'))} | Maks DD: {fmt_pct(metrics.get('Maks Drawdown'))}",
                    f"İşlem: {metrics.get('İşlem Sayısı',0)} | Kazanma: {fmt_pct(metrics.get('Kazanma Oranı'))} | Kelly: {fmt_num(metrics.get('Kelly % (Öneri)'),1)}%",
                ])
                y -= 8

                pdf_heading("Temel Analiz")
                if fa_row_r:
                    pdf_lines([f"{k}: {fa_row_r.get(k,'')}" for k in [
                        "ticker","longName","FA_gecer","FA_puan","sector",
                        "trailingPE","returnOnEquity","debtToEquity","revenueGrowth",
                    ]])
                y -= 8

                if sent_export:
                    pdf_heading("Haber Duygu Analizi")
                    pdf_lines(sent_export.splitlines())
                    y -= 8

                if gemini_export:
                    pdf_heading("Gemini Analizi")
                    pdf_lines(gemini_export.splitlines())
                    y -= 8

                if inc_trades and not tdf.empty:
                    pdf_heading("İşlem Listesi (İlk 25)")
                    cols = [c for c in ["entry_date","entry_price","exit_date","exit_price","exit_reason","pnl","getiri_%"] if c in tdf.columns]
                    pdf_lines([" | ".join(cols)])
                    for _, row_t in tdf.head(25).iterrows():
                        pdf_lines([" | ".join(str(row_t.get(c,""))[:14] for c in cols)])

                # Grafik gömme
                if inc_charts and KALEIDO_OK:
                    for name, fig in figs_for_report.items():
                        try:
                            img_b = fig.to_image(format="png", scale=2)
                            c_pdf.showPage()
                            c_pdf.setFont("Helvetica-Bold", 13)
                            c_pdf.drawString(left, H - 1.6*cm, f"Grafik: {name}")
                            ir = ImageReader(BytesIO(img_b))
                            usable_w = W - 3.2*cm
                            usable_h = H - 4*cm
                            c_pdf.drawImage(ir, left, 2*cm, width=usable_w, height=usable_h,
                                            preserveAspectRatio=True, anchor="c")
                        except Exception:
                            pass
                elif inc_charts and not KALEIDO_OK:
                    c_pdf.showPage()
                    c_pdf.setFont("Helvetica-Bold", 14)
                    c_pdf.drawString(left, H - 2*cm, "Grafikler PDF'e gömülemedi.")
                    c_pdf.setFont("Helvetica", 10)
                    c_pdf.drawString(left, H - 3*cm, "Neden: 'kaleido' kütüphanesi requirements.txt'te eksik.")
                    c_pdf.drawString(left, H - 4*cm, "Çözüm: HTML raporu indirip tarayıcıdan PDF olarak kaydedin.")

                c_pdf.save()
                buf.seek(0)
                pdf_data = buf.read()

            st.success("✅ PDF hazır!")
            st.download_button(
                "⬇️ PDF İndir",
                data=pdf_data,
                file_name=f"{ticker}_FA_TA_rapor.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
