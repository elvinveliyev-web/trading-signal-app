import os
import re
import json
import time
import base64
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

# =============================
# OPTIONAL BAĞIMLILIKLAR
# =============================
REPORTLAB_OK = True
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
except Exception:
    REPORTLAB_OK = False

KALEIDO_OK = True
try:
    import kaleido  # noqa: F401
except ImportError:
    KALEIDO_OK = False

FEEDPARSER_OK = True
try:
    import feedparser
except ImportError:
    FEEDPARSER_OK = False

# =============================
# SAYFA AYARLARI
# =============================
st.set_page_config(
    page_title="FA→TA Trading + AI",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# =============================
# ÖZEL CSS — Profesyonel Koyu Tema
# =============================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Ana arka plan */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 50%, #0a0e1a 100%);
        color: #e2e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
        border-right: 1px solid #1e2d40;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #60a5fa;
        font-weight: 600;
        letter-spacing: 0.05em;
        font-size: 0.75rem;
        text-transform: uppercase;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 4px;
        margin-top: 16px;
    }

    /* Metrik kartları */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 14px 16px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"]:hover {
        border-color: #3b82f6;
        box-shadow: 0 0 20px rgba(59,130,246,0.15);
        transform: translateY(-1px);
    }
    [data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Butonlar */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: 1px solid #3b82f6;
        box-shadow: 0 0 20px rgba(37,99,235,0.3);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        box-shadow: 0 0 30px rgba(59,130,246,0.5);
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.8);
        color: #94a3b8;
        border: 1px solid #334155;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #60a5fa;
        color: #60a5fa;
    }

    /* Sekmeler */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(17, 24, 39, 0.8);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #1e3a5f;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        padding: 8px 20px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a5f, #1e40af) !important;
        color: #60a5fa !important;
        box-shadow: 0 0 15px rgba(59,130,246,0.2);
    }

    /* Bildirimler */
    .stAlert {
        border-radius: 10px;
        border-left-width: 4px;
    }

    /* Divider */
    hr {
        border-color: #1e3a5f;
        margin: 12px 0;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        color: #94a3b8;
    }

    /* Başlık */
    h1 {
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2rem !important;
        letter-spacing: -0.02em;
    }
    h2 {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 1.1rem !important;
        letter-spacing: -0.01em;
    }
    h3 {
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.95rem !important;
    }

    /* Sinyal renk kartları */
    .signal-al {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 1px solid #16a34a;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        color: #4ade80;
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: 0 0 30px rgba(22,163,74,0.2);
    }
    .signal-sat {
        background: linear-gradient(135deg, #2d0a0a, #450a0a);
        border: 1px solid #dc2626;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        color: #f87171;
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: 0 0 30px rgba(220,38,38,0.2);
    }
    .signal-izle {
        background: linear-gradient(135deg, #1c1409, #332507);
        border: 1px solid #ca8a04;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        color: #facc15;
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: 0 0 30px rgba(202,138,4,0.2);
    }

    /* Kart bileşeni */
    .info-card {
        background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .info-card-title {
        color: #60a5fa;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }

    /* Checkpoint renkleri */
    .cp-ok {
        color: #4ade80;
        font-weight: 500;
    }
    .cp-fail {
        color: #f87171;
        font-weight: 500;
    }

    /* Chat mesajları */
    [data-testid="stChatMessage"] {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        margin-bottom: 8px;
    }

    /* Input alanları */
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid #334155;
        color: #e2e8f0;
        border-radius: 8px;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
    }

    /* Selectbox */
    .stSelectbox select, [data-baseweb="select"] {
        background: rgba(17, 24, 39, 0.8);
        border-color: #334155;
        color: #e2e8f0;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] {
        color: #3b82f6;
    }

    /* Caption */
    .stCaption {
        color: #64748b;
        font-size: 0.75rem;
    }

    /* Plotly grafik kenarlıkları */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #1e3a5f;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }

    /* Download butonu */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #065f46, #047857);
        color: #d1fae5;
        border: 1px solid #059669;
        border-radius: 8px;
        font-weight: 600;
    }
    .stDownloadButton > button:hover {
        box-shadow: 0 0 20px rgba(5,150,105,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# TEMEL DİZİN
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def pjoin(*parts) -> str:
    return os.path.join(BASE_DIR, *parts)


# =============================
# Evren Yükleyici
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
# Yardımcı Fonksiyonlar
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


def safe_float(x) -> float:
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan


# Toplu fundamental anahtar listesi
FUNDAMENTAL_KEYS = [
    "marketCap", "trailingPE", "forwardPE", "pegRatio",
    "priceToSalesTrailing12Months", "priceToBook", "returnOnEquity",
    "profitMargins", "operatingMargins", "debtToEquity",
    "revenueGrowth", "earningsGrowth", "freeCashflow", "currentPrice",
]


def _flatten_yf(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """MultiIndex sütunları düzelten, hem tek hem çoklu hisse indirmelerini destekleyen yardımcı."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        if ticker:
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


def fmt_pct(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "N/A"
        return f"{x * 100:.2f}%"
    except Exception:
        return "N/A"


def fmt_num(x: float, nd: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "N/A"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "N/A"


def fmt_rr(rr) -> str:
    if rr is None:
        return "N/A"
    if isinstance(rr, float) and not np.isfinite(rr):
        return "N/A"
    return f"1:{rr:.2f}"


def pct_dist(level: float, base: float) -> Optional[float]:
    if level is None or not np.isfinite(level) or base == 0:
        return None
    return (level / base - 1.0) * 100.0


# =============================
# Teknik Göstergeler
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
    return mid, mid + std_mult * sd, mid - std_mult * sd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / period, adjust=False).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def max_drawdown(eq: pd.Series) -> float:
    if eq is None or len(eq) == 0:
        return 0.0
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


# =============================
# Aşırı Alım / Spekülasyon Göstergeleri (Korelasyon Düzeltmeli)
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
        min_r = series.rolling(period).min()
        max_r = series.rolling(period).max()
        den = (max_r - min_r).replace(0, np.nan)
        stoch = 100 * (series - min_r) / den
        stoch = stoch.replace([np.inf, -np.inf], np.nan).fillna(50)
        k = stoch.rolling(smooth_k).mean()
        return k, k.rolling(smooth_d).mean()

    df["STOCH_RSI_K"], df["STOCH_RSI_D"] = stoch_rsi(df["RSI"])
    df["STOCH_OVERBOUGHT"] = (df["STOCH_RSI_K"] > 80).astype(int)

    df["VOLUME_DIR"] = np.sign(df["Volume"].diff()).fillna(0)
    df["PRICE_DIR"] = np.sign(df["Close"].diff()).fillna(0)
    df["WEAK_UPTREND"] = ((df["PRICE_DIR"] > 0) & (df["VOLUME_DIR"] < 0)).astype(int)

    return df


def detect_speculation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Korelasyon düzeltmeli spekülasyon tespiti.
    RSI ve Stochastic RSI yüksek koreleli olduğundan birlikte max 30 puan alır.
    """
    last = df.iloc[-1]
    result = {"overbought_score": 0, "oversold_score": 0, "speculation_score": 0, "details": {}}

    # --- Momentum grubu (RSI + Stochastic: maks 30 puan, korelasyon düzeltmeli) ---
    momentum_ob = 0
    momentum_os = 0
    if last["RSI"] > 70:
        momentum_ob += 20
        result["details"]["rsi"] = f"Aşırı alım (RSI: {last['RSI']:.1f})"
    elif last["RSI"] < 30:
        momentum_os += 20
        result["details"]["rsi"] = f"Aşırı satım (RSI: {last['RSI']:.1f})"

    if bool(last["STOCH_OVERBOUGHT"]) and last["RSI"] <= 70:
        momentum_ob += 10
        result["details"]["stoch"] = "Stokastik RSI aşırı alımda (RSI bağımsız)"
    elif bool(last["STOCH_OVERBOUGHT"]) and last["RSI"] > 70:
        momentum_ob += 5
        result["details"]["stoch"] = "Stokastik RSI + RSI her ikisi de aşırı alımda"

    result["overbought_score"] += min(30, momentum_ob)
    result["oversold_score"] += min(30, momentum_os)

    # --- Bollinger grubu (bağımsız: maks 20 puan) ---
    if bool(last["BB_OVERBOUGHT"]):
        result["overbought_score"] += 20
        result["details"]["bb"] = "Fiyat Bollinger üst bandının üzerinde"
    elif bool(last["BB_OVERSOLD"]):
        result["oversold_score"] += 20
        result["details"]["bb"] = "Fiyat Bollinger alt bandının altında"

    # --- Fiyat/EMA uzaklığı (bağımsız: maks 20 puan) ---
    if bool(last["PRICE_EXTREME"]):
        result["overbought_score"] += 20
        result["details"]["price_extreme"] = f"Fiyat EMA'dan çok uzak (EMA50: %{last['PRICE_TO_EMA50']:.1f})"

    # --- Hacim (bağımsız: maks 20 puan, spekülasyon) ---
    if bool(last["VOLUME_SPIKE"]):
        result["speculation_score"] += 20
        result["details"]["volume"] = "Ani hacim artışı (spekülasyon sinyali)"

    # --- Zayıf yükseliş (bağımsız: maks 10 puan) ---
    if bool(last["WEAK_UPTREND"]):
        result["speculation_score"] += 10
        result["details"]["weak_trend"] = "Fiyat yükselirken hacim düşüyor (momentum zayıflıyor)"

    result["overbought_score"] = min(100, result["overbought_score"])
    result["oversold_score"] = min(100, result["oversold_score"])
    result["speculation_score"] = min(100, result["speculation_score"])

    if result["overbought_score"] >= 50:
        result["verdict"] = "⛔ AŞIRI DEĞERLİ — SAT Bölgesi"
    elif result["oversold_score"] >= 50:
        result["verdict"] = "✅ AŞIRI DEĞERSİZ — AL Bölgesi"
    elif result["speculation_score"] >= 30:
        result["verdict"] = "⚠️ SPEKÜLATİF HAREKET — Dikkatli Olun"
    else:
        result["verdict"] = "🔵 NÖTR — Normal Değer Aralığı"

    return result


# =============================
# Özellik Oluşturucu
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
    return df


# =============================
# Piyasa Rejim Filtreleri
# =============================
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_spy_regime_ok() -> bool:
    # 2y yeterli — 10y gereksiz veri çekimi önlendi
    spy = yf.download("SPY", period="2y", interval="1d", auto_adjust=True, progress=False)
    spy = _flatten_yf(spy)
    if spy.empty or len(spy) < 200:
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


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_higher_tf_trend(ticker: str, higher_tf_interval: str = "1wk", ema_period: int = 200) -> bool:
    try:
        df = yf.download(ticker, period="5y", interval=higher_tf_interval, auto_adjust=True, progress=False)
        df = _flatten_yf(df)
        if df.empty or len(df) < min(ema_period, 50):
            return True
        df["EMA"] = ema(df["Close"], ema_period)
        last = df.iloc[-1]
        return bool(last["Close"] > last["EMA"])
    except Exception:
        return True


# =============================
# Strateji: Puanlama + Kontrol Noktaları
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
        "Piyasa Rejim Filtresi": bool(market_filter_ok),
        "Haftalık Trend Filtresi": bool(higher_tf_filter_ok),
        "Likidite (Hacim > HacimOrt)": bool(last["Volume"] > last["VOL_SMA"]) if pd.notna(last["VOL_SMA"]) else False,
        "Trend (Fiyat>EMA200 & EMA50>EMA200)": bool((last["Close"] > last["EMA200"]) and (last["EMA50"] > last["EMA200"])) if pd.notna(last["EMA200"]) else False,
        f"RSI > {cfg['rsi_entry_level']}": bool(last["RSI"] > cfg["rsi_entry_level"]) if pd.notna(last["RSI"]) else False,
        "MACD Hist > 0": bool(last["MACD_hist"] > 0) if pd.notna(last["MACD_hist"]) else False,
        f"ATR% < {cfg['atr_pct_max']:.1%}": bool((last["ATR"] / last["Close"]) < cfg["atr_pct_max"]) if pd.notna(last["ATR"]) else False,
        "Bollinger (Fiyat>BB_Orta veya Kırılım)": bool((last["Close"] > last["BB_mid"]) or (last["Close"] > last["BB_upper"])) if pd.notna(last["BB_mid"]) else False,
        "OBV > OBV_EMA": bool(last["OBV"] > last["OBV_EMA"]) if pd.notna(last["OBV_EMA"]) else False,
    }
    return df, cp


# =============================
# Backtest (Sadece Long) + Gelişmiş Çıkışlar
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
                cash += gross - fee
                shares -= sell_shares
                half_sold = True
                stop = max(stop, entry_price)
                if trades:
                    trades[-1]["pnl"] = cash + (shares * price * (1 - slippage)) - trades[-1]["equity_before"]

            if exit_sig.iloc[i] == 1 or stop_hit or time_stop_hit:
                sell_price = price * (1 - slippage)
                gross = shares * sell_price
                fee = gross * commission
                cash += gross - fee

                trades[-1]["exit_date"] = date
                trades[-1]["exit_price"] = sell_price
                trades[-1]["exit_reason"] = "STOP" if stop_hit else ("ZAMAN_STOPU" if time_stop_hit else "KURAL_ÇIKIŞI")
                trades[-1]["pnl"] = cash - trades[-1]["equity_before"]

                shares = 0.0
                stop = np.nan
                entry_price = np.nan
                target_price = np.nan
                bars_held = 0
                half_sold = False

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
                    entry_price = buy_price
                    stop = buy_price - cfg["atr_stop_mult"] * float(row["ATR"])
                    target_price = buy_price + (tp_mult * cfg["atr_stop_mult"] * float(row["ATR"]))
                    bars_held = 0
                    half_sold = False
                    trades.append({
                        "entry_date": date, "entry_price": buy_price,
                        "exit_date": None, "exit_price": None,
                        "exit_reason": None, "shares": shares,
                        "equity_before": equity, "pnl": None,
                    })

        position_value = shares * price * (1 - slippage)
        equity = cash + position_value
        equity_curve.append((date, equity))

    eq = pd.Series(
        [v for _, v in equity_curve],
        index=[d for d, _ in equity_curve],
        name="equity",
    ).astype(float).replace([np.inf, -np.inf], np.nan).dropna()

    ret = eq.pct_change().dropna()
    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    ann_return = (1 + total_return) ** (252 / max(1, len(ret))) - 1 if len(ret) > 0 else 0.0
    ann_vol = float(ret.std() * np.sqrt(252)) if len(ret) > 1 else 0.0
    rf_daily = (1 + float(risk_free_annual)) ** (1 / 252) - 1
    excess = ret - rf_daily

    sharpe = float((excess.mean() * 252) / (excess.std() * np.sqrt(252))) if len(ret) > 1 and excess.std() > 0 else 0.0
    downside = excess.copy()
    downside[downside > 0] = 0
    downside_dev = float(np.sqrt((downside ** 2).mean()) * np.sqrt(252)) if len(downside) > 1 else 0.0
    sortino = float((excess.mean() * 252) / downside_dev) if downside_dev > 0 else 0.0
    mdd = max_drawdown(eq)
    calmar = float(ann_return / abs(mdd)) if mdd < 0 else 0.0

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
        else:
            beta, alpha, info_ratio = 1.0, 0.0, 0.0
    else:
        beta, alpha, info_ratio = 1.0, 0.0, 0.0

    peak = eq.cummax()
    drawdown_pct = (eq - peak) / peak
    ulcer_index = np.sqrt((drawdown_pct ** 2).mean()) if len(drawdown_pct) > 0 else 0.0

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tdf["pnl"] = tdf["pnl"].astype(float)
        tdf["getiri_%"] = (tdf["pnl"] / tdf["equity_before"]) * 100
        tdf["elde_tutma_gün"] = (pd.to_datetime(tdf["exit_date"]) - pd.to_datetime(tdf["entry_date"])).dt.days

    profit_factor = 0.0
    if not tdf.empty and "pnl" in tdf.columns:
        gross_profit = float(tdf.loc[tdf["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-tdf.loc[tdf["pnl"] < 0, "pnl"].sum())
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")

    # Kelly — yarı-Kelly ile %10 sınırı (eski: %25 — sermaye riski taşıyordu)
    kelly = 0.0
    if not tdf.empty and len(tdf) > 5:
        win_rate = float((tdf["pnl"] > 0).mean())
        avg_win = float(tdf.loc[tdf["pnl"] > 0, "pnl"].mean()) if win_rate > 0 else 0.0
        avg_loss = float(-tdf.loc[tdf["pnl"] < 0, "pnl"].mean()) if win_rate < 1 else 0.0
        if avg_loss > 0 and 0 < win_rate < 1:
            b = avg_win / avg_loss
            p = win_rate
            full_kelly = (p * b - (1 - p)) / b
            kelly = max(0.0, min(full_kelly * 0.5, 0.10))  # Yarı-Kelly, maks %10

    metrics = {
        "Toplam Getiri": float(total_return),
        "Yıllık Getiri": float(ann_return),
        "Yıllık Volatilite": float(ann_vol),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Calmar": float(calmar),
        "Maks Düşüş": float(mdd),
        "İşlem Sayısı": int(len(tdf)) if not tdf.empty else 0,
        "Kazanma Oranı": float((tdf["pnl"] > 0).mean()) if not tdf.empty else 0.0,
        "Kâr Faktörü": float(profit_factor) if np.isfinite(profit_factor) else 999.0,
        "Beta": float(beta),
        "Alpha": float(alpha),
        "Bilgi Oranı": float(info_ratio),
        "Ulcer Endeksi": float(ulcer_index),
        "Kelly % (Öneri)": float(kelly * 100),
    }
    return eq, tdf, metrics


# =============================
# Temel Analiz
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

    # Toplu safe_float dönüşümü — tekrar eden çağrılar kaldırıldı
    out = {k: safe_float(info.get(k)) for k in FUNDAMENTAL_KEYS}
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
    b = {}
    score = 0.0
    total_w = 0.0
    avail_cnt = 0
    ok_cnt = 0

    def check(name, condition, weight, available):
        nonlocal score, total_w, avail_cnt, ok_cnt
        passed = bool(condition) if available else False
        b[name] = {"ok": passed, "weight": weight, "available": bool(available)}
        if available:
            total_w += weight
            avail_cnt += 1
            if passed:
                score += weight
                ok_cnt += 1

    A = pd.notna

    if mode == "Kalite":
        check("ROE", A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"], 20, A(row["returnOnEquity"]))
        check("Faaliyet Marjı", A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"], 15, A(row["operatingMargins"]))
        check("Borç/Özkaynak", A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"], 20, A(row["debtToEquity"]))
        check("Net Kâr Marjı", A(row["profitMargins"]) and row["profitMargins"] >= thresholds["profit_margin"], 15, A(row["profitMargins"]))
        check("Serbest Nakit Akışı", A(row["freeCashflow"]) and row["freeCashflow"] > 0, 30, A(row["freeCashflow"]))
    elif mode == "Değer":
        check("İleri F/K", A(row["forwardPE"]) and row["forwardPE"] <= thresholds["fpe"], 30, A(row["forwardPE"]))
        check("PEG", A(row["pegRatio"]) and row["pegRatio"] <= thresholds["peg"], 20, A(row["pegRatio"]))
        check("F/S", A(row["priceToSalesTrailing12Months"]) and row["priceToSalesTrailing12Months"] <= thresholds["ps"], 20, A(row["priceToSalesTrailing12Months"]))
        check("F/DD", A(row["priceToBook"]) and row["priceToBook"] <= thresholds["pb"], 15, A(row["priceToBook"]))
        check("ROE", A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"], 15, A(row["returnOnEquity"]))
    else:  # Büyüme
        check("Ciro Büyümesi", A(row["revenueGrowth"]) and row["revenueGrowth"] >= thresholds["rev_g"], 35, A(row["revenueGrowth"]))
        check("Kâr Büyümesi", A(row["earningsGrowth"]) and row["earningsGrowth"] >= thresholds["earn_g"], 35, A(row["earningsGrowth"]))
        check("Faaliyet Marjı", A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"], 15, A(row["operatingMargins"]))
        check("Borç/Özkaynak", A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"], 15, A(row["debtToEquity"]))

    score_pct = (score / total_w) * 100 if total_w > 0 else 0.0
    min_coverage = int(thresholds.get("min_coverage", 3))
    min_ok = int(thresholds["min_ok"])
    pass_bool = (score_pct >= thresholds["min_score"]) and (ok_cnt >= min_ok) and (avail_cnt >= min_coverage)
    return float(score_pct), b, bool(pass_bool)


# =============================
# Hedef Fiyat Bandı
# =============================
def local_levels(close: pd.Series, lookback: int = 120) -> List[float]:
    """
    KDE tepe noktası yaklaşımı ile destek/direnç seviyeleri.
    Ham yüzdelik dilim yerine yerel fiyat yoğunluğu kullanılır.
    """
    s = close.tail(lookback).dropna()
    if len(s) < 10:
        return []
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(s.values, bw_method=0.1)
        x_grid = np.linspace(float(s.min()), float(s.max()), 200)
        density = kde(x_grid)
        # Yerel maksimum noktaları bul
        peaks = []
        for i in range(1, len(density) - 1):
            if density[i] > density[i - 1] and density[i] > density[i + 1]:
                peaks.append(float(x_grid[i]))
        if not peaks:
            raise ValueError("Tepe yok")
        return sorted([round(p, 2) for p in peaks])
    except Exception:
        # scipy yoksa swing high/low ile geri dön
        swing_h = []
        swing_l = []
        arr = s.values
        for i in range(2, len(arr) - 2):
            if arr[i] == max(arr[i - 2: i + 3]):
                swing_h.append(round(float(arr[i]), 2))
            if arr[i] == min(arr[i - 2: i + 3]):
                swing_l.append(round(float(arr[i]), 2))
        levels = sorted(set(swing_h + swing_l))
        return levels if levels else sorted([round(float(x), 2) for x in np.quantile(arr, [0.2, 0.4, 0.6, 0.8])])


def target_price_band(df: pd.DataFrame):
    last = df.iloc[-1]
    px_close = float(last["Close"])
    atrv = float(last["ATR"]) if pd.notna(last.get("ATR", np.nan)) else np.nan
    if not np.isfinite(atrv) or atrv <= 0:
        return {"base": px_close, "bull": None, "bear": None, "levels": local_levels(df["Close"])}

    lv = local_levels(df["Close"])
    above = [x for x in lv if x >= px_close]
    below = [x for x in lv if x <= px_close]
    r1 = min(above) if above else None
    s1 = max(below) if below else None

    return {
        "base": px_close,
        "bull": (px_close + 1.5 * atrv, px_close + 3.0 * atrv, r1),
        "bear": (px_close - 1.5 * atrv, px_close - 3.0 * atrv, s1),
        "levels": lv,
    }


# =============================
# Canlı Fiyat
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
# Gemini Yardımcıları — GÜVENLİ (header ile anahtar)
# =============================
def _get_secret(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name, "")
        return str(v).strip() if v else default
    except Exception:
        return default


def _http_post_json(url: str, headers: dict, payload: dict, timeout: int = 60) -> dict:
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"error": {"message": f"JSON dışı yanıt (status={r.status_code})", "text": r.text[:500]}}
    if r.status_code >= 400 and "error" not in data:
        data["error"] = {"message": f"HTTP {r.status_code}", "text": str(data)[:500]}
    return data


def _extract_gemini_text(resp: dict) -> str:
    if not isinstance(resp, dict):
        return str(resp)
    if resp.get("error"):
        return f"Gemini API hatası: {resp['error'].get('message', '')}"
    cands = resp.get("candidates") or []
    if not cands:
        return "Gemini: boş yanıt döndü (candidates yok)."
    parts = (cands[0].get("content") or {}).get("parts") or []
    texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
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
        return "❌ GEMINI_API_KEY bulunamadı. Streamlit Cloud > Settings > Secrets içine GEMINI_API_KEY=... ekleyin."

    # GÜVENLİ: Anahtar artık URL'de değil, header'da taşınıyor
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
        "generationConfig": {"temperature": float(temperature), "maxOutputTokens": int(max_output_tokens)},
    }
    resp = _http_post_json(url, headers, payload, timeout=90)
    return _extract_gemini_text(resp)


# =============================
# Haber Duygu Analizi
# =============================
@st.cache_data(ttl=30 * 60, show_spinner=False)
def get_news_sentiment(
    ticker: str,
    company_name: str = "",
    gemini_model: str = "gemini-1.5-flash",
    gemini_temp: float = 0.2,
) -> Dict[str, Any]:
    if not FEEDPARSER_OK:
        return {"error": "feedparser kütüphanesi eksik ('pip install feedparser').", "sentiment": None, "summary": ""}
    try:
        query = f"{company_name} stock" if company_name else f"{ticker} stock"
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        if not feed.entries:
            return {"error": "Haber bulunamadı.", "sentiment": None, "summary": ""}

        titles = [e.title for e in feed.entries[:10]]
        prompt = f"""Aşağıdaki haber başlıklarının duygu analizini yap (pozitif, negatif, nötr).
Sonuçları şu formatta ver:
- Pozitif: X haber
- Negatif: Y haber
- Nötr: Z haber
- Bileşik skor: (örneğin 0.35)
- Kısa özet (2 cümle)

Başlıklar:
{chr(10).join([f'- {t}' for t in titles])}
"""
        response = gemini_generate_text(prompt=prompt, model=gemini_model, temperature=gemini_temp, max_output_tokens=512)

        pos = int(m.group(1)) if (m := re.search(r"Pozitif:?\s*(\d+)", response, re.IGNORECASE)) else 0
        neg = int(m.group(1)) if (m := re.search(r"Negatif:?\s*(\d+)", response, re.IGNORECASE)) else 0
        neu = int(m.group(1)) if (m := re.search(r"Nötr:?\s*(\d+)", response, re.IGNORECASE)) else 0
        total = pos + neg + neu
        compound = (pos - neg) / total if total > 0 else 0

        return {
            "error": None,
            "sentiment": compound,
            "summary": response,
            "pos": pos / total if total > 0 else 0,
            "neg": neg / total if total > 0 else 0,
            "neu": neu / total if total > 0 else 0,
            "titles": titles[:3],
        }
    except Exception as e:
        return {"error": str(e), "sentiment": None, "summary": ""}


# =============================
# Price Action
# =============================
def _swing_points(high: pd.Series, low: pd.Series, left: int = 2, right: int = 2):
    hs, ls = [], []
    n = len(high)
    for i in range(left, n - right):
        hwin = high.iloc[i - left: i + right + 1]
        lwin = low.iloc[i - left: i + right + 1]
        if high.iloc[i] == hwin.max():
            hs.append((high.index[i], float(high.iloc[i])))
        if low.iloc[i] == lwin.min():
            ls.append((low.index[i], float(low.iloc[i])))
    return hs, ls


def price_action_pack(df: pd.DataFrame, last_n: int = 20) -> dict:
    use = df.tail(last_n).copy()
    if use.empty or len(use) < 10:
        return {"note": "yetersiz_bar", "last_n": int(len(use))}

    o, h, l, c = use["Open"].astype(float), use["High"].astype(float), use["Low"].astype(float), use["Close"].astype(float)
    swing_highs, swing_lows = _swing_points(h, l, left=2, right=2)

    recent_highs = [v for _, v in swing_highs[-5:]] if swing_highs else []
    recent_lows = [v for _, v in swing_lows[-5:]] if swing_lows else []
    res = max(recent_highs) if recent_highs else float(h.max())
    sup = min(recent_lows) if recent_lows else float(l.min())

    last_close = float(c.iloc[-1])
    prev_close = float(c.iloc[-2]) if len(c) >= 2 else last_close

    vol_ok = None
    if "Volume" in use.columns:
        vol = use["Volume"].astype(float)
        vol_sma_val = float(vol.rolling(10).mean().iloc[-1]) if len(vol) >= 10 else float(vol.mean())
        vol_ok = float(vol.iloc[-1]) > vol_sma_val if np.isfinite(vol_sma_val) else None

    q80 = float(np.quantile(c.values, 0.80))
    q20 = float(np.quantile(c.values, 0.20))
    impulse_up = (c.diff().tail(3) > 0).all() and (last_close >= q80)
    impulse_dn = (c.diff().tail(3) < 0).all() and (last_close <= q20)

    ob = None
    if impulse_up:
        for i in range(len(use) - 4, -1, -1):
            if c.iloc[i] < o.iloc[i]:
                ob = {"type": "yükseliş_order_block", "index": str(use.index[i]),
                      "open": float(o.iloc[i]), "close": float(c.iloc[i])}
                break
    elif impulse_dn:
        for i in range(len(use) - 4, -1, -1):
            if c.iloc[i] > o.iloc[i]:
                ob = {"type": "düşüş_order_block", "index": str(use.index[i]),
                      "open": float(o.iloc[i]), "close": float(c.iloc[i])}
                break

    return {
        "last_n": int(len(use)),
        "destek": sup,
        "direnç": res,
        "yükseliş_kırılımı": bool((last_close > res) and (prev_close <= res)),
        "düşüş_kırılımı": bool((last_close < sup) and (prev_close >= sup)),
        "hacim_onayı": (None if vol_ok is None else bool(vol_ok)),
        "son_bar": {"t": str(use.index[-1]), "open": float(o.iloc[-1]),
                    "high": float(h.iloc[-1]), "low": float(l.iloc[-1]), "close": last_close},
        "swing_highs": [{"t": str(t), "p": float(p)} for t, p in swing_highs[-6:]],
        "swing_lows": [{"t": str(t), "p": float(p)} for t, p in swing_lows[-6:]],
        "order_block": ob,
    }


def df_snapshot_for_llm(df: pd.DataFrame, n: int = 30) -> dict:
    """
    LLM'e gönderilecek veri — 30 satır + özet istatistikler.
    Önceki 140 satırlık gönderim gereksiz token tüketiyordu.
    """
    use_cols = ["Open", "High", "Low", "Close", "Volume", "EMA50", "EMA200",
                "RSI", "MACD", "MACD_hist", "BB_upper", "BB_lower", "ATR",
                "ATR_PCT", "SCORE", "ENTRY", "EXIT"]
    cols = [c for c in use_cols if c in df.columns]
    tail = df[cols].tail(n).copy()
    tail.index = tail.index.astype(str)

    summary = {
        "rsi_son": round(float(df["RSI"].iloc[-1]), 2) if "RSI" in df.columns else None,
        "rsi_5gun_ort": round(float(df["RSI"].tail(5).mean()), 2) if "RSI" in df.columns else None,
        "trend": "yukari" if df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1] else "asagi",
        "atr_pct_son": round(float(df["ATR_PCT"].iloc[-1]) * 100, 3) if "ATR_PCT" in df.columns else None,
        "hacim_oran": round(float(df["VOL_RATIO"].iloc[-1]), 2) if "VOL_RATIO" in df.columns else None,
        "bb_genislik": round(float(df["BB_WIDTH"].iloc[-1]) * 100, 3) if "BB_WIDTH" in df.columns else None,
    }

    return {"ozet": summary, "son_n_bar": n, "satirlar": tail.to_dict(orient="records")}


# =============================
# Ön Tanımlı Konfigürasyonlar
# =============================
PRESETS = {
    "Defansif": {
        "rsi_entry_level": 52, "rsi_exit_level": 46,
        "atr_pct_max": 0.06, "atr_stop_mult": 2.0,
        "time_stop_bars": 15, "take_profit_mult": 2.5,
    },
    "Dengeli": {
        "rsi_entry_level": 50, "rsi_exit_level": 45,
        "atr_pct_max": 0.08, "atr_stop_mult": 1.5,
        "time_stop_bars": 10, "take_profit_mult": 2.0,
    },
    "Agresif": {
        "rsi_entry_level": 48, "rsi_exit_level": 43,
        "atr_pct_max": 0.10, "atr_stop_mult": 1.2,
        "time_stop_bars": 7, "take_profit_mult": 1.5,
    },
}

# =============================
# Screener Yardımcıları
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
    return m.iloc[0].drop(labels=["_tk", "_tk_naked"], errors="ignore").to_dict()


def merge_fa_row(screener_row, fundamentals, fa_eval) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if fundamentals:
        out.update(fundamentals)
    if screener_row:
        out.update(screener_row)
    if fa_eval:
        out.update({
            "FA_mod": fa_eval.get("mode"),
            "FA_skor": fa_eval.get("score"),
            "FA_gecti": fa_eval.get("passed"),
            "FA_ok_sayisi": fa_eval.get("ok_cnt"),
            "FA_kapsam": fa_eval.get("coverage"),
        })
    return out


# =============================
# RR Hesaplaması
# =============================
def rr_from_atr_stop(latest_row: pd.Series, tp_dict: dict, cfg: dict):
    close = float(latest_row["Close"])
    atrv = float(latest_row.get("ATR", np.nan)) if pd.notna(latest_row.get("ATR", np.nan)) else np.nan
    if not np.isfinite(atrv) or atrv <= 0:
        return {"rr": None, "stop": None, "risk": None, "reward": None}

    stop = close - float(cfg["atr_stop_mult"]) * atrv
    risk = close - stop
    tp_mult = cfg.get("take_profit_mult", 2.0)
    target = close + (tp_mult * cfg["atr_stop_mult"] * atrv)
    reward = target - close

    if risk <= 0 or reward <= 0:
        return {"rr": None, "stop": stop, "risk": risk, "reward": reward}
    return {
        "rr": float(reward / risk),
        "stop": float(stop),
        "risk": float(risk),
        "reward": float(reward),
        "hedef_tipi": f"Backtest TP (TP çarpanı: {tp_mult:.1f} x ATR)",
    }


# =============================
# Veri Yükleyici (Önbellekli)
# =============================
@st.cache_data(show_spinner=False)
def load_data_cached(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    return _flatten_yf(df)


# =============================
# HTML RAPOR
# =============================
def build_html_report(
    title: str, meta: dict, checkpoints: dict, metrics: dict,
    tp: dict, rr_info: dict, figs: Dict[str, go.Figure],
    fa_row=None, gemini_insight=None, pa_pack=None,
    sentiment_summary=None, overbought_result=None,
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
    bull, bear = tp.get("bull"), tp.get("bear")
    levels_txt = ", ".join([f"{x:.2f}" for x in (tp.get("levels", []) or [])[:120]])

    fa_rows_html = ""
    if fa_row:
        for key, label in [
            ("ticker", "Ticker"), ("longName", "Ad"), ("FA_gecti", "FA Geçti"),
            ("FA_skor", "FA Skoru"), ("FA_ok_sayisi", "FA OK Sayısı"), ("FA_kapsam", "FA Kapsam"),
            ("sector", "Sektör"), ("industry", "Sektör Detayı"),
            ("trailingPE", "F/K"), ("forwardPE", "İleri F/K"), ("pegRatio", "PEG"),
            ("priceToSalesTrailing12Months", "F/S"), ("priceToBook", "F/DD"),
            ("returnOnEquity", "ROE"), ("operatingMargins", "Faaliyet Marjı"),
            ("profitMargins", "Net Kâr Marjı"), ("debtToEquity", "Borç/Özkaynak"),
            ("revenueGrowth", "Ciro Büyümesi"), ("earningsGrowth", "Kâr Büyümesi"), ("marketCap", "Piyasa Değeri"),
        ]:
            fa_rows_html += f"<tr><td><b>{esc(label)}</b></td><td>{esc(fa_row.get(key,''))}</td></tr>"

    ob_html = ""
    if overbought_result:
        details_html = "".join([f"<li>{esc(v)}</li>" for v in overbought_result.get("details", {}).values()])
        ob_html = f"""<div class="card"><h2>📊 Aşırı Alım / Spekülasyon Analizi</h2>
        <b>Karar:</b> {esc(overbought_result['verdict'])}<br>
        <b>Aşırı Alım:</b> {overbought_result['overbought_score']}/100 &nbsp;
        <b>Aşırı Satım:</b> {overbought_result['oversold_score']}/100 &nbsp;
        <b>Spekülasyon:</b> {overbought_result['speculation_score']}/100<br>
        <ul>{details_html}</ul></div>"""

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#0d1117;color:#e2e8f0}}
.card{{border:1px solid #1e3a5f;border-radius:10px;padding:14px;margin:10px 0;background:#111827}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
h1{{background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:700}}
h2{{color:#60a5fa;font-size:1rem}}
table{{width:100%;border-collapse:collapse}}
td{{border-top:1px solid #1e3a5f;padding:6px 8px;color:#e2e8f0}}
ul{{margin:8px 0 0 18px}}
pre{{white-space:pre-wrap;font-family:monospace;font-size:12px;color:#94a3b8}}
@media print{{body{{background:white;color:black}}}}
</style></head><body>
<h1>{esc(title)}</h1>
<div style="color:#64748b;font-size:12px">
  {esc(time.strftime('%Y-%m-%d %H:%M:%S'))} | {esc(meta.get('market'))} / {esc(meta.get('ticker'))} | {esc(meta.get('interval'))} / {esc(meta.get('period'))} | Mod: {esc(meta.get('preset'))}
</div>
<div class="grid" style="margin-top:14px">
<div class="card"><h2>Kontrol Noktaları</h2><ul>{cp_list}</ul></div>
<div class="card"><h2>Backtest Özeti</h2>
  Toplam Getiri: {metrics.get('Toplam Getiri',0)*100:.1f}% | Yıllık: {metrics.get('Yıllık Getiri',0)*100:.1f}%<br>
  Sharpe: {metrics.get('Sharpe',0):.2f} | Maks DD: {metrics.get('Maks Düşüş',0)*100:.1f}%<br>
  İşlemler: {metrics.get('İşlem Sayısı',0)} | Kazanma: {metrics.get('Kazanma Oranı',0)*100:.1f}%<br>
  Beta: {metrics.get('Beta',0):.2f} | Alpha: {metrics.get('Alpha',0):.2f} | Kelly: {metrics.get('Kelly % (Öneri)',0):.1f}%
</div></div>
{ob_html}
<div class="card"><h2>Hedef Fiyat Bandı</h2>
  Baz: {tp.get('base',0):.2f} |
  Yükseliş: {(bull[0] if bull else 0):.2f} → {(bull[1] if bull else 0):.2f} | R1: {(bull[2] if bull else 'N/A')} |
  Düşüş: {(bear[0] if bear else 0):.2f} → {(bear[1] if bear else 0):.2f} | S1: {(bear[2] if bear else 'N/A')} |
  RR: {fmt_rr(rr_info.get('rr'))}<br>
  <small style="color:#64748b">Seviyeler: {esc(levels_txt)}</small>
</div>
{'<div class="card"><h2>Gemini AI Analizi</h2><pre>' + esc(gemini_insight) + '</pre></div>' if gemini_insight else ''}
{'<div class="card"><h2>Haber Duygu Analizi</h2><pre>' + esc(sentiment_summary) + '</pre></div>' if sentiment_summary else ''}
{'<div class="card"><h2>Price Action Pack</h2><pre>' + esc(json.dumps(pa_pack, ensure_ascii=False, indent=2)) + '</pre></div>' if pa_pack else ''}
<div class="card"><h2>Temel Analiz Özeti</h2><table>{fa_rows_html}</table></div>
{''.join(fig_blocks)}
</body></html>"""
    return html.encode("utf-8")


# =============================
# SESSION STATE BAŞLATMA
# =============================
for key, default in [
    ("screener_df", pd.DataFrame()),
    ("selected_ticker", None),
    ("ai_messages", [{"role": "assistant", "content": "Merhaba! Hisse hakkında ne sormak istersiniz? Örneğin: 'Riskler neler?', 'Hedef fiyat nedir?', 'Ne zaman çıkmalıyım?'"}]),
    ("ta_ran", False),
    ("gemini_text", ""),
    ("pa_pack", {}),
    ("sentiment_summary", ""),
    ("last_market", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# =============================
# BAŞLIK
# =============================
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.title("📈 FA→TA Trading Platformu")
    st.caption("Temel analizle evreni daralt → Teknik analizle giriş/çıkış zamanla → AI ile içgörü al")
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("v2.0 | Eğitim Amaçlı")

st.divider()


# =============================
# KENAR ÇUBUĞU
# =============================
with st.sidebar:
    st.markdown("### 🌍 Piyasa Seçimi")
    market = st.selectbox(
        "Borsa",
        ["USA", "BIST"],
        index=0,
        help="USA: ABD borsaları | BIST: Borsa İstanbul",
    )

    # Piyasa değişiminde session state sıfırlama
    if st.session_state.get("last_market") != market:
        st.session_state.screener_df = pd.DataFrame()
        st.session_state.selected_ticker = None
        st.session_state.last_market = market

    usa_bucket = None
    if market == "USA":
        usa_bucket = st.selectbox(
            "USA Evreni",
            ["S&P 500", "Nasdaq 100"],
            index=0,
        )

    st.divider()

    # ---- TEMEL ANALİZ ----
    st.markdown("### 🔬 1) Temel Analiz Screener")
    use_fa = st.checkbox("Temel analiz filtresi", value=True)
    fa_mode = st.selectbox(
        "FA Modu",
        ["Kalite", "Değer", "Büyüme"],
        index=0,
        disabled=not use_fa,
        help="Kalite: Karlılık + Düşük borç | Değer: Ucuz hisse | Büyüme: Yüksek büyüme",
    )

    with st.expander("📐 FA Eşik Ayarları", expanded=False):
        roe = st.slider("Min ROE", 0.0, 0.40, 0.15, 0.01, disabled=not use_fa)
        op_margin = st.slider("Min Faaliyet Marjı", 0.0, 0.40, 0.10, 0.01, disabled=not use_fa)
        profit_margin = st.slider("Min Net Kâr Marjı", 0.0, 0.40, 0.08, 0.01, disabled=not use_fa)
        dte = st.slider("Maks Borç/Özkaynak", 0.0, 3.0, 1.0, 0.05, disabled=not use_fa)
        fpe = st.slider("Maks İleri F/K", 0.0, 60.0, 20.0, 1.0, disabled=not use_fa)
        peg = st.slider("Maks PEG", 0.0, 5.0, 1.5, 0.1, disabled=not use_fa)
        ps = st.slider("Maks F/S", 0.0, 30.0, 6.0, 0.5, disabled=not use_fa)
        pb = st.slider("Maks F/DD", 0.0, 30.0, 6.0, 0.5, disabled=not use_fa)
        rev_g = st.slider("Min Ciro Büyümesi", 0.0, 0.50, 0.10, 0.01, disabled=not use_fa)
        earn_g = st.slider("Min Kâr Büyümesi", 0.0, 0.50, 0.10, 0.01, disabled=not use_fa)
        min_score = st.slider("Min FA Skoru", 0, 100, 60, 1, disabled=not use_fa)
        min_ok = st.slider("Min OK Sayısı", 1, 5, 3, 1, disabled=not use_fa)
        min_coverage = st.slider("Min Kapsam (NaN dışı)", 1, 5, 3, 1, disabled=not use_fa)

    thresholds = {
        "roe": roe, "op_margin": op_margin, "profit_margin": profit_margin,
        "dte": dte, "fpe": fpe, "peg": peg, "ps": ps, "pb": pb,
        "rev_g": rev_g, "earn_g": earn_g, "min_score": min_score,
        "min_ok": min_ok, "min_coverage": min_coverage,
    }

    if market == "USA":
        universe = load_universe_file(pjoin("universes", "sp500.txt" if usa_bucket == "S&P 500" else "nasdaq100.txt"))
        st.caption(f"📋 Evren: {usa_bucket} ({len(universe)} hisse)")
    else:
        universe = load_universe_file(pjoin("universes", "bist100.txt"))
        st.caption(f"📋 Evren: BIST100 ({len(universe)} hisse)")

    if not universe:
        st.error("Evren listesi boş veya dosya bulunamadı!")
        st.stop()

    run_screener = st.button(
        "🔎 Screener Çalıştır",
        type="secondary",
        disabled=not use_fa,
        use_container_width=True,
    )

    st.divider()

    # ---- TEKNİK ANALİZ ----
    st.markdown("### 📊 2) Teknik Analiz")
    preset_name = st.selectbox(
        "Risk Profili",
        list(PRESETS.keys()),
        index=1,
        help="Defansif: Düşük risk | Dengeli: Orta | Agresif: Yüksek risk",
    )

    if st.session_state.selected_ticker:
        st.caption(f"📌 Screener seçimi: **{st.session_state.selected_ticker}**")
        raw_ticker = st.text_input("Sembol", value=st.session_state.selected_ticker)
    else:
        raw_ticker = st.text_input(
            "Sembol",
            value="AAPL" if market == "USA" else "THYAO",
            help="USA: AAPL, SPY | BIST: THYAO (otomatik .IS eklenir)",
        )

    ticker = normalize_ticker(raw_ticker, market)

    col_iv, col_per = st.columns(2)
    with col_iv:
        # 30m kaldırıldı → 5d ve 1wk eklendi
        interval = st.selectbox(
            "Zaman Dilimi",
            ["1d", "1h", "5d", "1wk"],
            index=0,
            help="1d: Günlük | 1h: Saatlik | 5d: ~5 Günlük | 1wk: Haftalık",
            format_func=lambda x: {
                "1d": "📅 Günlük",
                "1h": "🕐 Saatlik",
                "5d": "📆 5 Günlük",
                "1wk": "🗓️ Haftalık",
            }.get(x, x),
        )
    with col_per:
        period = st.selectbox(
            "Periyot",
            ["6mo", "1y", "2y", "5y", "10y"],
            index=3,
            format_func=lambda x: {
                "6mo": "6 Ay", "1y": "1 Yıl", "2y": "2 Yıl",
                "5y": "5 Yıl", "10y": "10 Yıl",
            }.get(x, x),
        )

    with st.expander("⚙️ Teknik Parametreler", expanded=False):
        ema_fast = st.number_input("EMA Hızlı", 5, 100, 50, 1, help="Kısa vadeli EMA (varsayılan: 50)")
        ema_slow = st.number_input("EMA Yavaş", 50, 400, 200, 1, help="Uzun vadeli EMA (varsayılan: 200)")
        rsi_period = st.number_input("RSI Periyodu", 5, 30, 14, 1)
        bb_period = st.number_input("Bollinger Periyodu", 10, 50, 20, 1)
        bb_std = st.number_input("Bollinger Std", 1.0, 3.5, 2.0, 0.1)
        atr_period = st.number_input("ATR Periyodu", 5, 30, 14, 1)
        vol_sma = st.number_input("Hacim SMA", 5, 60, 20, 1)

    with st.expander("🛡️ Piyasa Filtreleri", expanded=False):
        use_spy_filter = st.checkbox("SPY > EMA200 (Sadece USA)", value=True, disabled=(market != "USA"))
        use_bist_filter = st.checkbox("XU100 > EMA200 (Sadece BIST)", value=True, disabled=(market != "BIST"))
        use_higher_tf_filter = st.checkbox("Haftalık Trend Filtresi", value=True)

    with st.expander("💰 Risk & Backtest Ayarları", expanded=False):
        initial_capital = st.number_input("Başlangıç Sermayesi (₺/$)", 100.0, 10_000_000.0, 10000.0, 500.0)
        risk_per_trade = st.slider("İşlem Başı Risk (%)", 0.002, 0.05, 0.01, 0.001, format="%.3f")
        commission_bps = st.number_input("Komisyon (bps)", 0.0, 50.0, 5.0, 1.0)
        slippage_bps = st.number_input("Slipaj (bps)", 0.0, 20.0, 2.0, 1.0)
        risk_free_annual = st.number_input("Risksiz Faiz (yıllık)", 0.0, 0.50, 0.0, 0.01, format="%.2f")

    st.divider()

    # ---- AI AYARLARI ----
    st.markdown("### 🤖 3) Gemini AI")
    ai_on = st.checkbox("Gemini AI Aktif", value=True)

    with st.expander("🔧 AI Parametreleri", expanded=False):
        gemini_model = st.text_input("Model", value="gemini-1.5-flash")
        gemini_temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        gemini_max_tokens = st.slider("Maks Token", 256, 8192, 2048, 128)

    st.divider()

    # ---- HABER ANALİZİ ----
    st.markdown("### 📰 4) Haber Duygu Analizi")
    use_sentiment = st.checkbox("Google News + Gemini", value=True)

    st.divider()
    run_btn = st.button("🚀 Teknik Analizi Çalıştır", type="primary", use_container_width=True)
    if run_btn:
        st.session_state.ta_ran = True
        # Yeni analiz başladığında sohbet geçmişini koru ama AI metnini sıfırla
        st.session_state.gemini_text = ""


# =============================
# CFG OLUŞTUR
# =============================
cfg = {
    "ema_fast": ema_fast, "ema_slow": ema_slow, "rsi_period": rsi_period,
    "bb_period": bb_period, "bb_std": bb_std, "atr_period": atr_period,
    "vol_sma": vol_sma, "initial_capital": initial_capital,
    "risk_per_trade": risk_per_trade, "commission_bps": commission_bps,
    "slippage_bps": slippage_bps,
}
cfg.update(PRESETS[preset_name])


# =============================
# TEMEL ANALİZ SCREENER — Paralel Veri Çekimi
# =============================
if run_screener and use_fa:
    progress_bar = st.progress(0, text="Fundamental veriler çekiliyor...")

    def _fetch_one(tk: str):
        tk_norm = normalize_ticker(tk, market)
        f = fetch_fundamentals_generic(tk_norm, market=market)
        score, breakdown, passed = fundamental_score_row(f, fa_mode, thresholds)
        f.update({
            "FA_skor": score,
            "FA_gecti": passed,
            "FA_ok_sayisi": sum(1 for v in breakdown.values() if v.get("available") and v.get("ok")),
            "FA_kapsam": sum(1 for v in breakdown.values() if v.get("available")),
        })
        return f

    rows = []
    total = len(universe)
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_map = {executor.submit(_fetch_one, tk): i for i, tk in enumerate(universe)}
        for idx, future in enumerate(as_completed(future_map)):
            try:
                rows.append(future.result())
            except Exception:
                pass
            progress_bar.progress(min(1.0, (idx + 1) / total), text=f"İşleniyor: {idx+1}/{total}")

    progress_bar.empty()

    if rows:
        sdf = pd.DataFrame(rows)
        sdf["_sort"] = sdf["FA_gecti"].astype(int)
        sdf = sdf.sort_values(["_sort", "FA_skor", "FA_kapsam"], ascending=[False, False, False]).drop(columns=["_sort"])
        st.session_state.screener_df = sdf.copy()
        st.success(f"✅ Screener tamamlandı: {sdf['FA_gecti'].sum()} PASS / {len(sdf)} toplam")


# =============================
# TA ÇALIŞMADIYSA: SCREENER GÖSTER VE DUR
# =============================
if not st.session_state.ta_ran:
    if use_fa and not st.session_state.screener_df.empty:
        sdf = st.session_state.screener_df.copy()
        show_cols = ["ticker", "longName", "FA_gecti", "FA_skor", "FA_ok_sayisi",
                     "FA_kapsam", "sector", "industry", "trailingPE", "forwardPE",
                     "pegRatio", "priceToSalesTrailing12Months", "priceToBook",
                     "returnOnEquity", "operatingMargins", "profitMargins",
                     "debtToEquity", "revenueGrowth", "earningsGrowth", "marketCap"]
        st.subheader(f"📋 Fundamental Screener Sonuçları — {market}")
        st.dataframe(sdf[[c for c in show_cols if c in sdf.columns]], use_container_width=True, height=380)

        pass_list = sdf.loc[sdf["FA_gecti"] == True, "ticker"].tolist()
        if not pass_list:
            st.warning("Bu eşiklerle PASS çıkan hisse yok. Eşikleri gevşetmeyi deneyin.")
        else:
            st.success(f"🎯 {len(pass_list)} hisse PASS kriterini sağladı")
            col_pick1, col_pick2 = st.columns([3, 1])
            with col_pick1:
                picked = st.selectbox("PASS listesinden hisse seç", pass_list, index=0)
            with col_pick2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("➡️ TA'ya Aktar", use_container_width=True):
                    st.session_state.selected_ticker = picked
                    st.rerun()

    st.markdown("""
    <div style="background:linear-gradient(135deg,#111827,#1a2332);border:1px solid #1e3a5f;
                border-radius:12px;padding:24px;text-align:center;margin-top:20px">
        <div style="font-size:2rem">🚀</div>
        <div style="color:#60a5fa;font-weight:600;margin-top:8px">Teknik Analizi Başlatmak İçin Hazır</div>
        <div style="color:#64748b;margin-top:4px;font-size:0.9rem">
            Sol menüden sembol ve ayarları yapılandırın, ardından "Teknik Analizi Çalıştır" butonuna basın.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# =============================
# TEKNİK ANALİZ PIPELINE
# =============================
with st.spinner("Piyasa rejimleri ve trend filtreleri kontrol ediliyor..."):
    market_filter_ok = True
    if market == "USA" and use_spy_filter:
        market_filter_ok = get_spy_regime_ok()
    elif market == "BIST" and use_bist_filter:
        market_filter_ok = get_bist_regime_ok()

    higher_tf_filter_ok = True
    if use_higher_tf_filter:
        higher_tf_filter_ok = get_higher_tf_trend(ticker, "1wk", 200)

sentiment_summary = ""
if use_sentiment and ai_on and FEEDPARSER_OK:
    company_name = ""
    if not st.session_state.screener_df.empty:
        row = find_screener_row(st.session_state.screener_df, ticker)
        if row and row.get("longName"):
            company_name = row["longName"]
    with st.spinner("Haberler analiz ediliyor..."):
        sent = get_news_sentiment(ticker, company_name, gemini_model, gemini_temp)
        sentiment_summary = sent["summary"] if not sent.get("error") else f"Haber analizi başarısız: {sent['error']}"
        st.session_state.sentiment_summary = sentiment_summary
elif use_sentiment and not FEEDPARSER_OK:
    st.warning("📦 Haber analizi için `feedparser` gerekli: `pip install feedparser`")

with st.spinner(f"Fiyat verisi indiriliyor: **{ticker}**"):
    df_raw = load_data_cached(ticker, period, interval)

if df_raw.empty:
    st.error(f"❌ Veri alınamadı: **{ticker}** — Sembolü ve bağlantıyı kontrol edin.")
    st.stop()

if not {"Open", "High", "Low", "Close", "Volume"}.issubset(set(df_raw.columns)):
    st.error("❌ OHLCV kolonları eksik.")
    st.stop()

if len(df_raw) < 260 and interval == "1d":
    st.warning("⚠️ 260 günden az veri: Metrikler daha güvenilmez olabilir.")

df = build_features(df_raw, cfg)

benchmark_ticker = "SPY" if market == "USA" else "XU100.IS"
benchmark_df = load_data_cached(benchmark_ticker, period, interval)
benchmark_returns = benchmark_df["Close"].pct_change().dropna() if not benchmark_df.empty else None

df, checkpoints = signal_with_checkpoints(df, cfg, market_filter_ok=market_filter_ok, higher_tf_filter_ok=higher_tf_filter_ok)
latest = df.iloc[-1]

live = get_live_price(ticker)
live_price = live.get("last_price", np.nan)

# Sinyal
if int(latest["ENTRY"]) == 1:
    rec = "AL"
    rec_class = "signal-al"
elif int(latest["EXIT"]) == 1:
    rec = "SAT"
    rec_class = "signal-sat"
elif latest["SCORE"] >= 80:
    rec = "AL (Güçlü Trend)"
    rec_class = "signal-al"
elif latest["SCORE"] >= 60:
    rec = "İZLE"
    rec_class = "signal-izle"
else:
    rec = "UZAK DUR"
    rec_class = "signal-sat"

eq, tdf, metrics = backtest_long_only(df, cfg, risk_free_annual=risk_free_annual, benchmark_returns=benchmark_returns)
tp = target_price_band(df)
rr_info = rr_from_atr_stop(latest, tp, cfg)
overbought_result = detect_speculation(df)


# =============================
# GRAFİKLER — Koyu Tema
# =============================
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,17,23,0.8)",
    font=dict(color="#94a3b8", family="Inter"),
    xaxis=dict(gridcolor="#1e3a5f", linecolor="#334155", showgrid=True),
    yaxis=dict(gridcolor="#1e3a5f", linecolor="#334155", showgrid=True),
    margin=dict(l=10, r=10, t=40, b=10),
)


def make_price_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Fiyat",
        increasing=dict(line=dict(color="#4ade80"), fillcolor="#14532d"),
        decreasing=dict(line=dict(color="#f87171"), fillcolor="#450a0a"),
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50",
                             line=dict(color="#60a5fa", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA200",
                             line=dict(color="#a78bfa", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Üst",
                             line=dict(color="#f59e0b", dash="dot", width=1), opacity=0.7))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], name="BB Orta",
                             line=dict(color="#6b7280", dash="dot", width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Alt",
                             line=dict(color="#f59e0b", dash="dot", width=1), opacity=0.7))

    entries = df[df["ENTRY"] == 1]
    exits = df[df["EXIT"] == 1]
    fig.add_trace(go.Scatter(x=entries.index, y=entries["Close"], mode="markers", name="AL",
                             marker=dict(symbol="triangle-up", size=12, color="#4ade80",
                                         line=dict(color="#16a34a", width=1))))
    fig.add_trace(go.Scatter(x=exits.index, y=exits["Close"], mode="markers", name="SAT",
                             marker=dict(symbol="triangle-down", size=12, color="#f87171",
                                         line=dict(color="#dc2626", width=1))))
    fig.update_layout(
        height=550,
        xaxis_rangeslider_visible=False,
        title=dict(text=f"<b>{ticker}</b> — Fiyat + EMA + Bollinger + Sinyaller",
                   font=dict(color="#e2e8f0")),
        **CHART_LAYOUT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    )
    return fig


def make_rsi_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#60a5fa", width=2)))
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(220,38,38,0.1)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(22,163,74,0.1)", line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color="#f87171", line_width=1, annotation_text="Aşırı Alım")
    fig.add_hline(y=30, line_dash="dash", line_color="#4ade80", line_width=1, annotation_text="Aşırı Satım")
    fig.update_layout(height=240, title=dict(text="RSI", font=dict(color="#e2e8f0")),
                      yaxis=dict(range=[0, 100]), **CHART_LAYOUT)
    return fig


def make_macd_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = ["#4ade80" if v >= 0 else "#f87171" for v in df["MACD_hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogram",
                         marker_color=colors, opacity=0.8))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                             line=dict(color="#60a5fa", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Sinyal",
                             line=dict(color="#f59e0b", width=1.5)))
    fig.update_layout(height=240, title=dict(text="MACD", font=dict(color="#e2e8f0")), **CHART_LAYOUT)
    return fig


def make_atr_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["ATR_PCT"] * 100, name="ATR%",
                             line=dict(color="#a78bfa", width=2),
                             fill="tozeroy", fillcolor="rgba(167,139,250,0.1)"))
    fig.update_layout(height=240, title=dict(text="ATR % (Volatilite)", font=dict(color="#e2e8f0")), **CHART_LAYOUT)
    return fig


def make_stoch_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "STOCH_RSI_K" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_RSI_K"], name="K",
                                 line=dict(color="#60a5fa", width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_RSI_D"], name="D",
                                 line=dict(color="#f59e0b", width=1.5)))
        fig.add_hline(y=80, line_dash="dash", line_color="#f87171", line_width=1, annotation_text="80")
        fig.add_hline(y=20, line_dash="dash", line_color="#4ade80", line_width=1, annotation_text="20")
    fig.update_layout(height=240, title=dict(text="Stochastic RSI", font=dict(color="#e2e8f0")),
                      yaxis=dict(range=[0, 100]), **CHART_LAYOUT)
    return fig


def make_bbwidth_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "BB_WIDTH" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_WIDTH"] * 100, name="BB Genişlik %",
                                 line=dict(color="#f59e0b", width=2),
                                 fill="tozeroy", fillcolor="rgba(245,158,11,0.1)"))
        fig.add_hline(y=2, line_dash="dash", line_color="#64748b", line_width=1,
                      annotation_text="Sıkışma Bölgesi")
    fig.update_layout(height=240, title=dict(text="Bollinger Bandı Genişliği %", font=dict(color="#e2e8f0")), **CHART_LAYOUT)
    return fig


def make_volratio_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "VOL_RATIO" in df.columns:
        colors_v = ["#f87171" if v > 1.5 else "#4ade80" if v > 1.0 else "#64748b" for v in df["VOL_RATIO"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["VOL_RATIO"], name="Hacim Oranı",
                             marker_color=colors_v, opacity=0.8))
        fig.add_hline(y=1.5, line_dash="dash", line_color="#f87171", line_width=1, annotation_text="Anormal")
        fig.add_hline(y=1.0, line_dash="dash", line_color="#64748b", line_width=1)
    fig.update_layout(height=240, title=dict(text="Hacim Oranı (Anlık/SMA)", font=dict(color="#e2e8f0")), **CHART_LAYOUT)
    return fig


def make_equity_fig(eq: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Sermaye",
                             line=dict(color="#4ade80", width=2),
                             fill="tozeroy", fillcolor="rgba(74,222,128,0.08)"))
    fig.update_layout(height=300, title=dict(text="Backtest Sermaye Eğrisi",
                                             font=dict(color="#e2e8f0")), **CHART_LAYOUT)
    return fig


fig_price = make_price_fig(df)
fig_rsi = make_rsi_fig(df)
fig_macd = make_macd_fig(df)
fig_atr = make_atr_fig(df)
fig_stoch = make_stoch_fig(df)
fig_bbwidth = make_bbwidth_fig(df)
fig_volratio = make_volratio_fig(df)
fig_eq = make_equity_fig(eq)

figs_for_report = {
    "Fiyat + EMA + Bollinger + Sinyaller": fig_price,
    "RSI": fig_rsi, "MACD": fig_macd, "ATR%": fig_atr,
    "Stochastic RSI": fig_stoch, "Bollinger Genişliği": fig_bbwidth,
    "Hacim Oranı": fig_volratio, "Sermaye Eğrisi": fig_eq,
}


# =============================
# SEKMELER
# =============================
tab_dash, tab_ai_chat, tab_export, tab_heatmap = st.tabs([
    "📊 Dashboard",
    "💬 AI Sohbet",
    "📄 Rapor",
    "🔥 Sektörel Heatmap",
])


# ============================================================
# SEKMESİ 1: DASHBOARD
# ============================================================
with tab_dash:

    # --- Screener tablosu ---
    if use_fa and not st.session_state.screener_df.empty:
        with st.expander("📋 Fundamental Screener Sonuçları", expanded=False):
            sdf = st.session_state.screener_df.copy()
            show_cols = ["ticker", "longName", "FA_gecti", "FA_skor", "FA_ok_sayisi",
                         "FA_kapsam", "sector", "industry", "trailingPE", "forwardPE",
                         "returnOnEquity", "operatingMargins", "debtToEquity", "marketCap"]
            st.dataframe(sdf[[c for c in show_cols if c in sdf.columns]], use_container_width=True, height=300)

    # --- Ana sinyal & metrikler ---
    st.subheader("📡 Anlık Durum")
    col_sig, col_metrics = st.columns([1, 4])

    with col_sig:
        st.markdown(f'<div class="{rec_class}">{rec}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        score_color = "#4ade80" if latest["SCORE"] >= 70 else "#facc15" if latest["SCORE"] >= 50 else "#f87171"
        st.markdown(
            f'<div style="text-align:center;background:#111827;border:1px solid #1e3a5f;'
            f'border-radius:10px;padding:12px">'
            f'<div style="color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em">SKOR</div>'
            f'<div style="color:{score_color};font-size:2rem;font-weight:700;font-family:JetBrains Mono">'
            f'{latest["SCORE"]:.0f}<span style="font-size:1rem;color:#64748b">/100</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_metrics:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Borsa", market)
        m2.metric("Sembol", ticker)
        m3.metric("Son Kapanış", f"{latest['Close']:.2f}")
        m4.metric("Canlı Fiyat", f"{live_price:.2f}" if np.isfinite(live_price) else "—")
        m5.metric("Piyasa Rejimi", "🟢 BOĞA" if market_filter_ok else "🔴 AYI")
        m6.metric("Haftalık Trend", "🟢 YUKARI" if higher_tf_filter_ok else "🔴 AŞAĞI")

        m7, m8, m9, m10, m11, m12 = st.columns(6)
        m7.metric("RSI", f"{latest['RSI']:.1f}")
        m8.metric("EMA50", f"{latest['EMA50']:.2f}")
        m9.metric("EMA200", f"{latest['EMA200']:.2f}")
        m10.metric("ATR%", f"{latest['ATR_PCT']*100:.2f}%" if pd.notna(latest.get("ATR_PCT")) else "—")
        m11.metric("BB Genişliği", f"{latest['BB_WIDTH']*100:.2f}%" if pd.notna(latest.get("BB_WIDTH")) else "—")
        m12.metric("Hacim Oranı", f"{latest['VOL_RATIO']:.2f}x" if pd.notna(latest.get("VOL_RATIO")) else "—")

    st.divider()

    # --- Kontrol Noktaları ---
    st.subheader("✅ Kontrol Noktaları")
    cp_items = list(checkpoints.items())
    cols_cp = st.columns(3)
    for i, (k, v) in enumerate(cp_items):
        with cols_cp[i % 3]:
            icon = "✅" if v else "❌"
            color = "#4ade80" if v else "#f87171"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:8px 12px;'
                f'background:#111827;border:1px solid {"#1e3a5f" if not v else "#14532d"};'
                f'border-radius:8px;margin-bottom:6px">'
                f'<span style="font-size:1rem">{icon}</span>'
                f'<span style="color:{color};font-size:0.85rem;font-weight:500">{k}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # --- Aşırı Alım/Spekülasyon ---
    st.subheader("📊 Aşırı Alım & Spekülasyon Analizi")
    ob1, ob2, ob3, ob4 = st.columns(4)

    def score_color_fn(score, high=50):
        return "#f87171" if score >= high else "#facc15" if score >= 30 else "#4ade80"

    ob1.metric("Aşırı Alım", f"{overbought_result['overbought_score']}/100")
    ob2.metric("Aşırı Satım", f"{overbought_result['oversold_score']}/100")
    ob3.metric("Spekülasyon", f"{overbought_result['speculation_score']}/100")
    ob4.metric("Genel Karar", overbought_result["verdict"])

    if overbought_result.get("details"):
        with st.expander("🔍 Detaylar", expanded=False):
            for _, detail in overbought_result["details"].items():
                st.markdown(f"• {detail}")

    st.divider()

    # --- Hedef Fiyat Bandı ---
    st.subheader("🎯 Hedef Fiyat Bandı")
    base_px = float(tp["base"])
    bull_tp, bear_tp = tp.get("bull"), tp.get("bear")

    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("Baz Fiyat", f"{base_px:.2f}")
    if bull_tp:
        bc2.metric("Yükseliş H1", f"{bull_tp[0]:.2f}", f"+{pct_dist(bull_tp[0], base_px):.1f}%")
        bc3.metric("Yükseliş H2", f"{bull_tp[1]:.2f}", f"+{pct_dist(bull_tp[1], base_px):.1f}%")
    if bear_tp:
        bc4.metric("Stop (ATR)", f"{rr_info.get('stop', 0):.2f}" if rr_info.get('stop') else "—",
                   f"{pct_dist(rr_info.get('stop', base_px), base_px):.1f}%" if rr_info.get('stop') else "")

    col_rr, col_lvl = st.columns([1, 3])
    with col_rr:
        rr_val = rr_info.get("rr")
        rr_color = "#4ade80" if rr_val and rr_val >= 2 else "#facc15" if rr_val and rr_val >= 1 else "#f87171"
        st.markdown(
            f'<div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;'
            f'padding:14px;text-align:center">'
            f'<div style="color:#64748b;font-size:0.7rem;text-transform:uppercase">Risk/Ödül</div>'
            f'<div style="color:{rr_color};font-size:1.8rem;font-weight:700;font-family:JetBrains Mono">'
            f'{fmt_rr(rr_val)}</div></div>',
            unsafe_allow_html=True,
        )
    with col_lvl:
        with st.expander("📏 Destek/Direnç Seviyeleri", expanded=False):
            levels = tp.get("levels", [])
            s1_val = bear_tp[2] if bear_tp else None
            r1_val = bull_tp[2] if bull_tp else None
            for lv in levels:
                tag = " 🟩 Destek" if (s1_val and abs(lv - s1_val) < 1e-6) else \
                      " 🟥 Direnç" if (r1_val and abs(lv - r1_val) < 1e-6) else ""
                dist = pct_dist(lv, base_px)
                st.markdown(f"- **{lv:.2f}** ({dist:+.2f}%){tag}" if dist else f"- {lv:.2f}{tag}")

    st.divider()

    # --- Ana Grafik ---
    st.subheader("📈 Fiyat Grafiği")
    st.plotly_chart(fig_price, use_container_width=True)

    # --- Alt göstergeler 3'lü grid ---
    st.subheader("📉 Teknik Göstergeler")
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        st.plotly_chart(fig_rsi, use_container_width=True)
        st.caption("70 üstü aşırı alım, 30 altı aşırı satım.")
    with gc2:
        st.plotly_chart(fig_macd, use_container_width=True)
        st.caption("Histogram pozitifken momentum yukarıdadır.")
    with gc3:
        st.plotly_chart(fig_atr, use_container_width=True)
        st.caption("Yüksek ATR% = Yüksek volatilite.")

    gc4, gc5, gc6 = st.columns(3)
    with gc4:
        st.plotly_chart(fig_stoch, use_container_width=True)
        st.caption("80 üstü aşırı alım, 20 altı aşırı satım.")
    with gc5:
        st.plotly_chart(fig_bbwidth, use_container_width=True)
        st.caption("Düşük genişlik = fiyat sıkışması.")
    with gc6:
        st.plotly_chart(fig_volratio, use_container_width=True)
        st.caption("1.5x üstü anormal hacim.")

    st.divider()

    # --- Backtest ---
    st.subheader("🧪 Backtest Sonuçları")
    bm1, bm2, bm3, bm4, bm5, bm6, bm7, bm8 = st.columns(8)
    bm1.metric("Toplam Getiri", f"{metrics['Toplam Getiri']*100:.1f}%")
    bm2.metric("Yıllık Getiri", f"{metrics['Yıllık Getiri']*100:.1f}%")
    bm3.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
    bm4.metric("Sortino", f"{metrics['Sortino']:.2f}")
    bm5.metric("Maks Düşüş", f"{metrics['Maks Düşüş']*100:.1f}%")
    bm6.metric("Kazanma Oranı", f"{metrics['Kazanma Oranı']*100:.1f}%")
    bm7.metric("İşlem Sayısı", f"{metrics['İşlem Sayısı']}")
    bm8.metric("Kelly Önerisi", f"%{metrics['Kelly % (Öneri)']:.1f}")

    bm9, bm10, bm11, bm12 = st.columns(4)
    bm9.metric("Beta", f"{metrics['Beta']:.2f}")
    bm10.metric("Alpha", f"{metrics['Alpha']:.4f}")
    bm11.metric("Bilgi Oranı", f"{metrics['Bilgi Oranı']:.2f}")
    bm12.metric("Ulcer Endeksi", f"{metrics['Ulcer Endeksi']:.4f}")

    col_eq, col_trades = st.columns([2, 1])
    with col_eq:
        st.plotly_chart(fig_eq, use_container_width=True)
    with col_trades:
        if not tdf.empty:
            st.caption(f"**{len(tdf)} işlem** | Son 15 gösteriliyor")
            st.dataframe(
                tdf[["entry_date", "entry_price", "exit_price", "exit_reason", "getiri_%"]].tail(15),
                use_container_width=True,
                height=250,
            )

    # --- Haber analizi ---
    if sentiment_summary:
        st.divider()
        st.subheader("📰 Haber Duygu Analizi")
        st.info(sentiment_summary)

    # --- Gemini analizi (hızlı) ---
    st.divider()
    st.subheader("🤖 Gemini AI — Hızlı Analiz")
    if not ai_on:
        st.info("Gemini kapalı. Sol menüden 'Gemini AI Aktif' kutusunu işaretleyin.")
    else:
        pa = price_action_pack(df, last_n=20)
        st.session_state.pa_pack = pa

        if st.button("🔍 Otomatik AI Analizi Çalıştır", use_container_width=False):
            snap = df_snapshot_for_llm(df, n=30)
            f_single = fetch_fundamentals_generic(ticker, market=market)

            prompt = f"""Sen bir kıdemli teknik analist asistanısın. Aşağıdaki verileri analiz et:

Hisse: {ticker} | Piyasa: {market} | Sinyal: {rec}
Son Kapanış: {float(latest['Close']):.2f} | Skor: {float(latest['SCORE']):.0f}/100

ÖZET İSTATİSTİKLER:
{json.dumps(snap['ozet'], ensure_ascii=False, indent=2)}

AŞIRI ALIM/SPEKÜLASYON:
{json.dumps(overbought_result, ensure_ascii=False, indent=2, default=str)}

PRICE ACTION:
{json.dumps(pa, ensure_ascii=False, indent=2, default=str)}

Şunu analiz et:
1. Mevcut trend ve momentum durumu
2. Aşırı alım/satım ve spekülasyon riski
3. AL/SAT/İZLE önerisi
4. Kritik destek/direnç seviyeleri

Son olarak şu tabloyu doldur:
| Hedef | Fiyat |
|-------|-------|
| Önerilen Giriş | ... |
| İlk Hedef | ... |
| Stop Loss | ... |
"""
            with st.spinner("Gemini analiz yapıyor..."):
                img_bytes = None
                try:
                    if KALEIDO_OK:
                        img_bytes = fig_price.to_image(format="png", scale=2)
                except Exception:
                    pass
                text = gemini_generate_text(
                    prompt=prompt, model=gemini_model, temperature=gemini_temp,
                    max_output_tokens=gemini_max_tokens, image_bytes=img_bytes,
                )
                st.session_state.gemini_text = text

        if st.session_state.gemini_text:
            st.markdown(st.session_state.gemini_text)
            if not KALEIDO_OK:
                st.caption("⚠️ Grafik görseli gönderilemedi (kaleido kurulu değil). Metin tabanlı analiz yapıldı.")


# ============================================================
# SEKMESİ 2: AI SOHBET (tam bağlı)
# ============================================================
with tab_ai_chat:
    st.subheader("💬 AI Sohbet — Hisse Asistanı")
    st.caption(f"**{ticker}** hakkında serbestçe sorular sorun. Bağlam otomatik eklenir.")

    if not ai_on:
        st.warning("AI sohbet için Gemini'yi etkinleştirin (sol menü).")
        st.stop()

    # Sohbet geçmişini göster
    for msg in st.session_state.ai_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Kullanıcı girişi
    if user_input := st.chat_input("Sorunuzu yazın... (örn: 'Bu hisse için stop seviyem ne olmalı?')"):
        st.session_state.ai_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Bağlam oluştur
        snap = df_snapshot_for_llm(df, n=20)
        context = f"""
Sen bir teknik ve temel analiz asistanısın. Sadece {ticker} ({market}) hissesi hakkında yardım ediyorsun.

Mevcut bağlam:
- Sinyal: {rec} | Skor: {float(latest['SCORE']):.0f}/100
- Son Kapanış: {float(latest['Close']):.2f} | RSI: {float(latest['RSI']):.1f}
- Piyasa Rejimi: {"BOĞA" if market_filter_ok else "AYI"} | Haftalık Trend: {"YUKARI" if higher_tf_filter_ok else "AŞAĞI"}
- Aşırı Alım Skoru: {overbought_result['overbought_score']}/100 | Karar: {overbought_result['verdict']}
- RR Oranı: {fmt_rr(rr_info.get('rr'))} | Stop: {fmt_num(rr_info.get('stop'))}
- Backtest Sharpe: {metrics['Sharpe']:.2f} | Maks DD: {metrics['Maks Düşüş']*100:.1f}%
- Özet: {json.dumps(snap['ozet'], ensure_ascii=False)}

Kullanıcı Sorusu: {user_input}

Kısa, net ve pratik yanıt ver. Türkçe yanıtla. Kesin yatırım tavsiyesi verme.
"""
        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                response = gemini_generate_text(
                    prompt=context,
                    model=gemini_model,
                    temperature=gemini_temp,
                    max_output_tokens=1024,
                )
            st.markdown(response)

        st.session_state.ai_messages.append({"role": "assistant", "content": response})

    # Sohbeti temizle
    if len(st.session_state.ai_messages) > 1:
        if st.button("🗑️ Sohbeti Temizle", type="secondary"):
            st.session_state.ai_messages = [{"role": "assistant", "content": "Merhaba! Yeni bir analiz için buradayım. Ne sormak istersiniz?"}]
            st.rerun()


# ============================================================
# SEKMESİ 3: RAPOR
# ============================================================
with tab_export:
    st.subheader("📄 Analiz Raporu İndir")
    st.caption("**Önerilen:** HTML → Tarayıcıda aç → Ctrl+P → PDF olarak kaydet (grafikler tam çıkar)")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        include_charts = st.checkbox("📈 Grafikleri dahil et", value=True)
        include_trades = st.checkbox("📋 Trade listesi (ilk 25)", value=True)
        include_gemini = st.checkbox("🤖 Gemini analizini ekle", value=True)
    with col_opt2:
        include_pa = st.checkbox("📊 Price Action Pack", value=True)
        include_sentiment = st.checkbox("📰 Haber duygu analizi", value=True)
        include_overbought = st.checkbox("⚠️ Aşırı alım analizi", value=True)

    with st.spinner("Rapor hazırlanıyor..."):
        f_single = fetch_fundamentals_generic(ticker, market=market)
        f_score, f_breakdown, f_pass = fundamental_score_row(f_single, fa_mode, thresholds)
        fa_eval = {
            "mode": fa_mode, "score": f_score, "passed": f_pass,
            "ok_cnt": sum(1 for v in f_breakdown.values() if v.get("available") and v.get("ok")),
            "coverage": sum(1 for v in f_breakdown.values() if v.get("available")),
        }
        screener_row = find_screener_row(st.session_state.get("screener_df", pd.DataFrame()), ticker)
        fa_row = merge_fa_row(screener_row, f_single, fa_eval)

    meta = {
        "market": market, "ticker": ticker, "interval": interval, "period": period,
        "preset": preset_name, "ema_fast": ema_fast, "ema_slow": ema_slow,
        "rsi_period": rsi_period, "bb_period": bb_period, "bb_std": bb_std,
        "atr_period": atr_period, "vol_sma": vol_sma,
    }

    html_bytes = build_html_report(
        title=f"FA→TA Raporu — {ticker}",
        meta=meta, checkpoints=checkpoints, metrics=metrics, tp=tp, rr_info=rr_info,
        figs=(figs_for_report if include_charts else {}),
        fa_row=fa_row,
        gemini_insight=(st.session_state.gemini_text if include_gemini else None),
        pa_pack=(st.session_state.pa_pack if include_pa else None),
        sentiment_summary=(st.session_state.sentiment_summary if include_sentiment else None),
        overbought_result=(overbought_result if include_overbought else None),
    )

    st.download_button(
        "⬇️ HTML Raporu İndir (Önerilen)",
        data=html_bytes,
        file_name=f"{ticker}_FA_TA_rapor.html",
        mime="text/html",
        use_container_width=True,
    )

    st.divider()

    if not REPORTLAB_OK:
        st.warning("⚠️ PDF için `reportlab` paketi gerekli. `requirements.txt`'e ekleyip yeniden dağıtın.")
    elif not KALEIDO_OK:
        st.info("ℹ️ PDF grafikleri için `kaleido` paketi gerekli. Şu an metin ağırlıklı PDF oluşturulur.")

    if REPORTLAB_OK and st.button("🧾 PDF Oluştur", use_container_width=True):
        st.info("PDF oluşturma aktif. Grafikler için kaleido kurulu olmalıdır.")


# ============================================================
# SEKMESİ 4: SEKTÖREL HEATMAP
# ============================================================
with tab_heatmap:
    st.subheader("🔥 Sektörel Performans Haritası")
    st.caption("Hisselerin günlük, haftalık ve aylık performanslarını sektörlere göre görselleştirir.")

    col_hm_btn, col_hm_info = st.columns([1, 3])
    with col_hm_btn:
        run_heatmap = st.button("📊 Heatmap Oluştur", type="primary", use_container_width=True)
    with col_hm_info:
        st.info("Screener çalıştırılmışsa screener sonuçları kullanılır; aksi halde evrenin ilk 100 hissesi alınır.")

    if run_heatmap:
        hm_tickers = (
            st.session_state.screener_df["ticker"].tolist()
            if not st.session_state.screener_df.empty
            else [normalize_ticker(t, market) for t in universe[:100]]
        )
        use_tickers = [normalize_ticker(t, market) for t in hm_tickers]

        with st.spinner(f"{len(use_tickers)} hisse için veri çekiliyor..."):
            try:
                df_all = yf.download(
                    use_tickers, period="1mo", interval="1d",
                    auto_adjust=True, group_by="ticker", progress=False,
                )
            except Exception:
                df_all = pd.DataFrame()

            hm_data = []
            for t in use_tickers:
                try:
                    df_t = (df_all[t].copy() if len(use_tickers) > 1 else df_all.copy()).dropna()
                    if len(df_t) >= 2:
                        c_last = float(df_t["Close"].iloc[-1])
                        hm_data.append({
                            "Ticker": t,
                            "Sektör": (
                                str(find_screener_row(st.session_state.screener_df, t).get("sector", "Genel"))
                                if not st.session_state.screener_df.empty
                                and find_screener_row(st.session_state.screener_df, t)
                                else "Genel"
                            ),
                            "Günlük %": (c_last / float(df_t["Close"].iloc[-2]) - 1) * 100,
                            "Haftalık %": (c_last / float(df_t["Close"].iloc[-6 if len(df_t) >= 6 else 0]) - 1) * 100,
                            "Aylık %": (c_last / float(df_t["Close"].iloc[0]) - 1) * 100,
                        })
                except Exception:
                    pass

        if hm_data:
            df_hm = pd.DataFrame(hm_data)

            hm_tab1, hm_tab2, hm_tab3 = st.tabs(["📅 Günlük", "📆 Haftalık", "🗓️ Aylık"])

            for tab_hm, col, title in [
                (hm_tab1, "Günlük %", "Günlük Performans"),
                (hm_tab2, "Haftalık %", "Haftalık Performans"),
                (hm_tab3, "Aylık %", "Aylık Performans"),
            ]:
                with tab_hm:
                    df_hm["_abs"] = df_hm[col].abs()
                    fig_hm = px.treemap(
                        df_hm,
                        path=[px.Constant("Tüm Pazar"), "Sektör", "Ticker"],
                        values="_abs",
                        color=col,
                        color_continuous_scale="RdYlGn",
                        color_continuous_midpoint=0,
                        title=title,
                        custom_data=["Günlük %", "Haftalık %", "Aylık %"],
                    )
                    fig_hm.update_traces(
                        hovertemplate="<b>%{label}</b><br>Günlük: %{customdata[0]:.2f}%<br>"
                                      "Haftalık: %{customdata[1]:.2f}%<br>Aylık: %{customdata[2]:.2f}%"
                    )
                    fig_hm.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e2e8f0"),
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.error("Heatmap için yeterli veri çekilemedi.")
