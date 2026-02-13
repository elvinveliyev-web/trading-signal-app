import os
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import requests

# OpenAI (new SDK)
from openai import OpenAI

st.set_page_config(page_title="FA→TA Trading + AI", layout="wide")

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
        return True  # fail-open
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
# Fundamentals (USA) via yfinance
# =============================
@st.cache_data(ttl=12 * 3600, show_spinner=False)
def fetch_fundamentals_usa(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    out = {
        "ticker": ticker,
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
    }

    dte = out["debtToEquity"]
    if pd.notna(dte) and dte > 10:
        out["debtToEquity"] = dte / 100.0

    return out

def fundamental_score_row(row: dict, mode: str, thresholds: dict) -> tuple[float, dict, bool]:
    b = {}

    def ok(name, cond, weight):
        b[name] = {"ok": bool(cond), "weight": weight}
        return weight if cond else 0.0

    score = 0.0
    total_w = 0.0

    if mode == "Quality":
        total_w += 20; score += ok("ROE", pd.notna(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"], 20)
        total_w += 15; score += ok("Op Margin", pd.notna(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"], 15)
        total_w += 20; score += ok("Debt/Equity", pd.notna(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"], 20)
        total_w += 15; score += ok("Profit Margin", pd.notna(row["profitMargins"]) and row["profitMargins"] >= thresholds["profit_margin"], 15)
        total_w += 30; score += ok("FCF", pd.notna(row["freeCashflow"]) and row["freeCashflow"] > 0, 30)

    elif mode == "Value":
        total_w += 30; score += ok("Forward P/E", pd.notna(row["forwardPE"]) and row["forwardPE"] <= thresholds["fpe"], 30)
        total_w += 20; score += ok("PEG", pd.notna(row["pegRatio"]) and row["pegRatio"] <= thresholds["peg"], 20)
        total_w += 20; score += ok("P/S", pd.notna(row["priceToSalesTrailing12Months"]) and row["priceToSalesTrailing12Months"] <= thresholds["ps"], 20)
        total_w += 15; score += ok("P/B", pd.notna(row["priceToBook"]) and row["priceToBook"] <= thresholds["pb"], 15)
        total_w += 15; score += ok("ROE", pd.notna(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"], 15)

    else:  # Growth
        total_w += 35; score += ok("Revenue Growth", pd.notna(row["revenueGrowth"]) and row["revenueGrowth"] >= thresholds["rev_g"], 35)
        total_w += 35; score += ok("Earnings Growth", pd.notna(row["earningsGrowth"]) and row["earningsGrowth"] >= thresholds["earn_g"], 35)
        total_w += 15; score += ok("Op Margin", pd.notna(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"], 15)
        total_w += 15; score += ok("Debt/Equity", pd.notna(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"], 15)

    score = (score / total_w) * 100 if total_w > 0 else 0.0
    ok_count = sum(1 for v in b.values() if v["ok"])
    pass_bool = (score >= thresholds["min_score"]) and (ok_count >= thresholds["min_ok"])
    return float(score), b, bool(pass_bool)

# =============================
# Target price band (non-LLM): scenario + levels
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

    return {
        "base": px,
        "bull": (bull1, bull2, r1),
        "bear": (bear1, bear2, s1),
        "levels": lv,
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
# LLM helpers
# =============================
def df_snapshot_for_llm(df: pd.DataFrame, n: int = 140) -> dict:
    use_cols = [
        "Open","High","Low","Close","Volume","EMA50","EMA200","RSI","MACD","MACD_signal","MACD_hist",
        "BB_mid","BB_upper","BB_lower","ATR","ATR_PCT","VOL_SMA","SCORE","ENTRY","EXIT"
    ]
    cols = [c for c in use_cols if c in df.columns]
    tail = df[cols].tail(n).copy()
    tail.index = tail.index.astype(str)
    return {
        "cols": cols,
        "n": int(len(tail)),
        "last_index": str(tail.index[-1]) if len(tail) else None,
        "rows": tail.to_dict(orient="records"),
    }

def build_ai_context(df: pd.DataFrame, ticker: str, market: str, latest: pd.Series, checkpoints: dict, metrics: dict, tp: dict, live: dict) -> dict:
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

# Secrets'te yoksa kullanıcı yan menüden girebilsin + düzgün hata
def _get_openai_key() -> str:
    k = st.secrets.get("OPENAI_API_KEY", "").strip()
    if not k:
        k = st.session_state.get("OPENAI_API_KEY_UI", "").strip()
    return k

def call_openai(messages, model: str, temperature: float = 0.2):
    api_key = _get_openai_key()
    if not api_key:
        raise ValueError('OPENAI_API_KEY eksik. Streamlit Secrets’e ekle: OPENAI_API_KEY="sk-..." (veya yan menüden gir).')
    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=model,
        input=messages,
        temperature=temperature,
    )
    return resp.output_text

# =============================
# Presets
# =============================
BIST_EXAMPLES = ["THYAO", "ASELS", "KCHOL", "SISE", "BIMAS"]

PRESETS = {
    "Defansif": {"rsi_entry_level": 52, "rsi_exit_level": 46, "atr_pct_max": 0.06, "atr_stop_mult": 3.5},
    "Dengeli": {"rsi_entry_level": 50, "rsi_exit_level": 45, "atr_pct_max": 0.08, "atr_stop_mult": 3.0},
    "Agresif": {"rsi_entry_level": 48, "rsi_exit_level": 43, "atr_pct_max": 0.10, "atr_stop_mult": 2.5},
}

# =============================
# Universe helpers: S&P 500 + Nasdaq-100 (requests + user-agent, fail-open)
# =============================
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0; +https://streamlit.io)"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        dfu = tables[0]
        return sorted(dfu["Symbol"].astype(str).str.upper().tolist())
    except Exception:
        return ["A
::contentReference[oaicite:0]{index=0}
