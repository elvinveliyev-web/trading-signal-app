import os
import re
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

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
# USA Universe: S&P500 + Nasdaq100 (auto)
# =============================
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_sp500() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tbl = pd.read_html(url)[0]
    t = tbl["Symbol"].astype(str).str.strip().str.upper().tolist()
    t = [x.replace(".", "-") for x in t]  # BRK.B -> BRK-B
    return sorted(list(set(t)))

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_nasdaq100() -> list[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)

    cand, col = None, None
    for tb in tables:
        for c in tb.columns:
            lc = str(c).lower()
            if "ticker" in lc or "symbol" in lc:
                cand, col = tb, c
                break
        if cand is not None:
            break

    if cand is None or col is None:
        return []

    t = cand[col].astype(str).str.strip().str.upper().tolist()
    t = [x.replace(".", "-") for x in t]
    t = [x for x in t if re.fullmatch(r"[A-Z0-9\-]{1,10}", x)]
    return sorted(list(set(t)))

def build_usa_universe(extra_list: list[str]) -> list[str]:
    sp = fetch_sp500()
    ndx = fetch_nasdaq100()
    return sorted(list(set(sp + ndx + extra_list)))

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

        # trailing stop
        if shares > 0 and pd.notna(row["ATR"]) and row["ATR"] > 0:
            new_stop = price - cfg["atr_stop_mult"] * float(row["ATR"])
            stop = max(stop, new_stop) if pd.notna(stop) else new_stop

        # exit
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

        # entry
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
        "data_ok": True,
    }

    # basit veri sağlık kontrolü
    must = ["forwardPE", "pegRatio", "priceToSalesTrailing12Months", "priceToBook", "returnOnEquity", "operatingMargins", "profitMargins", "debtToEquity"]
    okcount = sum(pd.notna(out.get(k)) for k in must)
    if okcount < 3:
        out["data_ok"] = False

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
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
    resp = client.responses.create(model=model, input=messages, temperature=temperature)
    return resp.output_text

# =============================
# Presets
# =============================
PRESETS = {
    "Defansif": {"rsi_entry_level": 52, "rsi_exit_level": 46, "atr_pct_max": 0.06, "atr_stop_mult": 3.5},
    "Dengeli": {"rsi_entry_level": 50, "rsi_exit_level": 45, "atr_pct_max": 0.08, "atr_stop_mult": 3.0},
    "Agresif": {"rsi_entry_level": 48, "rsi_exit_level": 43, "atr_pct_max": 0.10, "atr_stop_mult": 2.5},
}

# =============================
# UI
# =============================
st.title("📈 FA→TA Trading Uygulaması + 🤖 AI Analiz")
st.caption("Önce fundamental ile evreni daralt, sonra teknik analizle giriş/çıkış zamanla. Otomatik emir göndermez.")

if "screener_df" not in st.session_state:
    st.session_state.screener_df = pd.DataFrame()
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = [{"role": "assistant", "content": "Sorunu yaz: örn. “Riskler ne, hedef bant ne, hangi şartta çıkarım?”"}]

with st.sidebar:
    st.header("Piyasa")
    market = st.selectbox("Market", ["USA", "BIST"], index=0)

    st.header("1) Fundamental Screener (opsiyonel)")
    use_fa = st.checkbox("Fundamental filtreyi kullan", value=(market == "USA"))
    fa_mode = st.selectbox("Fundamental Mod", ["Quality", "Value", "Growth"], index=0, disabled=(not use_fa or market != "USA"))

    st.caption("Eşikler (USA için)")
    roe = st.slider("ROE min", 0.0, 0.40, 0.15, 0.01, disabled=(not use_fa or market != "USA"))
    op_margin = st.slider("Operating Margin min", 0.0, 0.40, 0.10, 0.01, disabled=(not use_fa or market != "USA"))
    profit_margin = st.slider("Profit Margin min", 0.0, 0.40, 0.08, 0.01, disabled=(not use_fa or market != "USA"))
    dte = st.slider("Debt/Equity max", 0.0, 3.0, 1.0, 0.05, disabled=(not use_fa or market != "USA"))
    fpe = st.slider("Forward P/E max", 0.0, 60.0, 20.0, 1.0, disabled=(not use_fa or market != "USA"))
    peg = st.slider("PEG max", 0.0, 5.0, 1.5, 0.1, disabled=(not use_fa or market != "USA"))
    ps = st.slider("P/S max", 0.0, 30.0, 6.0, 0.5, disabled=(not use_fa or market != "USA"))
    pb = st.slider("P/B max", 0.0, 30.0, 6.0, 0.5, disabled=(not use_fa or market != "USA"))
    rev_g = st.slider("Revenue Growth min", 0.0, 0.50, 0.10, 0.01, disabled=(not use_fa or market != "USA"))
    earn_g = st.slider("Earnings Growth min", 0.0, 0.50, 0.10, 0.01, disabled=(not use_fa or market != "USA"))
    min_score = st.slider("Min Fundamental Score", 0, 100, 60, 1, disabled=(not use_fa or market != "USA"))
    min_ok = st.slider("Min OK sayısı", 1, 5, 3, 1, disabled=(not use_fa or market != "USA"))

    thresholds = {"roe": roe, "op_margin": op_margin, "profit_margin": profit_margin, "dte": dte, "fpe": fpe, "peg": peg, "ps": ps, "pb": pb,
                  "rev_g": rev_g, "earn_g": earn_g, "min_score": min_score, "min_ok": min_ok}

    st.subheader("Universe (USA)")
    extra = st.text_area("Ek sembol ekle (virgülle): örn APA, MU", value="").strip()
    extra_list = [x.strip().upper().replace(".", "-") for x in extra.split(",") if x.strip()]
    universe = build_usa_universe(extra_list)
    st.caption(f"S&P500 + Nasdaq100 toplam: **{len(universe)}** sembol")

    run_screener = st.button("🔎 Screener Çalıştır", type="secondary", disabled=(not use_fa or market != "USA"))

    st.divider()
    st.header("2) Teknik Analiz + Backtest")
    preset_name = st.selectbox("Teknik Mod", list(PRESETS.keys()), index=1)

    st.subheader("Sembol (TA)")
    if st.session_state.selected_ticker:
        st.caption(f"Screener seçimi: **{st.session_state.selected_ticker}**")
        raw_ticker = st.text_input("Sembol", value=st.session_state.selected_ticker)
    else:
        raw_ticker = st.text_input("Sembol (ör: AAPL, NVCR, SPY) / BIST: THYAO", value="AAPL" if market == "USA" else "THYAO")
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

# -----------------------------
# Fundamental screener action (USA only)
# -----------------------------
if run_screener and market == "USA" and use_fa:
    with st.spinner("Fundamental veriler çekiliyor (yfinance)..."):
        rows = []
        for tk in universe:
            f = fetch_fundamentals_usa(tk)

            # Yahoo veri gelmediyse PASS hesaplamak anlamsız; kullanıcı görsün
            if not f.get("data_ok", True):
                f["FA_score"] = 0.0
                f["FA_pass"] = False
                f["FA_ok_count"] = 0
                rows.append(f)
                time.sleep(0.4)   # rate-limit azalt
                continue

            score, breakdown, passed = fundamental_score_row(f, fa_mode, thresholds)
            f["FA_score"] = score
            f["FA_pass"] = passed
            f["FA_ok_count"] = sum(1 for v in breakdown.values() if v["ok"])
            rows.append(f)

            time.sleep(0.4)  # Streamlit Cloud için önemli (0.2–0.6 arası iyi)

        sdf = pd.DataFrame(rows)
        sdf["FA_pass_int"] = sdf["FA_pass"].astype(int)
        sdf = sdf.sort_values(["FA_pass_int", "FA_score"], ascending=[False, False]).drop(columns=["FA_pass_int"])
        st.session_state.screener_df = sdf.copy()

# -----------------------------
# Screener display (if available)
# -----------------------------
if market == "USA" and use_fa and not st.session_state.screener_df.empty:
    st.subheader("🧾 Fundamental Screener Sonuçları (USA)")
    sdf = st.session_state.screener_df.copy()

    show_cols = [
        "ticker", "FA_pass", "FA_score", "FA_ok_count",
        "sector", "industry",
        "forwardPE", "pegRatio", "priceToSalesTrailing12Months", "priceToBook",
        "returnOnEquity", "operatingMargins", "profitMargins", "debtToEquity",
        "revenueGrowth", "earningsGrowth",
        "marketCap", "data_ok"
    ]
    sdf_show = sdf[[c for c in show_cols if c in sdf.columns]].copy()
    st.dataframe(sdf_show, use_container_width=True, height=320)

    pass_list = sdf.loc[sdf["FA_pass"] == True, "ticker"].tolist()
    if len(pass_list) == 0:
        st.warning("Bu eşiklerle PASS çıkan hisse yok. Eşikleri gevşet veya mode değiştir.")
    else:
        st.success(f"PASS sayısı: {len(pass_list)}")
        picked = st.selectbox("PASS listesinden hisse seç (TA’ya gönder)", pass_list, index=0)
        if st.button("➡️ Seçimi Teknik Analize Aktar"):
            st.session_state.selected_ticker = picked
            st.rerun()

# -----------------------------
# Technical run
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

@st.cache_data(show_spinner=False)
def load_data_cached(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    return _flatten_yf(df)

if not run_btn:
    st.info("Soldan ayarları yapıp **Teknik Analizi Çalıştır**’a bas.")
    st.stop()

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

st.subheader("🎯 Hedef Fiyat Bandı (Senaryo)")
bcol1, bcol2, bcol3 = st.columns(3)
bcol1.metric("Base", f"{tp['base']:.2f}")
if tp["bull"]:
    bull1, bull2, r1 = tp["bull"]
    bcol2.metric("Bull Band", f"{bull1:.2f} → {bull2:.2f}")
    if r1:
        bcol2.caption(f"Yakın direnç: {r1:.2f}")
else:
    bcol2.metric("Bull Band", "N/A")

if tp["bear"]:
    bear1, bear2, s1 = tp["bear"]
    bcol3.metric("Bear Band", f"{bear1:.2f} → {bear2:.2f}")
    if s1:
        bcol3.caption(f"Yakın destek: {s1:.2f}")
else:
    bcol3.metric("Bear Band", "N/A")

with st.expander("Seviye listesi (yaklaşık)", expanded=False):
    st.write(tp["levels"])

st.subheader("📊 Fiyat + EMA + Bollinger + Sinyaller")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA Fast"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA Slow"))
fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], name="BB Mid", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(dash="dot")))

entries = df[df["ENTRY"] == 1]
exits = df[df["EXIT"] == 1]
fig.add_trace(go.Scatter(x=entries.index, y=entries["Close"], mode="markers", name="ENTRY", marker=dict(symbol="triangle-up", size=10)))
fig.add_trace(go.Scatter(x=exits.index, y=exits["Close"], mode="markers", name="EXIT", marker=dict(symbol="triangle-down", size=10)))
fig.update_layout(height=600, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📉 RSI / MACD / ATR%")
colA, colB, colC = st.columns(3)

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
fig_rsi.add_hline(y=70)
fig_rsi.add_hline(y=30)
fig_rsi.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
colA.plotly_chart(fig_rsi, use_container_width=True)

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"))
fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
fig_macd.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
colB.plotly_chart(fig_macd, use_container_width=True)

fig_atr = go.Figure()
fig_atr.add_trace(go.Scatter(x=df.index, y=df["ATR_PCT"] * 100, name="ATR%"))
fig_atr.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="%")
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
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity"))
    fig_eq.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_eq, use_container_width=True)

# =============================
# AI Chat
# =============================
st.subheader("🤖 AI Analiz (Chat)")

if not st.secrets.get("OPENAI_API_KEY", ""):
    st.warning("OPENAI_API_KEY bulunamadı. Streamlit Cloud > Secrets'e ekle: OPENAI_API_KEY=...")
    ai_on = False

for m in st.session_state.ai_messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Sorunu yaz... (ör: 'Hedef bant ne, riskler neler?')")
if user_q and ai_on:
    st.session_state.ai_messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    ctx = build_ai_context(
        ticker=ticker,
        market=market,
        latest=latest,
        checkpoints=checkpoints,
        metrics=metrics,
        tp=tp,
        live=live,
        df=df,
    )

    system = (
        "Sen bir yatırım analizi asistanısın. YATIRIM TAVSİYESİ VERME.\n"
        "Kullanıcıya: (1) kısa özet, (2) riskler, (3) hedef fiyat bandı (bull/base/bear), "
        "(4) invalidation/çıkış koşulları, (5) takip edilecek 3 metrik ver.\n"
        "Veri: aşağıdaki JSON bağlamıdır. Uydurma haber/finansal veri üretme."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Bağlam JSON:\n" + json.dumps(ctx, ensure_ascii=False)},
        {"role": "user", "content": user_q},
    ]

    with st.chat_message("assistant"):
        with st.spinner("AI analiz ediyor..."):
            try:
                ans = call_openai(messages, model=ai_model, temperature=ai_temp)
            except Exception as e:
                ans = f"AI çağrısı hata verdi: {e}"
        st.markdown(ans)

    st.session_state.ai_messages.append({"role": "assistant", "content": ans})

st.caption("Uyarı: Bu uygulama yatırım tavsiyesi değildir. Eğitim/analiz amaçlıdır.")
