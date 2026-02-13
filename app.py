import os
import re
import json
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

def fmt_pct(x: float) -> str:
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "—"

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
    t = [x for x in t if re.fullmatch(r"[A-Z0-9\\-]{1,10}", x)]
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
        "Trend (Close>EMA200 & EMA50>EMA200)": bool((last["Close"] > last["EMA200"]) and (last["EMA50"] > last["EMA200"])) if pd.notna(last["EMA200"]) else False,
        f"RSI > {cfg['rsi_entry_level']}": bool(last["RSI"] > cfg["rsi_entry_level"]) if pd.notna(last["RSI"]) else False,
        "MACD Hist > 0": bool(last["MACD_hist"] > 0) if pd.notna(last["MACD_hist"]) else False,
        f"ATR% < {cfg['atr_pct_max']:.2%}": bool((last["ATR"] / last["Close"]) < cfg["atr_pct_max"]) if pd.notna(last["ATR"]) and pd.notna(last["Close"]) else False,
        "Bollinger (Close>BB_mid or Breakout)": bool((last["Close"] > last["BB_mid"]) or (last["Close"] > last["BB_upper"])) if pd.notna(last["BB_mid"]) else False,
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
    downside[downside > 0] = 0.0
    downside_std = float(downside.std())
    sortino = float((excess.mean() * 252) / (downside_std * np.sqrt(252))) if len(ret) > 1 and downside_std > 0 else 0.0

    mdd = max_drawdown(eq)

    realized = [t for t in trades if t.get("exit_date") is not None]
    n_trades = len(realized)
    wins = [t for t in realized if (t.get("pnl") is not None and t["pnl"] > 0)]
    win_rate = (len(wins) / n_trades) if n_trades > 0 else 0.0

    metrics = {
        "Total Return": total_return,
        "Annualized Return": ann_return,
        "Annualized Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": mdd,
        "Trades": float(n_trades),
        "Win Rate": win_rate,
        "Final Equity": float(eq.iloc[-1]) if len(eq) else float(cfg["initial_capital"]),
    }

    trades_df = pd.DataFrame(realized)
    return eq, trades_df, metrics

# =============================
# Data download
# =============================
@st.cache_data(ttl=3 * 3600, show_spinner=False)
def download_ohlcv(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    df = _flatten_yf(df)
    return df

# =============================
# Live price helper (AI için faydalı)
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
        "Open","High","Low","Close","Volume",
        "EMA50","EMA200","RSI",
        "MACD","MACD_signal","MACD_hist",
        "BB_mid","BB_upper","BB_lower",
        "ATR","ATR_PCT","VOL_SMA",
        "SCORE","ENTRY","EXIT"
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

def build_ai_context(
    df: pd.DataFrame,
    ticker: str,
    market: str,
    latest: pd.Series,
    checkpoints: dict,
    metrics: dict,
    live: dict,
) -> dict:
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
        "data_snapshot": df_snapshot_for_llm(df, n=160),  # ✅ df bug fix
        "constraints": [
            "Yatırım tavsiyesi verme. Sadece eğitim amaçlı analiz.",
            "Mutlaka riskler ve geçersiz kılacak koşulları yaz (invalidations).",
        ],
    }

def call_openai(messages, model: str, temperature: float = 0.2):
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
    resp = client.responses.create(
        model=model,
        input=messages,
        temperature=temperature,
    )
    return resp.output_text

# =============================
# UI
# =============================
st.title("FA→TA Trading + AI (Screener + Backtest + AI Chat)")

with st.sidebar:
    st.header("Ayarlar")

    market = st.selectbox("Market", ["USA", "BIST"], index=0)
    use_fa = st.checkbox("FA aktif (UI gating)", value=True)

    st.subheader("Universe (USA)")
    universe_mode = st.selectbox(
        "Universe Modu",
        ["Auto: S&P500 + Nasdaq100"],
        index=0,
        disabled=(market != "USA"),
    )
    extra = st.text_area("Ek sembol ekle (virgülle): örn APA, MU", value="").strip()
    extra_list = [x.strip().upper().replace(".", "-") for x in extra.split(",") if x.strip()]

    if market == "USA":
        universe = build_usa_universe(extra_list)  # ✅ auto universe
        st.caption(f"S&P500 + Nasdaq100 toplam: **{len(universe)}** sembol")
    else:
        universe = []

    run_screener = st.button("🔎 Screener Çalıştır", type="secondary", disabled=(not use_fa or market != "USA"))

    st.divider()
    st.subheader("Tek Sembol Backtest")
    ticker_in = st.text_input("Ticker", value="AAPL")
    period = st.selectbox("Period", ["1y", "2y", "5y", "10y"], index=2)
    interval = st.selectbox("Interval", ["1d", "1h"], index=0)

    st.subheader("Strateji Parametreleri")
    cfg = {
        "ema_fast": st.number_input("EMA Fast (EMA50)", min_value=5, max_value=200, value=50, step=1),
        "ema_slow": st.number_input("EMA Slow (EMA200)", min_value=20, max_value=400, value=200, step=1),
        "rsi_period": st.number_input("RSI Period", min_value=5, max_value=30, value=14, step=1),
        "rsi_entry_level": st.number_input("RSI Entry Level", min_value=40.0, max_value=80.0, value=55.0, step=1.0),
        "rsi_exit_level": st.number_input("RSI Exit Level", min_value=10.0, max_value=60.0, value=45.0, step=1.0),
        "bb_period": st.number_input("BB Period", min_value=10, max_value=50, value=20, step=1),
        "bb_std": st.number_input("BB Std", min_value=1.0, max_value=4.0, value=2.0, step=0.1),
        "atr_period": st.number_input("ATR Period", min_value=5, max_value=30, value=14, step=1),
        "atr_pct_max": st.number_input("ATR% Max", min_value=0.01, max_value=0.20, value=0.08, step=0.01),
        "vol_sma": st.number_input("Volume SMA", min_value=5, max_value=60, value=20, step=1),
        "atr_stop_mult": st.number_input("ATR Stop Mult", min_value=0.5, max_value=10.0, value=3.0, step=0.5),
        "risk_per_trade": st.number_input("Risk per trade (equity %)", min_value=0.001, max_value=0.10, value=0.01, step=0.001),
        "initial_capital": st.number_input("Initial Capital", min_value=100.0, max_value=1_000_000.0, value=10_000.0, step=100.0),
        "commission_bps": st.number_input("Commission (bps)", min_value=0.0, max_value=50.0, value=1.0, step=0.5),
        "slippage_bps": st.number_input("Slippage (bps)", min_value=0.0, max_value=50.0, value=2.0, step=0.5),
    }
    risk_free_annual = st.number_input("Risk-free annual (örn 0.05)", min_value=0.0, max_value=0.20, value=0.05, step=0.01)

    st.divider()
    st.header("AI Ayarları")
    ai_on = st.checkbox("AI Chat aktif", value=True)
    ai_model = st.text_input("Model", value="gpt-4.1-mini")
    ai_temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

# =============================
# Main content
# =============================
colA, colB = st.columns([1.2, 1])

with colA:
    st.subheader("Tek Sembol Analiz + Backtest")

    ticker = normalize_ticker(ticker_in, market)
    df = download_ohlcv(ticker, period=period, interval=interval)

    if df.empty or len(df) < 60:
        st.warning("Veri yok veya yetersiz (en az ~60 bar önerilir). Ticker/period/interval değiştir.")
        st.stop()

    df_feat = build_features(df, cfg)
    market_ok = get_spy_regime_ok() if market == "USA" else True
    df_sig, checkpoints = signal_with_checkpoints(df_feat, cfg, market_ok)

    eq, trades_df, metrics = backtest_long_only(df_sig, cfg, risk_free_annual)

    live = get_live_price(ticker)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Return", fmt_pct(metrics["Total Return"]))
    m2.metric("Max DD", fmt_pct(metrics["Max Drawdown"]))
    m3.metric("Sharpe", f'{metrics["Sharpe"]:.2f}')
    m4.metric("Sortino", f'{metrics["Sortino"]:.2f}')

    st.caption("Checkpoint (son bar):")
    st.json(checkpoints)

    # Price chart + signals
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_sig.index,
        open=df_sig["Open"], high=df_sig["High"], low=df_sig["Low"], close=df_sig["Close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(x=df_sig.index, y=df_sig["EMA50"], mode="lines", name="EMA50"))
    fig.add_trace(go.Scatter(x=df_sig.index, y=df_sig["EMA200"], mode="lines", name="EMA200"))
    ent = df_sig[df_sig["ENTRY"] == 1]
    exi = df_sig[df_sig["EXIT"] == 1]
    fig.add_trace(go.Scatter(x=ent.index, y=ent["Close"], mode="markers", name="ENTRY", marker_symbol="triangle-up"))
    fig.add_trace(go.Scatter(x=exi.index, y=exi["Close"], mode="markers", name="EXIT", marker_symbol="triangle-down"))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Equity Curve")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Equity"))
    fig2.update_layout(height=320)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Trades")
    if trades_df is None or trades_df.empty:
        st.info("Trade yok (entry koşulları hiç oluşmamış olabilir).")
    else:
        show_cols = [c for c in ["entry_date", "entry_price", "exit_date", "exit_price", "exit_reason", "shares", "pnl"] if c in trades_df.columns]
        st.dataframe(trades_df[show_cols], use_container_width=True)

with colB:
    st.subheader("USA Screener (Top Skorlar)")
    if market != "USA":
        st.info("Screener yalnızca USA seçiliyken aktif.")
    else:
        if run_screener:
            market_ok = get_spy_regime_ok()
            st.caption(f"Market Regime (SPY > EMA200): **{market_ok}**")

            max_n = 120
            uni = universe[:max_n]

            rows = []
            prog = st.progress(0)
            for i, sym in enumerate(uni, start=1):
                try:
                    df_u = download_ohlcv(sym, period="2y", interval="1d")
                    if df_u.empty or len(df_u) < 120:
                        continue
                    df_feat_u = build_features(df_u, cfg)
                    df_sig_u, cp = signal_with_checkpoints(df_feat_u, cfg, market_ok)
                    last = df_sig_u.iloc[-1]

                    rows.append({
                        "Ticker": sym,
                        "Score": float(last["SCORE"]),
                        "Entry": int(last["ENTRY"]),
                        "Close": float(last["Close"]),
                        "RSI": float(last["RSI"]),
                        "ATR%": float(last["ATR_PCT"]) if pd.notna(last["ATR_PCT"]) else np.nan,
                        "CP_Market": bool(cp["Market Filter OK"]),
                        "CP_Trend": bool(cp["Trend (Close>EMA200 & EMA50>EMA200)"]),
                        "CP_Liq": bool(cp["Liquidity (Volume > VolSMA)"]),
                    })
                except Exception:
                    pass
                prog.progress(i / max_n)

            prog.empty()

            if not rows:
                st.warning("Sonuç yok. Parametreler çok sıkı olabilir veya veri alınamamış olabilir.")
            else:
                out = pd.DataFrame(rows)
                out = out.sort_values(["Entry", "Score"], ascending=[False, False]).reset_index(drop=True)
                st.dataframe(out.head(50), use_container_width=True)
                st.caption("Not: Universe büyükse yavaşlar. Şu an performans için ilk 120 sembol ile sınırlandı.")

# =============================
# AI Chat (sağ altta / sayfa altında)
# =============================
st.divider()
st.subheader("🤖 AI Analiz (Chat)")

if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = [
        {"role": "assistant", "content": "Sorunu yaz: örn. “Riskler ne, hangi şartta çıkarım?”"}
    ]

if not st.secrets.get("OPENAI_API_KEY", ""):
    st.warning("OPENAI_API_KEY bulunamadı. Streamlit Cloud > Secrets'e ekle: OPENAI_API_KEY=...")
    ai_on = False

for m in st.session_state.ai_messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Sorunu yaz... (ör: 'Riskler neler?')")
if user_q and ai_on:
    st.session_state.ai_messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    latest = df_sig.iloc[-1]
    ctx = build_ai_context(
        df=df_sig,  # ✅ df bug fix: df burada veriliyor
        ticker=ticker,
        market=market,
        latest=latest,
        checkpoints=checkpoints,
        metrics=metrics,
        live=live,
    )

    system = (
        "Sen bir yatırım analizi asistanısın. YATIRIM TAVSİYESİ VERME.\n"
        "Kullanıcıya: (1) kısa özet, (2) riskler, (3) invalidation/çıkış koşulları, (4) takip edilecek 3 metrik ver.\n"
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

st.caption("⚠️ Bu uygulama eğitim amaçlıdır. Yatırım tavsiyesi değildir.")
