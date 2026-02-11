import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Trading Signal App", layout="wide")

# -----------------------------
# Indicators
# -----------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=close.index).ewm(alpha=1/period, adjust=False).mean()
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
    return tr.ewm(alpha=1/period, adjust=False).mean()

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min()) if len(dd) else 0.0

# -----------------------------
# Feature builder
# -----------------------------
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
    return df

# -----------------------------
# Market regime filter (SPY)
# -----------------------------
@st.cache_data(ttl=6*3600, show_spinner=False)
def get_spy_regime_ok() -> bool:
    """
    Long market filter:
    True if SPY is above its 200-day EMA (bull regime).
    Cached for 6 hours to reduce calls.
    """
    spy = yf.download("SPY", period="10y", interval="1d", auto_adjust=False, progress=False)
    if spy is None or spy.empty or len(spy) < 260:
        return True  # fail-open
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]
    spy = spy.dropna()
    spy["EMA200"] = ema(spy["Close"], 200)
    last = spy.iloc[-1]
    return bool(last["Close"] > last["EMA200"])

# -----------------------------
# Strategy: scoring + checkpoints
# -----------------------------
def signal_with_checkpoints(df: pd.DataFrame, cfg: dict, mkt_ok: bool):
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
        w["liq"] * liq_ok.astype(int) +
        w["trend"] * trend_ok.astype(int) +
        w["rsi"] * rsi_ok.astype(int) +
        w["macd"] * macd_ok.astype(int) +
        w["vol"] * vol_ok.astype(int) +
        w["bb"] * (bb_ok | bb_break).astype(int) +
        w["obv"] * obv_ok.astype(int)
    ).astype(float)

    # ENTRY includes market filter
    entry_triggers = (rsi_cross.astype(int) + macd_turn.astype(int) + bb_break.astype(int)) >= 2
    entry = trend_ok & vol_ok & liq_ok & entry_triggers & mkt_ok

    exit_ = (
        (df["Close"] < df["EMA50"]) |
        (df["MACD_hist"] < 0) |
        (df["RSI"] < cfg["rsi_exit_level"]) |
        (df["Close"] < df["BB_mid"])
    )

    df["SCORE"] = score
    df["ENTRY"] = entry.astype(int)
    df["EXIT"] = exit_.astype(int)

    last = df.iloc[-1]
    cp = {
        "Market Filter (SPY > EMA200)": bool(mkt_ok),
        "Liquidity (Volume > VolSMA)": bool(last["Volume"] > last["VOL_SMA"]) if pd.notna(last["VOL_SMA"]) else False,
        "Trend (Close>EMA200 & EMA50>EMA200)": bool((last["Close"] > last["EMA200"]) and (last["EMA50"] > last["EMA200"])) if pd.notna(last["EMA200"]) else False,
        f"RSI > {cfg['rsi_entry_level']}": bool(last["RSI"] > cfg["rsi_entry_level"]) if pd.notna(last["RSI"]) else False,
        "MACD Hist > 0": bool(last["MACD_hist"] > 0) if pd.notna(last["MACD_hist"]) else False,
        f"ATR% < {cfg['atr_pct_max']:.2%}": bool((last["ATR"] / last["Close"]) < cfg["atr_pct_max"]) if pd.notna(last["ATR"]) else False,
        "Bollinger (Close>BB_mid or Breakout)": bool((last["Close"] > last["BB_mid"]) or (last["Close"] > last["BB_upper"])) if pd.notna(last["BB_mid"]) else False,
        "OBV > OBV_EMA": bool(last["OBV"] > last["OBV_EMA"]) if pd.notna(last["OBV_EMA"]) else False,
    }

    return df, cp

# -----------------------------
# Backtest (cash + position correct)
# -----------------------------
def backtest_long_only(df: pd.DataFrame, cfg: dict):
    df = df.copy()
    entry_sig = df["ENTRY"].shift(1).fillna(0).astype(int)
    exit_sig  = df["EXIT"].shift(1).fillna(0).astype(int)

    cash = float(cfg["initial_capital"])
    shares = 0.0
    stop = np.nan

    trades = []
    equity_curve = []

    commission = cfg["commission_bps"] / 10000.0
    slippage   = cfg["slippage_bps"] / 10000.0

    for i in range(len(df)):
        row  = df.iloc[i]
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

        position_value = shares * price * (1 - slippage)
        equity = cash + position_value

        # entry
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

                    trades.append({
                        "entry_date": date,
                        "entry_price": buy_price,
                        "exit_date": None,
                        "exit_price": None,
                        "exit_reason": None,
                        "shares": shares,
                        "equity_before": equity,
                        "pnl": None
                    })

        position_value = shares * price * (1 - slippage)
        equity = cash + position_value
        equity_curve.append((date, equity))

    eq = pd.Series([v for _, v in equity_curve], index=[d for d, _ in equity_curve], name="equity").astype(float)
    eq = eq.replace([np.inf, -np.inf], np.nan).dropna()

    ret = eq.pct_change().dropna()
    total_return = eq.iloc[-1] / eq.iloc[0] - 1 if len(eq) > 1 else 0.0
    ann_return = (1 + total_return) ** (252 / max(1, len(ret))) - 1 if len(ret) > 0 else 0.0
    ann_vol = float(ret.std() * np.sqrt(252)) if len(ret) > 1 else 0.0
    sharpe = float((ret.mean() * 252) / (ret.std() * np.sqrt(252))) if len(ret) > 1 and ret.std() > 0 else 0.0
    mdd = max_drawdown(eq)

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tdf["pnl"] = tdf["pnl"].astype(float)
        tdf["return_%"] = (tdf["pnl"] / tdf["equity_before"]) * 100
        tdf["holding_days"] = (pd.to_datetime(tdf["exit_date"]) - pd.to_datetime(tdf["entry_date"])).dt.days

    metrics = {
        "Total Return": total_return,
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe (rf=0)": sharpe,
        "Max Drawdown": mdd,
        "Trades": int(len(tdf)) if not tdf.empty else 0,
        "Win Rate": float((tdf["pnl"] > 0).mean()) if not tdf.empty else 0.0
    }
    return eq, tdf, metrics

# -----------------------------
# Presets (US)
# -----------------------------
US_TICKERS = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","NFLX","JPM","XOM","SPY","QQQ"]
PRESETS = {
    "Defansif": {"rsi_entry_level": 52, "rsi_exit_level": 46, "atr_pct_max": 0.06, "atr_stop_mult": 3.5},
    "Dengeli":  {"rsi_entry_level": 50, "rsi_exit_level": 45, "atr_pct_max": 0.08, "atr_stop_mult": 3.0},
    "Agresif":  {"rsi_entry_level": 48, "rsi_exit_level": 43, "atr_pct_max": 0.10, "atr_stop_mult": 2.5},
}

# -----------------------------
# UI
# -----------------------------
st.title("📈 Sinyal Üreten Trading Uygulaması (ABD odaklı)")
st.caption("Otomatik emir göndermez. 7 indikatör + kontrol noktaları ile AL/SAT/BEKLE üretir.")

with st.sidebar:
    st.header("ABD Piyasası")
    preset_name = st.selectbox("Mod", list(PRESETS.keys()), index=1)
    use_dropdown = st.checkbox("Sembol listesinden seç", value=True)

    if use_dropdown:
        ticker = st.selectbox("Sembol", US_TICKERS, index=0)
    else:
        ticker = st.text_input("Sembol (ör: AAPL, MSFT, SPY, QQQ)", value="AAPL").strip().upper()

    period = st.selectbox("Periyot", ["6mo", "1y", "2y", "5y", "10y"], index=3)
    interval = st.selectbox("Interval", ["1d", "1h", "30m"], index=0)

    st.divider()
    st.header("Strateji Parametreleri")
    ema_fast = st.number_input("EMA Fast (trend içi)", min_value=5, max_value=100, value=50, step=1)
    ema_slow = st.number_input("EMA Slow (trend filtresi)", min_value=50, max_value=400, value=200, step=1)
    rsi_period = st.number_input("RSI Period", min_value=5, max_value=30, value=14, step=1)
    bb_period = st.number_input("Bollinger Period", min_value=10, max_value=50, value=20, step=1)
    bb_std = st.number_input("Bollinger Std", min_value=1.0, max_value=3.5, value=2.0, step=0.1)
    atr_period = st.number_input("ATR Period", min_value=5, max_value=30, value=14, step=1)
    vol_sma = st.number_input("Volume SMA", min_value=5, max_value=60, value=20, step=1)

    st.header("Market Filter")
    use_spy_filter = st.checkbox("SPY > EMA200 filtresi", value=True, help="Ayı piyasasında long sinyallerini azaltır.")

    st.header("Risk / Backtest")
    initial_capital = st.number_input("Başlangıç Sermayesi", min_value=100.0, value=10000.0, step=500.0)
    risk_per_trade = st.slider("Trade başı risk (equity %)", min_value=0.002, max_value=0.05, value=0.01, step=0.001)
    commission_bps = st.number_input("Komisyon (bps)", min_value=0.0, value=5.0, step=1.0)
    slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, value=2.0, step=1.0)

    run_btn = st.button("🚀 Çalıştır", type="primary")

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

if not run_btn:
    st.info("Soldan sembol ve parametreleri seçip **Çalıştır**’a bas.")
    st.stop()

mkt_ok = True
if use_spy_filter:
    with st.spinner("SPY rejimi kontrol ediliyor..."):
        mkt_ok = get_spy_regime_ok()

@st.cache_data(show_spinner=False)
def load_data_cached(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.dropna()

with st.spinner("Veri indiriliyor..."):
    df_raw = load_data_cached(ticker, period, interval)

if df_raw.empty or len(df_raw) < 260:
    st.error("Yetersiz veri. Daha uzun periyot seç (ör. 5y/10y) veya farklı sembol dene.")
    st.stop()

df = build_features(df_raw, cfg)
df, checkpoints = signal_with_checkpoints(df, cfg, mkt_ok=mkt_ok)

latest = df.iloc[-1]
if int(latest["ENTRY"]) == 1:
    rec = "AL"
elif int(latest["EXIT"]) == 1:
    rec = "SAT"
else:
    rec = "İZLE (Güçlü Trend)" if latest["SCORE"] >= 80 else ("BEKLE (Orta)" if latest["SCORE"] >= 60 else "UZAK DUR")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Sembol", ticker)
c2.metric("Son Fiyat", f"{latest['Close']:.2f}")
c3.metric("Skor", f"{latest['SCORE']:.0f}/100")
c4.metric("Sinyal", rec)
c5.metric("Mod", preset_name)
c6.metric("SPY Rejim", "BULL ✅" if mkt_ok else "BEAR ❌")

st.subheader("✅ Kontrol Noktaları (Son Bar)")
cp_cols = st.columns(3)
for i, (k, v) in enumerate(checkpoints.items()):
    with cp_cols[i % 3]:
        st.write(("🟢 " if v else "🔴 ") + k)

if use_spy_filter and not mkt_ok:
    st.warning("SPY şu an EMA200 altında. Market filter açık olduğu için ENTRY (AL) sinyalleri azalabilir/kapanabilir.")

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
fig.add_trace(go.Scatter(x=entries.index, y=entries["Close"], mode="markers", name="ENTRY",
                         marker=dict(symbol="triangle-up", size=10)))
fig.add_trace(go.Scatter(x=exits.index, y=exits["Close"], mode="markers", name="EXIT",
                         marker=dict(symbol="triangle-down", size=10)))

fig.update_layout(height=600, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📉 RSI / MACD / ATR%")
ind_cols = st.columns(3)

with ind_cols[0]:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
    fig_rsi.add_hline(y=cfg["rsi_entry_level"])
    fig_rsi.add_hline(y=cfg["rsi_exit_level"])
    fig_rsi.update_layout(height=250)
    st.plotly_chart(fig_rsi, use_container_width=True)

with ind_cols[1]:
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"))
    fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
    fig_macd.update_layout(height=250)
    st.plotly_chart(fig_macd, use_container_width=True)

with ind_cols[2]:
    atr_pct = (df["ATR"] / df["Close"]).replace([np.inf, -np.inf], np.nan)
    fig_atr = go.Figure()
    fig_atr.add_trace(go.Scatter(x=df.index, y=atr_pct, name="ATR%"))
    fig_atr.add_hline(y=cfg["atr_pct_max"])
    fig_atr.update_layout(height=250, yaxis_tickformat=".1%")
    st.plotly_chart(fig_atr, use_container_width=True)

st.subheader("🧪 Backtest (Long-only) + Benchmark (Buy&Hold)")
eq, trades, metrics = backtest_long_only(df, cfg)

bh = (df["Close"] / df["Close"].iloc[0]) * cfg["initial_capital"]

mcols = st.columns(7)
mcols[0].metric("Strat Total", f"{metrics['Total Return']:.2%}")
mcols[1].metric("BH Total", f"{(bh.iloc[-1]/bh.iloc[0]-1):.2%}")
mcols[2].metric("Ann Return", f"{metrics['Annualized Return']:.2%}")
mcols[3].metric("Ann Vol", f"{metrics['Annualized Volatility']:.2%}")
mcols[4].metric("Sharpe", f"{metrics['Sharpe (rf=0)']:.2f}")
mcols[5].metric("Max DD", f"{metrics['Max Drawdown']:.2%}")
mcols[6].metric("Trades", f"{metrics['Trades']}")

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy Equity"))
fig_eq.add_trace(go.Scatter(x=bh.index, y=bh.values, name="Buy&Hold Equity"))
fig_eq.update_layout(height=320)
st.plotly_chart(fig_eq, use_container_width=True)

st.subheader("📑 İşlemler")
if trades.empty:
    st.write("Trade oluşmadı. Modu Agresif yap veya periyodu büyüt.")
else:
    show = trades.copy()
    show["entry_date"] = show["entry_date"].astype(str)
    show["exit_date"] = show["exit_date"].astype(str)
    st.dataframe(show[["entry_date","entry_price","exit_date","exit_price","exit_reason","shares","pnl","return_%","holding_days"]],
                 use_container_width=True)
