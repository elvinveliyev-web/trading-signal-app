import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="AI-Style Trading Signal App", layout="wide")

# -----------------------------
# Helpers: Indicators
# -----------------------------
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def rsi(close, period=14):
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=close.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close, period=20, std_mult=2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = mid + std_mult * sd
    lower = mid - std_mult * sd
    return mid, upper, lower

def true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(high, low, close, period=14):
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = (equity_curve / peak) - 1.0
    return float(dd.min())

# -----------------------------
# Strategy (Rule-based, "AI-like" scoring + checkpoints)
# Indicators: EMA50, EMA200, RSI14, MACD hist, Bollinger, ATR, OBV
# -----------------------------
def build_features(df, cfg):
    df = df.copy()
    df["EMA50"] = ema(df["Close"], cfg["ema_fast"])
    df["EMA200"] = ema(df["Close"], cfg["ema_slow"])
    df["RSI"] = rsi(df["Close"], cfg["rsi_period"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(
        df["Close"], cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"]
    )
    df["BB_mid"], df["BB_upper"], df["BB_lower"] = bollinger(
        df["Close"], cfg["bb_period"], cfg["bb_std"]
    )
    df["ATR"] = atr(df["High"], df["Low"], df["Close"], cfg["atr_period"])
    df["OBV"] = obv(df["Close"], df["Volume"])
    df["VOL_SMA"] = df["Volume"].rolling(cfg["vol_sma"]).mean()

    # Useful extras
    df["RET"] = df["Close"].pct_change()
    df["OBV_EMA"] = ema(df["OBV"], 21)
    return df

def signal_with_checkpoints(df, cfg):
    """
    Produces:
      - score (0..100)
      - entry signal (1/0)
      - exit signal (1/0)
      - checkpoint dict for last bar
    """
    df = df.copy()

    # Checkpoints (boolean conditions)
    # 1) Liquidity/volume check
    liq_ok = (df["Volume"] > df["VOL_SMA"]).fillna(False)

    # 2) Trend filter
    trend_ok = (df["Close"] > df["EMA200"]) & (df["EMA50"] > df["EMA200"])

    # 3) Momentum confirmation (RSI regime + cross)
    rsi_ok = df["RSI"] > cfg["rsi_entry_level"]
    rsi_cross = (df["RSI"] > cfg["rsi_entry_level"]) & (df["RSI"].shift(1) <= cfg["rsi_entry_level"])

    # 4) MACD confirmation
    macd_ok = df["MACD_hist"] > 0
    macd_turn = (df["MACD_hist"] > 0) & (df["MACD_hist"].shift(1) <= 0)

    # 5) Volatility/risk filter (avoid too wild days)
    # Use ATR% threshold
    atr_pct = (df["ATR"] / df["Close"]).replace([np.inf, -np.inf], np.nan)
    vol_ok = atr_pct < cfg["atr_pct_max"]

    # 6) Breakout / mean reversion blend using Bollinger
    # We prefer: price above BB mid, or breakout above upper if trend strong
    bb_ok = df["Close"] > df["BB_mid"]
    bb_break = (df["Close"] > df["BB_upper"]) & trend_ok

    # 7) OBV confirmation (volume flow)
    obv_ok = df["OBV"] > df["OBV_EMA"]

    # Score system (AI-like aggregation)
    # Weights sum to 100
    w = {
        "liq": 10,
        "trend": 25,
        "rsi": 15,
        "macd": 15,
        "vol": 10,
        "bb": 15,
        "obv": 10,
    }

    score = (
        w["liq"] * liq_ok.astype(int) +
        w["trend"] * trend_ok.astype(int) +
        w["rsi"] * rsi_ok.astype(int) +
        w["macd"] * macd_ok.astype(int) +
        w["vol"] * vol_ok.astype(int) +
        w["bb"] * (bb_ok | bb_break).astype(int) +
        w["obv"] * obv_ok.astype(int)
    ).astype(float)

    # Entry rule: must pass trend + volatility + and at least 2 of (RSI cross, MACD turn, BB breakout)
    entry_triggers = (rsi_cross.astype(int) + macd_turn.astype(int) + bb_break.astype(int)) >= 2
    entry = trend_ok & vol_ok & liq_ok & entry_triggers

    # Exit rule: any strong deterioration
    # - Close drops below EMA50 OR
    # - MACD hist turns negative OR
    # - RSI falls below exit level OR
    # - Close falls below BB mid (weakness)
    exit_ = (
        (df["Close"] < df["EMA50"]) |
        (df["MACD_hist"] < 0) |
        (df["RSI"] < cfg["rsi_exit_level"]) |
        (df["Close"] < df["BB_mid"])
    )

    df["SCORE"] = score
    df["ENTRY"] = entry.astype(int)
    df["EXIT"] = exit_.astype(int)

    # Build checkpoints for the last bar
    last = df.iloc[-1]
    cp = {
        "Liquidity (Volume > VolSMA)": bool(last["Volume"] > last["VOL_SMA"]) if pd.notna(last["VOL_SMA"]) else False,
        "Trend (Close>EMA200 & EMA50>EMA200)": bool((last["Close"] > last["EMA200"]) and (last["EMA50"] > last["EMA200"])) if pd.notna(last["EMA200"]) else False,
        f"RSI > {cfg['rsi_entry_level']}": bool(last["RSI"] > cfg["rsi_entry_level"]) if pd.notna(last["RSI"]) else False,
        "MACD Hist > 0": bool(last["MACD_hist"] > 0) if pd.notna(last["MACD_hist"]) else False,
        f"ATR% < {cfg['atr_pct_max']:.2%}": bool((last["ATR"] / last["Close"]) < cfg["atr_pct_max"]) if pd.notna(last["ATR"]) else False,
        "Bollinger (Close>BB_mid or Breakout)": bool((last["Close"] > last["BB_mid"]) or (last["Close"] > last["BB_upper"])) if pd.notna(last["BB_mid"]) else False,
        "OBV > OBV_EMA": bool(last["OBV"] > last["OBV_EMA"]) if pd.notna(last["OBV_EMA"]) else False,
    }

    return df, cp

def backtest_long_only(df, cfg):
    """
    Long-only backtest:
      - Enter next bar after ENTRY signal.
      - Exit next bar after EXIT signal, or ATR trailing stop.
      - Position sizing: fixed fraction of equity risked per trade using ATR stop distance.
    """
    df = df.copy()

    # Use shifted signals to avoid lookahead
    entry_sig = df["ENTRY"].shift(1).fillna(0).astype(int)
    exit_sig = df["EXIT"].shift(1).fillna(0).astype(int)

    equity = cfg["initial_capital"]
    in_pos = False
    entry_price = 0.0
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

        # Update trailing stop if in position
        if in_pos and pd.notna(row["ATR"]):
            # ATR trailing stop under price
            new_stop = price - cfg["atr_stop_mult"] * float(row["ATR"])
            stop = max(stop, new_stop)  # trail upwards only

        # Exit logic (stop or exit signal)
        if in_pos:
            stop_hit = (not math.isnan(stop)) and (price <= stop)
            if exit_sig.iloc[i] == 1 or stop_hit:
                # Sell with slippage and commission
                sell_price = price * (1 - slippage)
                gross = shares * sell_price
                fee = gross * commission
                equity = gross - fee

                trades[-1]["exit_date"] = date
                trades[-1]["exit_price"] = sell_price
                trades[-1]["exit_reason"] = "STOP" if stop_hit else "RULE_EXIT"
                trades[-1]["pnl"] = equity - trades[-1]["equity_before"]

                in_pos = False
                entry_price = 0.0
                shares = 0.0
                stop = np.nan

        # Entry logic
        if (not in_pos) and entry_sig.iloc[i] == 1 and pd.notna(row["ATR"]) and row["ATR"] > 0:
            # Risk-based sizing: risk_per_trade% of equity
            risk_cash = equity * cfg["risk_per_trade"]
            stop_dist = cfg["atr_stop_mult"] * float(row["ATR"])
            if stop_dist > 0:
                # shares such that stop loss ~= risk_cash
                raw_shares = risk_cash / stop_dist
                # buy with slippage and commission
                buy_price = price * (1 + slippage)
                cost = raw_shares * buy_price
                fee = cost * commission
                total_cost = cost + fee

                if total_cost <= equity:  # enough cash
                    shares = raw_shares
                    equity_before = equity
                    equity = equity - total_cost  # cash left is 0-ish but we mark equity as position value later
                    entry_price = buy_price
                    stop = entry_price - cfg["atr_stop_mult"] * float(row["ATR"])

                    trades.append({
                        "entry_date": date,
                        "entry_price": entry_price,
                        "exit_date": None,
                        "exit_price": None,
                        "exit_reason": None,
                        "shares": shares,
                        "equity_before": equity_before,
                        "pnl": None
                    })
                    in_pos = True

        # Mark-to-market equity: cash + position value
        if in_pos:
            mtm = shares * price * (1 - slippage)  # conservative
            total_equity = mtm  # since we used most cash to buy
        else:
            total_equity = equity

        equity_curve.append((date, total_equity))

    eq = pd.Series([v for _, v in equity_curve], index=[d for d, _ in equity_curve], name="equity").astype(float)
    eq = eq.replace([np.inf, -np.inf], np.nan).dropna()

    # Metrics
    ret = eq.pct_change().dropna()
    total_return = eq.iloc[-1] / eq.iloc[0] - 1 if len(eq) > 1 else 0.0
    ann_return = (1 + total_return) ** (252 / max(1, len(ret))) - 1 if len(ret) > 0 else 0.0
    ann_vol = float(ret.std() * math.sqrt(252)) if len(ret) > 1 else 0.0
    sharpe = float((ret.mean() * 252) / (ret.std() * math.sqrt(252))) if len(ret) > 1 and ret.std() > 0 else 0.0
    mdd = max_drawdown(eq)

    # Clean trades dataframe
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
# UI
# -----------------------------
st.title("📈 Sinyal Üreten Trading Uygulaması (7 indikatör + kontrol noktaları)")
st.caption("Otomatik emir göndermez. Trend + momentum + volatilite + hacim teyidi ile AL/SAT/BEKLE üretir.")

with st.sidebar:
    st.header("Veri")
    ticker = st.text_input("Sembol (ör: AAPL, MSFT, TSLA, BTC-USD)", value="AAPL").strip().upper()
    period = st.selectbox("Periyot", ["6mo", "1y", "2y", "5y", "10y"], index=1)
    interval = st.selectbox("Interval", ["1d", "1h", "30m"], index=0)
    st.divider()

    st.header("Strateji Parametreleri")
    cfg = {
        "ema_fast": st.number_input("EMA Fast (trend içi)", min_value=5, max_value=100, value=50, step=1),
        "ema_slow": st.number_input("EMA Slow (trend filtresi)", min_value=50, max_value=400, value=200, step=1),
        "rsi_period": st.number_input("RSI Period", min_value=5, max_value=30, value=14, step=1),
        "rsi_entry_level": st.number_input("RSI Entry Level", min_value=40, max_value=60, value=50, step=1),
        "rsi_exit_level": st.number_input("RSI Exit Level", min_value=30, max_value=55, value=45, step=1),
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": st.number_input("Bollinger Period", min_value=10, max_value=50, value=20, step=1),
        "bb_std": st.number_input("Bollinger Std", min_value=1.0, max_value=3.5, value=2.0, step=0.1),
        "atr_period": st.number_input("ATR Period", min_value=5, max_value=30, value=14, step=1),
        "atr_pct_max": st.slider("ATR% Max (vol filtresi)", min_value=0.01, max_value=0.20, value=0.08, step=0.01),
        "vol_sma": st.number_input("Volume SMA", min_value=5, max_value=60, value=20, step=1),
    }

    st.header("Risk / Backtest")
    cfg["initial_capital"] = st.number_input("Başlangıç Sermayesi", min_value=100.0, value=10000.0, step=500.0)
    cfg["risk_per_trade"] = st.slider("Trade başı risk (equity %)", min_value=0.002, max_value=0.05, value=0.01, step=0.001)
    cfg["atr_stop_mult"] = st.slider("ATR Stop Katsayısı", min_value=1.0, max_value=6.0, value=3.0, step=0.5)
    cfg["commission_bps"] = st.number_input("Komisyon (bps)", min_value=0.0, value=5.0, step=1.0)
    cfg["slippage_bps"] = st.number_input("Slippage (bps)", min_value=0.0, value=2.0, step=1.0)

    run_btn = st.button("🚀 Çalıştır", type="primary")

if not run_btn:
    st.info("Soldan sembol ve parametreleri seçip **Çalıştır**’a bas.")
    st.stop()

# -----------------------------
# Data Load
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    # Normalize columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna()
    return df

with st.spinner("Veri indiriliyor..."):
    df_raw = load_data(ticker, period, interval)

if df_raw.empty or len(df_raw) < 260:
    st.error("Yetersiz veri. Daha uzun periyot seç (ör. 2y/5y) veya farklı sembol dene.")
    st.stop()

# Basic sanity checks
if (df_raw["Volume"].median() == 0) and ("USD" not in ticker):
    st.warning("Hacim verisi zayıf/0 görünüyor. Bazı sembollerde yfinance hacim düzgün gelmeyebilir.")

# -----------------------------
# Compute features + signals
# -----------------------------
df = build_features(df_raw, cfg)
df, checkpoints = signal_with_checkpoints(df, cfg)

latest = df.iloc[-1]
prev = df.iloc[-2]

# Decide latest recommendation
# - If ENTRY today -> "AL"
# - Else if EXIT today -> "SAT"
# - Else use SCORE as "BEKLE / İZLE"
rec = "BEKLE"
if int(latest["ENTRY"]) == 1:
    rec = "AL"
elif int(latest["EXIT"]) == 1:
    rec = "SAT"
else:
    if latest["SCORE"] >= 80:
        rec = "İZLE (Güçlü Trend)"
    elif latest["SCORE"] >= 60:
        rec = "BEKLE (Orta)"
    else:
        rec = "UZAK DUR (Zayıf)"

# -----------------------------
# Layout
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sembol", ticker)
c2.metric("Son Fiyat", f"{latest['Close']:.2f}")
c3.metric("Skor", f"{latest['SCORE']:.0f}/100")
c4.metric("Sinyal", rec)

st.subheader("✅ Kontrol Noktaları (Son Bar)")
cp_cols = st.columns(3)
cp_items = list(checkpoints.items())
for i, (k, v) in enumerate(cp_items):
    with cp_cols[i % 3]:
        st.write(("🟢 " if v else "🔴 ") + k)

# -----------------------------
# Charts
# -----------------------------
st.subheader("📊 Fiyat + EMA + Bollinger + Sinyaller")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Price"
))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA200"))
fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], name="BB Mid", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(dash="dot")))

# Entry/Exit markers
entries = df[df["ENTRY"] == 1]
exits = df[df["EXIT"] == 1]
fig.add_trace(go.Scatter(
    x=entries.index, y=entries["Close"], mode="markers", name="ENTRY",
    marker=dict(symbol="triangle-up", size=10)
))
fig.add_trace(go.Scatter(
    x=exits.index, y=exits["Close"], mode="markers", name="EXIT",
    marker=dict(symbol="triangle-down", size=10)
))

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

# -----------------------------
# Backtest
# -----------------------------
st.subheader("🧪 Backtest (Long-only, ATR trailing stop + risk-based sizing)")
eq, trades, metrics = backtest_long_only(df, cfg)

mcols = st.columns(6)
mcols[0].metric("Total Return", f"{metrics['Total Return']:.2%}")
mcols[1].metric("Ann. Return", f"{metrics['Annualized Return']:.2%}")
mcols[2].metric("Ann. Vol", f"{metrics['Annualized Volatility']:.2%}")
mcols[3].metric("Sharpe", f"{metrics['Sharpe (rf=0)']:.2f}")
mcols[4].metric("Max DD", f"{metrics['Max Drawdown']:.2%}")
mcols[5].metric("Win Rate", f"{metrics['Win Rate']:.2%}")

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity"))
fig_eq.update_layout(height=300)
st.plotly_chart(fig_eq, use_container_width=True)

st.subheader("📑 İşlemler")
if trades.empty:
    st.write("Trade oluşmadı. Parametreleri değiştir (periyot büyüt, ATR% filtresini gevşet, vb.).")
else:
    show = trades.copy()
    show["entry_date"] = show["entry_date"].astype(str)
    show["exit_date"] = show["exit_date"].astype(str)
    st.dataframe(show[["entry_date","entry_price","exit_date","exit_price","exit_reason","shares","pnl","return_%","holding_days"]], use_container_width=True)

st.subheader("🔎 Son Bar Özeti")
st.write(
    f"- Close: **{latest['Close']:.2f}** | EMA50: **{latest['EMA50']:.2f}** | EMA200: **{latest['EMA200']:.2f}**\n"
    f"- RSI: **{latest['RSI']:.1f}** | MACD Hist: **{latest['MACD_hist']:.4f}**\n"
    f"- BB Mid: **{latest['BB_mid']:.2f}** | ATR: **{latest['ATR']:.2f}** | Volume: **{latest['Volume']:.0f}**\n"
)
