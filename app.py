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
# Data loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.dropna()

@st.cache_data(ttl=24*3600, show_spinner=False)
def get_sp500_tickers():
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    sp500 = tables[0]
    tickers = sp500["Symbol"].astype(str).tolist()
    return [t.replace(".", "-") for t in tickers]

@st.cache_data(ttl=24*3600, show_spinner=False)
def get_nasdaq100_tickers():
    tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any(("ticker" in c) or ("symbol" in c) for c in cols):
            df = t
            break
    if df is None:
        return []
    sym_col = None
    for c in df.columns:
        cl = str(c).lower()
        if "ticker" in cl or "symbol" in cl:
            sym_col = c
            break
    raw = df[sym_col].dropna().astype(str).str.strip().tolist()
    tickers = [t.replace(".", "-") for t in raw]
    tickers = [t for t in tickers if t and len(t) <= 10]
    return tickers

@st.cache_data(ttl=24*3600, show_spinner=False)
def get_bist100_tickers():
    """
    Tries multiple public pages to retrieve BIST100 (XU100) components.
    Returns yfinance-compatible tickers (e.g., THYAO.IS).
    Cached daily to avoid frequent scraping.
    """
    # 1) Yahoo Finance components (often works; if blocked, will fail gracefully)
    try:
        tables = pd.read_html("https://finance.yahoo.com/quote/XU100.IS/components/")
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("symbol" in c or "ticker" in c for c in cols):
                sym_col = None
                for c in t.columns:
                    cl = str(c).lower()
                    if "symbol" in cl or "ticker" in cl:
                        sym_col = c
                        break
                raw = t[sym_col].dropna().astype(str).str.strip().tolist()
                tickers = [x.replace(".", "-") for x in raw]
                tickers = [x if x.endswith(".IS") else f"{x}.IS" for x in tickers]
                tickers = [x for x in tickers if len(x) >= 4]
                if len(tickers) >= 50:
                    return tickers
    except Exception:
        pass

    # 2) TradingView components
    try:
        tables = pd.read_html("https://www.tradingview.com/symbols/BIST-XU100/components/")
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("symbol" in c for c in cols):
                sym_col = [c for c in t.columns if "symbol" in str(c).lower()][0]
                raw = t[sym_col].dropna().astype(str).str.strip().tolist()
                tickers = [x.replace(".", "-") for x in raw]
                tickers = [x if x.endswith(".IS") else f"{x}.IS" for x in tickers]
                tickers = [x for x in tickers if len(x) >= 4]
                if len(tickers) >= 50:
                    return tickers
    except Exception:
        pass

    # 3) Investing.com fallback (may require headers; read_html sometimes works)
    try:
        tables = pd.read_html("https://www.investing.com/indices/ise-100-components")
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            # look for a column that likely contains symbols
            if any("symbol" in c or "ticker" in c or "name" in c for c in cols):
                # Investing sometimes doesn't provide a clean symbol column; skip if unclear
                pass
    except Exception:
        pass

    return []

# -----------------------------
# Features + Signals
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

def market_filter_ok(cfg: dict) -> bool:
    """SPY rejim filtresi: SPY Close > EMA200 ise long'lar aktif."""
    if not cfg.get("use_spy_filter", True):
        return True
    spy = load_data("SPY", period="5y", interval="1d")
    if spy.empty or len(spy) < 260:
        return True  # fail-open
    spy["EMA200"] = ema(spy["Close"], 200)
    last = spy.iloc[-1]
    return bool(last["Close"] > last["EMA200"])

def signal_with_checkpoints(df: pd.DataFrame, cfg: dict):
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

    # More defensive weighting: trend heavier
    w = {"liq": 10, "trend": 35, "rsi": 15, "macd": 15, "vol": 10, "bb": 5, "obv": 10}
    score = (
        w["liq"] * liq_ok.astype(int) +
        w["trend"] * trend_ok.astype(int) +
        w["rsi"] * rsi_ok.astype(int) +
        w["macd"] * macd_ok.astype(int) +
        w["vol"] * vol_ok.astype(int) +
        w["bb"] * (bb_ok | bb_break).astype(int) +
        w["obv"] * obv_ok.astype(int)
    ).astype(float)

    # Entry: market filter + trend + vol + liquidity + at least 2 triggers
    entry_triggers = (rsi_cross.astype(int) + macd_turn.astype(int) + bb_break.astype(int)) >= 2
    mkt_ok = market_filter_ok(cfg)
    entry = (trend_ok & vol_ok & liq_ok & entry_triggers) & mkt_ok

    # Exit: weakness
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
        "Liquidity (Vol > VolSMA)": bool(last["Volume"] > last["VOL_SMA"]) if pd.notna(last["VOL_SMA"]) else False,
        "Trend (Close>EMA200 & EMA50>EMA200)": bool((last["Close"] > last["EMA200"]) and (last["EMA50"] > last["EMA200"])) if pd.notna(last["EMA200"]) else False,
        f"RSI > {cfg['rsi_entry_level']}": bool(last["RSI"] > cfg["rsi_entry_level"]) if pd.notna(last["RSI"]) else False,
        "MACD Hist > 0": bool(last["MACD_hist"] > 0) if pd.notna(last["MACD_hist"]) else False,
        f"ATR% < {cfg['atr_pct_max']:.2%}": bool((last["ATR"] / last["Close"]) < cfg["atr_pct_max"]) if pd.notna(last["ATR"]) else False,
        "Bollinger (Close>BB_mid or Breakout)": bool((last["Close"] > last["BB_mid"]) or (last["Close"] > last["BB_upper"])) if pd.notna(last["BB_mid"]) else False,
        "OBV > OBV_EMA": bool(last["OBV"] > last["OBV_EMA"]) if pd.notna(last["OBV_EMA"]) else False,
    }

    return df, cp

# -----------------------------
# Backtest (long-only, correct accounting)
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
# Scanner (manual)
# -----------------------------
@st.cache_data(show_spinner=False)
def scan_universe(tickers, cfg, period="5y", interval="1d", max_symbols=150):
    results = []
    tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
    tickers = tickers[:max_symbols]

    for tk in tickers:
        try:
            df = load_data(tk, period=period, interval=interval)
            if df.empty or len(df) < 260:
                continue
            dff = build_features(df, cfg)
            dff, _ = signal_with_checkpoints(dff, cfg)
            last = dff.iloc[-1]
            if int(last["ENTRY"]) == 1:
                results.append((tk, float(last["SCORE"]), float(last["Close"])))
        except Exception:
            continue

    results.sort(key=lambda x: x[1], reverse=True)
    return results

# -----------------------------
# UI
# -----------------------------
st.title("📈 Trading Sinyal Uygulaması (ABD + BIST100)")
st.caption("Hedef: az işlem ama daha sağlam sinyaller. Otomatik emir yok. Manual scan + detay + backtest.")

with st.sidebar:
    st.header("Piyasa / Evren")
    market = st.selectbox("Market", ["ABD", "BIST100"], index=0)
    us_universe = st.selectbox("ABD Evreni", ["Watchlist", "Nasdaq-100", "S&P 500"], index=1) if market == "ABD" else None

    st.divider()
    st.header("Veri")
    period = st.selectbox("Periyot", ["1y", "2y", "5y", "10y"], index=2)
    interval = st.selectbox("Interval", ["1d", "1h"], index=0)

    st.divider()
    st.header("Sağlamlık Filtresi")
    use_spy_filter = st.checkbox("SPY Market Filter (SPY > EMA200)", value=True, help="ABD için ayı piyasasında long sinyallerini azaltır.")

    st.divider()
    st.header("Defansif Strateji Parametreleri")
    ema_fast = st.number_input("EMA Fast", 5, 100, 50, 1)
    ema_slow = st.number_input("EMA Slow", 50, 400, 200, 1)
    rsi_period = st.number_input("RSI Period", 5, 30, 14, 1)
    rsi_entry_level = st.number_input("RSI Entry", 40, 60, 52, 1)
    rsi_exit_level = st.number_input("RSI Exit", 30, 55, 46, 1)
    bb_period = st.number_input("BB Period", 10, 50, 20, 1)
    bb_std = st.number_input("BB Std", 1.0, 3.5, 2.0, 0.1)
    atr_period = st.number_input("ATR Period", 5, 30, 14, 1)
    atr_pct_max = st.slider("ATR% Max", 0.01, 0.20, 0.06, 0.01)
    vol_sma = st.number_input("Volume SMA", 5, 60, 20, 1)

    st.divider()
    st.header("Backtest / Risk")
    initial_capital = st.number_input("Başlangıç Sermayesi", min_value=100.0, value=10000.0, step=500.0)
    risk_per_trade = st.slider("Trade başı risk (equity %)", 0.002, 0.05, 0.01, 0.001)
    atr_stop_mult = st.slider("ATR Stop Katsayısı", 1.0, 6.0, 3.5, 0.5)
    commission_bps = st.number_input("Komisyon (bps)", min_value=0.0, value=5.0, step=1.0)
    slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, value=2.0, step=1.0)

    st.divider()
    st.header("Scanner")
    max_symbols = st.slider("Max taranacak sembol", 20, 400, 150, 10, help="Streamlit Cloud için 150-250 arası öneririm.")
    scan_btn = st.button("🔎 Scan (AL sinyallerini bul)", type="primary")

cfg = {
    "ema_fast": ema_fast,
    "ema_slow": ema_slow,
    "rsi_period": rsi_period,
    "rsi_entry_level": rsi_entry_level,
    "rsi_exit_level": rsi_exit_level,
    "bb_period": bb_period,
    "bb_std": bb_std,
    "atr_period": atr_period,
    "atr_pct_max": atr_pct_max,
    "vol_sma": vol_sma,
    "initial_capital": initial_capital,
    "risk_per_trade": risk_per_trade,
    "atr_stop_mult": atr_stop_mult,
    "commission_bps": commission_bps,
    "slippage_bps": slippage_bps,
    "use_spy_filter": use_spy_filter,
}

US_WATCHLIST = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","NFLX","JPM","XOM","SPY","QQQ"]

if market == "ABD":
    if us_universe == "Watchlist":
        universe = US_WATCHLIST
    elif us_universe == "Nasdaq-100":
        universe = get_nasdaq100_tickers()
    else:
        universe = get_sp500_tickers()
else:
    universe = get_bist100_tickers()

st.subheader("🔎 Scanner Sonuçları (ENTRY = AL)")
st.write(f"Seçili Market: **{market}** | Evren: **{us_universe if market=='ABD' else 'XU100'}** | Evren büyüklüğü: **{len(universe)}** | Tarama limiti: **{max_symbols}**")

if market == "BIST100" and len(universe) == 0:
    st.warning("BIST100 bileşenleri çekilemedi (kaynak engeli/format değişimi olabilir). "
               "Biraz sonra tekrar dene veya farklı periyot/interval seç.")

if scan_btn:
    with st.spinner("Taranıyor..."):
        buys = scan_universe(universe, cfg=cfg, period=period, interval=interval, max_symbols=max_symbols)

    if not buys:
        st.warning("Bu taramada AL sinyali bulunmadı.")
        st.stop()

    out = pd.DataFrame(buys, columns=["Ticker", "Score", "Close"])
    st.dataframe(out, use_container_width=True)

    chosen = st.selectbox("Detay görmek için bir ticker seç", out["Ticker"].tolist(), index=0)
    st.divider()
    st.subheader(f"📌 Detay: {chosen}")

    df_raw = load_data(chosen, period=period, interval=interval)
    if df_raw.empty or len(df_raw) < 260:
        st.error("Seçilen sembolde veri yetersiz. Daha uzun periyot seç.")
        st.stop()

    df = build_features(df_raw, cfg)
    df, checkpoints = signal_with_checkpoints(df, cfg)
    latest = df.iloc[-1]

    rec = "BEKLE"
    if int(latest["ENTRY"]) == 1:
        rec = "AL"
    elif int(latest["EXIT"]) == 1:
        rec = "SAT"
    else:
        rec = "İZLE (Güçlü Trend)" if latest["SCORE"] >= 85 else ("BEKLE (Orta)" if latest["SCORE"] >= 70 else "UZAK DUR")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Son Fiyat", f"{latest['Close']:.2f}")
    c2.metric("Skor", f"{latest['SCORE']:.0f}/100")
    c3.metric("Sinyal", rec)
    c4.metric("Market Filter", "AÇIK" if use_spy_filter else "KAPALI")

    st.markdown("### ✅ Kontrol Noktaları (Son Bar)")
    cols = st.columns(3)
    for i, (k, v) in enumerate(checkpoints.items()):
        with cols[i % 3]:
            st.write(("🟢 " if v else "🔴 ") + k)

    st.markdown("### 📊 Fiyat + EMA + Bollinger")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA Fast"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA Slow"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], name="BB Mid", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(dash="dot")))
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📉 RSI / MACD / ATR%")
    i1, i2, i3 = st.columns(3)

    with i1:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
        fig_rsi.add_hline(y=cfg["rsi_entry_level"])
        fig_rsi.add_hline(y=cfg["rsi_exit_level"])
        fig_rsi.update_layout(height=250)
        st.plotly_chart(fig_rsi, use_container_width=True)

    with i2:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"))
        fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
        fig_macd.update_layout(height=250)
        st.plotly_chart(fig_macd, use_container_width=True)

    with i3:
        atr_pct = (df["ATR"] / df["Close"]).replace([np.inf, -np.inf], np.nan)
        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(x=df.index, y=atr_pct, name="ATR%"))
        fig_atr.add_hline(y=cfg["atr_pct_max"])
        fig_atr.update_layout(height=250, yaxis_tickformat=".1%")
        st.plotly_chart(fig_atr, use_container_width=True)

    st.markdown("### 🧪 Backtest (Long-only)")
    eq, trades, metrics = backtest_long_only(df, cfg)

    mcols = st.columns(6)
    mcols[0].metric("Total Return", f"{metrics['Total Return']:.2%}")
    mcols[1].metric("Ann Return", f"{metrics['Annualized Return']:.2%}")
    mcols[2].metric("Ann Vol", f"{metrics['Annualized Volatility']:.2%}")
    mcols[3].metric("Sharpe", f"{metrics['Sharpe (rf=0)']:.2f}")
    mcols[4].metric("Max DD", f"{metrics['Max Drawdown']:.2%}")
    mcols[5].metric("Trades", f"{metrics['Trades']}")

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity"))
    fig_eq.update_layout(height=300)
    st.plotly_chart(fig_eq, use_container_width=True)

    st.markdown("### 📑 İşlemler")
    if trades is None or trades.empty:
        st.write("Trade oluşmadı (bu defansif modda normal olabilir).")
    else:
        show = trades.copy()
        show["entry_date"] = show["entry_date"].astype(str)
        show["exit_date"] = show["exit_date"].astype(str)
        st.dataframe(
            show[["entry_date","entry_price","exit_date","exit_price","exit_reason","shares","pnl","return_%","holding_days"]],
            use_container_width=True
        )
else:
    st.info("Tarama yapmak için soldan **Scan** butonuna bas.")
