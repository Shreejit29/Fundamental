# Streamlit Smart Stock Screener App
# Step-by-step chunked version (Indian Stock Market)

# =============================
# 📦 CHUNK 1: Import Libraries
# =============================
import streamlit as st
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from pytrends.request import TrendReq
import warnings
warnings.filterwarnings("ignore")
import ta
import numpy as np

# =============================
# 🧠 CHUNK 2: All Analysis Functions
# =============================

def fetch_stock_metrics_yahoo(ticker):
    try:
        info = yf.Ticker(ticker).info

        def safe(val, multiplier=1, default=0):
            return round((val if val is not None else default) * multiplier, 2)

        return {
            'ROE (%)': safe(info.get('returnOnEquity'), 100),
            'ROCE (%)': 18,  # Placeholder, not available from Yahoo
            'Promoter Holding (%)': safe(info.get('heldPercentInsiders'), 100),
            'Net Profit Margin (%)': safe(info.get('netMargins'), 100),
            'Profit Consistency': True,  # Placeholder or custom logic
            'Revenue CAGR 5Y (%)': 15,   # Placeholder
            'Profit CAGR 5Y (%)': 18,    # Placeholder
            'EPS Growth (%)': 12,        # Placeholder
            'Sales Growth YOY (%)': safe(info.get('revenueGrowth'), 100),
            'Debt to Equity': safe(info.get('debtToEquity'), 1, 1.0),
            'Interest Coverage': 10,     # Placeholder
            'Free Cash Flow Positive': 3,  # Placeholder (scale: 1-5)
            'Current Ratio': safe(info.get('currentRatio')),
            'PE Ratio': safe(info.get('trailingPE'), 1, 20),
            'PEG Ratio': safe(info.get('pegRatio'), 1, 1.5),
            'PB Ratio': safe(info.get('priceToBook'), 1, 2),
            'Discount to Intrinsic (%)': 12,  # Placeholder
            'Asset Turnover': 1.2,            # Placeholder
            'Inventory Turnover': 4,          # Placeholder
            'Receivables Days': 60            # Placeholder
        }

    except Exception as e:
        print(f"❌ Failed to fetch fundamentals for {ticker}: {e}")
        return None

def fundamental_analysis_engine(metrics):
    score = {
        "Business Quality": 0,
        "Growth": 0,
        "Financial Safety": 0,
        "Valuation": 0,
        "Efficiency": 0
    }

    # Weight of each category in final score
    weights = {
        "Business Quality": 0.30,
        "Growth": 0.25,
        "Financial Safety": 0.20,
        "Valuation": 0.15,
        "Efficiency": 0.10
    }

    # --- Business Quality ---
    if metrics['ROE (%)'] >= 15: score["Business Quality"] += 7
    if metrics['ROCE (%)'] >= 15: score["Business Quality"] += 7
    if metrics['Promoter Holding (%)'] >= 50: score["Business Quality"] += 7
    if metrics['Net Profit Margin (%)'] >= 10: score["Business Quality"] += 5
    if metrics['Profit Consistency']: score["Business Quality"] += 4

    # --- Growth ---
    if metrics['Revenue CAGR 5Y (%)'] >= 12: score["Growth"] += 8
    if metrics['Profit CAGR 5Y (%)'] >= 12: score["Growth"] += 8
    if metrics['EPS Growth (%)'] >= 10: score["Growth"] += 6
    if metrics['Sales Growth YOY (%)'] >= 10: score["Growth"] += 3

    # --- Financial Safety ---
    if metrics['Debt to Equity'] < 0.5: score["Financial Safety"] += 6
    if metrics['Interest Coverage'] > 5: score["Financial Safety"] += 5
    if metrics['Free Cash Flow Positive'] >= 3: score["Financial Safety"] += 5
    if metrics['Current Ratio'] > 1.5: score["Financial Safety"] += 4

    # --- Valuation ---
    if metrics['PE Ratio'] < 25: score["Valuation"] += 5
    if metrics['PEG Ratio'] < 1.2: score["Valuation"] += 5
    if metrics['PB Ratio'] < 4: score["Valuation"] += 3
    if metrics['Discount to Intrinsic (%)'] > 10: score["Valuation"] += 2

    # --- Efficiency ---
    if metrics['Asset Turnover'] > 1: score["Efficiency"] += 4
    if metrics['Inventory Turnover'] > 3: score["Efficiency"] += 3
    if metrics['Receivables Days'] < 90: score["Efficiency"] += 3

    # --- Final weighted score ---
    final_score = round(sum(score[cat] * weights[cat] for cat in score), 2)

    # --- Verdict based on final score (0–100 scale) ---
    if final_score >= 85:
        verdict = "🔥 Strong Buy"
    elif final_score >= 70:
        verdict = "✅ Buy / Hold"
    elif final_score >= 50:
        verdict = "⚠️ Hold / Avoid"
    else:
        verdict = "❌ Avoid"

    return final_score, verdict

def clean_yf_data(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    if 'Close' not in df.columns or df['Close'].isnull().all():
        return None
    df.dropna(subset=['Close'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df if not df.empty else None
# STEP 3: Compute Only Core Indicators
def compute_indicators(df):
    try:
        close = df['Close']
        high = df['High']
        low = df['Low']

        df['RSI'] = RSIIndicator(close=close, fillna=True).rsi()
        macd = MACD(close=close, fillna=True)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['EMA_20'] = EMAIndicator(close=close, window=20, fillna=True).ema_indicator()
        df['EMA_50'] = EMAIndicator(close=close, window=50, fillna=True).ema_indicator()
        df['EMA_200'] = EMAIndicator(close=close, window=200, fillna=True).ema_indicator()
        df['ADX'] = ADXIndicator(high=high, low=low, close=close, fillna=True).adx()
        df['VolumeSpike'] = df['Volume'] > 1.5 * df['Volume'].rolling(20).mean()


        df = df.tail(100).dropna().reset_index(drop=True)

        if df.empty:
            print("⚠️ No valid rows after indicator calculation.")
            return None

        print(f"✅ Indicators calculated: {len(df)} rows")
        return df

    except Exception as e:
        print(f"❌ Indicator error: {e}")
        return None
# STEP 4: Support / Resistance Logic
def detect_support_resistance(df):
    if df is None or len(df) < 20:
        return "❌ Not enough data", None, None

    recent_high = df['Close'].rolling(window=20).max().iloc[-1]
    recent_low = df['Close'].rolling(window=20).min().iloc[-1]
    current = df['Close'].iloc[-1]

    dist_res = (recent_high - current) / recent_high * 100
    dist_sup = (current - recent_low) / recent_low * 100

    if dist_res < 1:
        label = "⚠️ Near Resistance"
    elif dist_sup < 5:
        label = "✅ Near Support"
    else:
        label = "Neutral"

    return label, round(recent_low, 2), round(recent_high, 2)
# STEP 5: Full Analysis Function
def fetch_and_compute(symbol, period="6mo", interval="1d"):
    try:
        raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        df = clean_yf_data(raw)
        if df is None or len(df) < 60:
            print(f"⚠️ Skipping {symbol}: Insufficient data.")
            return None
        df = compute_indicators(df)
        return df
    except Exception as e:
        print(f"❌ Failed to fetch/process data for {symbol}: {e}")
        return None
# STEP 6: Run the Analysis
def analyze_stock(symbol):
    print(f"\n📊 Analyzing {symbol}")
    df = fetch_and_compute(symbol)

    if df is not None and not df.empty:
        latest = df.iloc[-1]
        print(f"Close: ₹{latest['Close']:.2f}")
        print(f"RSI: {latest['RSI']:.2f}")
        print(f"MACD: {latest['MACD']:.2f}")
        print(f"MACD Signal: {latest['MACD_signal']:.2f}")
        print(f"EMA 20: ₹{latest['EMA_20']:.2f}")
        print(f"EMA 50: ₹{latest['EMA_50']:.2f}")
        print(f"EMA 200: ₹{latest['EMA_200']:.2f}")
        print(f"ADX: {latest['ADX']:.2f}")

        zone, support, resistance = detect_support_resistance(df)
        print(f"Zone: {zone}")
        print(f"Support: ₹{support}, Resistance: ₹{resistance}")
    else:
        print("❌ Could not analyze the stock.")

def get_google_trend_score(stock_name):
    try:
        pytrends = TrendReq(hl='en-IN', tz=330)
        search_term = f"{stock_name} share"
        pytrends.build_payload([search_term], timeframe='today 7-d', geo='IN')
        data = pytrends.interest_over_time()

        if not data.empty and search_term in data.columns:
            return int(data[search_term].iloc[-1])
        else:
            return 0

    except Exception as e:
        print(f"⚠️ Google Trend error for {stock_name}: {e}")
        return 0

def detect_buzz_signals(df, ticker):
    try:
        latest_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].rolling(20).mean().iloc[-1]

        # Avoid divide by zero
        vol_spike_score = round(latest_vol / avg_vol, 2) if avg_vol else 0

        # Last 5-day return
        prev_close = df['Close'].iloc[-5]
        recent_return = round((df['Close'].iloc[-1] - prev_close) / prev_close * 100, 2) if prev_close else 0

        # Google trend (ticker prefix)
        trends_score = get_google_trend_score(ticker.split('.')[0])

        buzz_flags = []
        if vol_spike_score > 2.5:
            buzz_flags.append("⚠️ Volume Spike")
        if recent_return > 15:
            buzz_flags.append("⚠️ Momentum Jump")
        if trends_score > 60:
            buzz_flags.append("⚠️ High Search Buzz")

        buzz_verdict = ", ".join(buzz_flags) if buzz_flags else "✅ Normal Attention"

        return {
            "Volume Spike": vol_spike_score,
            "5-Day Return": recent_return,
            "Trends Score": trends_score,
            "Buzz Verdict": buzz_verdict
        }

    except Exception as e:
        print(f"❌ Buzz signal error for {ticker}: {e}")
        return {
            "Volume Spike": 0,
            "5-Day Return": 0,
            "Trends Score": 0,
            "Buzz Verdict": "❌ Unable to detect"
        }

def calculate_final_score(f_score, e_score, expected_return, buzz_verdict, s_r_label):
    # Scale entry score (max 6) to 100 and weight it
    weights = {
    'fundamental': 0.40,
    'entry': 0.20,
    'return': 0.15,
    'buzz_penalty': -15,
    'resistance_penalty': -10}

    entry_scaled = min(e_score * (100 / 6), 100)

    entry_weighted = entry_scaled * weights['entry']
    final_stock_score = (
    f_score * weights['fundamental'] +
    entry_weighted +
    return_score +
    (weights['buzz_penalty'] if "⚠️" in buzz_verdict else 0) +
    (weights['resistance_penalty'] if "Resistance" in s_r_label else 0))

    # Expected return scoring
    if expected_return >= 15:
        return_score = 15
    elif expected_return >= 10:
        return_score = 5
    else:
        return_score = 0

    # Penalties
    buzz_penalty = -15 if "⚠️" in buzz_verdict else 0
    resistance_penalty = -10 if "Resistance" in s_r_label else 0

    # Final weighted sum
    final_stock_score = (
        f_score * 0.40 +
        entry_weighted +
        return_score +
        buzz_penalty +
        resistance_penalty
    )

    final_stock_score = max(0, round(final_stock_score, 2))

    # Verdict based on score
    if final_stock_score >= 80:
        verdict = "🔥 Excellent – Invest"
    elif final_stock_score >= 65:
        verdict = "✅ Good Pick"
    elif final_stock_score >= 50:
        verdict = "⚠️ Okay but Risky"
    else:
        verdict = "❌ Avoid"

    return final_stock_score, verdict, {
    'Fundamental': f_score * 0.40,
    'Entry': entry_weighted,
    'Return': return_score,
    'Buzz Penalty': buzz_penalty,
    'Resistance Penalty': resistance_penalty}


# =============================
# 🎛️ CHUNK 3: Streamlit Sidebar UI & Presets
# =============================
st.set_page_config(page_title="Smart Stock Screener", layout="wide")
st.title("📈 Smart Screener: Indian Stocks with Fundamentals + Entry Logic")
st.markdown("Scan NSE stocks with strong fundamentals, entry signals, and volume/sentiment filters.")

st.sidebar.header("🛠️ Configuration")
run_bulk = st.sidebar.checkbox("Run Bulk Scan", value=True)

preset = st.sidebar.selectbox(
    "🎯 Choose Screener Preset",
    ["All Stocks", "Long-Term", "Momentum", "Swing Entry", "Undervalued"]
)

if run_bulk:
    stock_list = [
    s.strip().upper()
    for s in st.sidebar.text_area(...).split(",")
    if s.strip()]
    stock_list = st.sidebar.text_area(
        "Enter NSE tickers (comma-separated):",
        value="TCS.NS,INFY.NS,HDFCBANK.NS,RELIANCE.NS,DMART.NS"
    ).split(",")
    run_button = st.sidebar.button("🚀 Run Screener")

    if run_button:
        results = []
        with st.spinner("🔍 Scanning stocks..."):
            for ticker in stock_list:
                ticker = ticker.strip().upper()
                try:
                    metrics = fetch_stock_metrics_yahoo(ticker)
                    f_score, _ = fundamental_analysis_engine(metrics)
                    df = get_technical_indicators(ticker)
                    s_r_label, support, resistance = detect_support_resistance(df)
                    rsi = df['RSI'].iloc[-1]
                    macd_diff = df['MACD'].iloc[-1] - df['MACD_signal'].iloc[-1]
                    entry_details = {
                    "RSI < 35": int(rsi < 35),
                    "MACD > Signal": int(macd_diff > 0),
                    "Above 50DMA": int(df['Close'].iloc[-1] > df['50DMA'].iloc[-1]),
                    "Above 200DMA": int(df['Close'].iloc[-1] > df['200DMA'].iloc[-1]),
                    "Support Zone": int(s_r_label == "✅ Near Support"),
                    "Volume Spike": int(df['VolumeSpike'].iloc[-1])}
                    entry_score = sum(entry_details.values())

                    buzz = detect_buzz_signals(df, ticker)
                    expected_return = 18
                    final_score, verdict = calculate_final_score(
                        f_score, entry_score, expected_return,
                        buzz['Buzz Verdict'], s_r_label
                    )
                    results.append({
                        "Ticker": ticker,
                        "Smart Score": final_score,
                        "Verdict": verdict,
                        "Fundamental Score": f_score,
                        "Entry Score": entry_score,
                        "Buzz Verdict": buzz['Buzz Verdict'],
                        "Trend Score": buzz['Trends Score'],
                        "5D Return": buzz['5-Day Return'],
                        "Volume Spike": buzz['Volume Spike'],
                        "Support": support,
                        "Resistance": resistance,
                        "Entry Zone": s_r_label
                    })
                except Exception as e:
                    st.warning(f"⚠️ Error processing {ticker}: {e}")

        if results:
            df_final = pd.DataFrame(results)

            # Apply Screener Preset Filter
            if preset == "Long-Term":
                df_final = df_final[(df_final["Fundamental Score"] >= 75) &
                                    (~df_final["Buzz Verdict"].str.contains("⚠️")) &
                                    (df_final["Entry Zone"] != "⚠️ Near Resistance")]
            elif preset == "Momentum":
                df_final = df_final[(df_final["5D Return"] >= 10) & (df_final["Volume Spike"] >= 2.0)]
            elif preset == "Swing Entry":
                df_final = df_final[(df_final["Entry Zone"] == "✅ Near Support") & (df_final["Entry Score"] >= 4)]
            elif preset == "Undervalued":
                df_final = df_final[(df_final["Fundamental Score"] >= 65) &
                                    (df_final["Buzz Verdict"].str.contains("Normal")) &
                                    (df_final["Smart Score"] >= 60)]

            st.success("✅ Scan Complete")
            st.dataframe(df_final.sort_values(by="Smart Score", ascending=False), use_container_width=True)

# =============================
# 🔎 CHUNK 5: Single Stock Lookup Panel
# =============================
st.subheader("🔍 Analyze Single Stock")
single_ticker = st.text_input("Enter a single NSE stock ticker (e.g., RELIANCE.NS)")

if single_ticker:
    with st.spinner("Analyzing..."):
        try:
            single_ticker = single_ticker.upper()

            # Fetch data
            metrics = fetch_stock_metrics_yahoo(single_ticker)
            f_score, f_verdict = fundamental_analysis_engine(metrics)
            df = fetch_and_compute(single_ticker)  # <== fix

            if df is None or df.empty:
                raise ValueError("No technical data available.")

            s_r_label, support, resistance = detect_support_resistance(df)
            s_r_label = s_r_label or "❓ Unknown"
            support_fmt = f"₹{support:.2f}" if support else "-"
            resistance_fmt = f"₹{resistance:.2f}" if resistance else "-"

            # Entry logic
            rsi = df['RSI'].iloc[-1]
            macd_diff = df['MACD'].iloc[-1] - df['MACD_signal'].iloc[-1]
            entry_score = sum([
                rsi < 35,
                macd_diff > 0,
                df['Close'].iloc[-1] > df['EMA_50'].iloc[-1],
                df['Close'].iloc[-1] > df['EMA_200'].iloc[-1],
                s_r_label == "✅ Near Support",
                df['VolumeSpike'].iloc[-1]
            ])

            buzz = detect_buzz_signals(df, single_ticker)
            expected_return = 18
            final_score, verdict = calculate_final_score(
                f_score, entry_score, expected_return, buzz['Buzz Verdict'], s_r_label
            )

            # ✅ Final output
            st.markdown(f"### 🧾 Final Verdict: {verdict}")
            st.metric("📊 Smart Score", f"{final_score}/100")
            st.metric("📈 Fundamental Score", f"{f_score}/100", delta=f_verdict)
            st.metric("📥 Entry Score", f"{entry_score}/6")
            st.write(f"**RSI:** {rsi:.2f} | **MACD Diff:** {macd_diff:.2f}")
            st.write(f"**Support/Resistance:** {s_r_label} (S: {support_fmt}, R: {resistance_fmt})")
            st.write(f"**Buzz Check:** {buzz['Buzz Verdict']}")

        except Exception as e:
            st.error(f"❌ Could not analyze {single_ticker}: {e}")
