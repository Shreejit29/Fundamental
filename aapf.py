# Streamlit Smart Stock Screener App
# Step-by-step chunked version (Indian Stock Market)

# =============================
# 📦 CHUNK 1: Import Libraries
# =============================
import streamlit as st
import pandas as pd
import yfinance as yf
import ta
from pytrends.request import TrendReq

# =============================
# 🧠 CHUNK 2: All Analysis Functions
# =============================

def fetch_stock_metrics_yahoo(ticker):
    info = yf.Ticker(ticker).info
    return {
        'ROE': info.get('returnOnEquity', 0) * 100,
        'ROCE': 18,
        'Promoter Holding': info.get('heldPercentInsiders', 0) * 100,
        'Net Profit Margin': info.get('netMargins', 0) * 100,
        'Profit Consistency': True,
        'Revenue CAGR 5Y': 15,
        'Profit CAGR 5Y': 18,
        'EPS Growth': 12,
        'Sales Growth YOY': info.get('revenueGrowth', 0) * 100,
        'Debt to Equity': info.get('debtToEquity', 1.0),
        'Interest Coverage': 10,
        'Free Cash Flow Positive': 3,
        'Current Ratio': 1.8,
        'PE Ratio': info.get('trailingPE', 20),
        'PEG Ratio': info.get('pegRatio', 1.5),
        'PB Ratio': info.get('priceToBook', 2),
        'Discount to Intrinsic': 12,
        'Asset Turnover': 1.2,
        'Inventory Turnover': 4,
        'Receivables Days': 60
    }

def fundamental_analysis_engine(metrics):
    score = {
        "Business Quality": 0,
        "Growth": 0,
        "Financial Safety": 0,
        "Valuation": 0,
        "Efficiency": 0
    }
    weights = {
        "Business Quality": 0.30,
        "Growth": 0.25,
        "Financial Safety": 0.20,
        "Valuation": 0.15,
        "Efficiency": 0.10
    }
    if metrics['ROE'] >= 15: score["Business Quality"] += 7
    if metrics['ROCE'] >= 15: score["Business Quality"] += 7
    if metrics['Promoter Holding'] >= 50: score["Business Quality"] += 7
    if metrics['Net Profit Margin'] >= 10: score["Business Quality"] += 5
    if metrics['Profit Consistency']: score["Business Quality"] += 4
    if metrics['Revenue CAGR 5Y'] >= 12: score["Growth"] += 8
    if metrics['Profit CAGR 5Y'] >= 12: score["Growth"] += 8
    if metrics['EPS Growth'] >= 10: score["Growth"] += 6
    if metrics['Sales Growth YOY'] >= 10: score["Growth"] += 3
    if metrics['Debt to Equity'] < 0.5: score["Financial Safety"] += 6
    if metrics['Interest Coverage'] > 5: score["Financial Safety"] += 5
    if metrics['Free Cash Flow Positive'] >= 3: score["Financial Safety"] += 5
    if metrics['Current Ratio'] > 1.5: score["Financial Safety"] += 4
    if metrics['PE Ratio'] < 25: score["Valuation"] += 5
    if metrics['PEG Ratio'] < 1.2: score["Valuation"] += 5
    if metrics['PB Ratio'] < 4: score["Valuation"] += 3
    if metrics['Discount to Intrinsic'] > 10: score["Valuation"] += 2
    if metrics['Asset Turnover'] > 1: score["Efficiency"] += 4
    if metrics['Inventory Turnover'] > 3: score["Efficiency"] += 3
    if metrics['Receivables Days'] < 90: score["Efficiency"] += 3
    final_score = round(sum(score[k] * weights[k] for k in score), 2)
    if final_score >= 85: verdict = "🔥 Strong Buy"
    elif final_score >= 70: verdict = "✅ Buy / Hold"
    elif final_score >= 50: verdict = "⚠️ Hold / Avoid"
    else: verdict = "❌ Avoid"
    return final_score, verdict
def clean_yf_data(df):
    if df is None or df.empty:
        return None

    # Flatten MultiIndex columns (e.g., from group_by="ticker")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Ensure 'Close' column exists and isn't all NaN
    if 'Close' not in df.columns or df['Close'].isnull().all():
        return None

    # Drop rows with missing Close
    df.dropna(subset=['Close'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df if not df.empty else None

def get_technical_indicators(ticker, period="6mo", interval="1d"):
    try:
        df_raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        df = clean_yf_data(df_raw)

        if df is None or len(df) < 60:
            print(f"⚠️ Skipping {ticker} — not enough data after cleaning.")
            return None

        close = df["Close"].astype(float)

        # RSI
        rsi = ta.momentum.RSIIndicator(close=close).rsi()
        df["RSI"] = rsi.values.flatten()

        # MACD
        macd_calc = ta.trend.MACD(close=close)
        df["MACD"] = macd_calc.macd().values.flatten()
        df["MACD_signal"] = macd_calc.macd_signal().values.flatten()

        # 50/200 Day Moving Averages
        df["50DMA"] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator().values.flatten()
        df["200DMA"] = ta.trend.SMAIndicator(close=close, window=200).sma_indicator().values.flatten()

        # Volume-related indicators
        if 'Volume' in df.columns and not df['Volume'].isnull().all():
            df["AvgVolume20"] = df["Volume"].rolling(20).mean()
            df["VolumeSpike"] = df["Volume"] > 1.5 * df["AvgVolume20"]
        else:
            df["VolumeSpike"] = False

        df.dropna(inplace=True)
        if df.empty or len(df) < 10:
            print(f"⚠️ Indicator computation left {ticker} with too little data.")
            return None

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"❌ Error in {ticker}: {e}")
        return None

def detect_support_resistance(df):
    recent_high = df['Close'].rolling(window=20).max().iloc[-1]
    recent_low = df['Close'].rolling(window=20).min().iloc[-1]
    current_price = df['Close'].iloc[-1]
    distance_to_resistance = (recent_high - current_price) / recent_high * 100
    distance_to_support = (current_price - recent_low) / recent_low * 100
    if distance_to_resistance < 1:
        level = "⚠️ Near Resistance – Wait"
    elif distance_to_support < 5:
        level = "✅ Near Support – Good Entry"
    else:
        level = "Neutral"
    return level, round(recent_low, 2), round(recent_high, 2)

def get_google_trend_score(stock_name):
    try:
        pytrends = TrendReq(hl='en-IN', tz=330)
        search_term = f"{stock_name} share"
        pytrends.build_payload([search_term], timeframe='today 7-d', geo='IN')
        data = pytrends.interest_over_time()
        if not data.empty:
            return int(data[search_term].iloc[-1])
    except:
        return 0

def detect_buzz_signals(df, ticker):
    latest_vol = df['Volume'].iloc[-1]
    avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
    vol_spike_score = round(latest_vol / avg_vol, 2)
    recent_return = round((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100, 2)
    trends_score = get_google_trend_score(ticker.split('.')[0])
    buzz_flags = []
    if vol_spike_score > 2.5: buzz_flags.append("⚠️ Volume Spike")
    if recent_return > 15: buzz_flags.append("⚠️ Momentum Jump")
    if trends_score > 60: buzz_flags.append("⚠️ High Search Buzz")
    buzz_verdict = ", ".join(buzz_flags) if buzz_flags else "✅ Normal Attention"
    return {
        "Volume Spike": vol_spike_score,
        "5-Day Return": recent_return,
        "Trends Score": trends_score,
        "Buzz Verdict": buzz_verdict
    }

def calculate_final_score(f_score, e_score, expected_return, buzz_verdict, s_r_label):
    entry_scaled = min(e_score * (100 / 6), 100)
    entry_weighted = entry_scaled * 0.20
    return_score = 15 if expected_return >= 15 else (5 if expected_return >= 10 else 0)
    buzz_penalty = -15 if "⚠️" in buzz_verdict else 0
    resistance_penalty = -10 if "Resistance" in s_r_label else 0
    final_stock_score = (
        f_score * 0.40 +
        entry_weighted +
        return_score +
        buzz_penalty +
        resistance_penalty
    )
    final_stock_score = max(0, round(final_stock_score, 2))
    if final_stock_score >= 80:
        verdict = "🔥 Excellent – Invest"
    elif final_stock_score >= 65:
        verdict = "✅ Good Pick"
    elif final_stock_score >= 50:
        verdict = "⚠️ Okay but Risky"
    else:
        verdict = "❌ Avoid"
    return final_stock_score, verdict

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
                    entry_score = sum([
                        rsi < 35,
                        macd_diff > 0,
                        df['Close'].iloc[-1] > df['50DMA'].iloc[-1],
                        df['Close'].iloc[-1] > df['200DMA'].iloc[-1],
                        s_r_label == "✅ Near Support – Good Entry",
                        df['VolumeSpike'].iloc[-1]
                    ])
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
                                    (df_final["Entry Zone"] != "⚠️ Near Resistance – Wait")]
            elif preset == "Momentum":
                df_final = df_final[(df_final["5D Return"] >= 10) & (df_final["Volume Spike"] >= 2.0)]
            elif preset == "Swing Entry":
                df_final = df_final[(df_final["Entry Zone"] == "✅ Near Support – Good Entry") & (df_final["Entry Score"] >= 4)]
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
            metrics = fetch_stock_metrics_yahoo(single_ticker)
            f_score, f_verdict = fundamental_analysis_engine(metrics)
            df = get_technical_indicators(single_ticker)
            s_r_label, support, resistance = detect_support_resistance(df)
            rsi = df['RSI'].iloc[-1]
            macd_diff = df['MACD'].iloc[-1] - df['MACD_signal'].iloc[-1]
            entry_score = sum([
                rsi < 35,
                macd_diff > 0,
                df['Close'].iloc[-1] > df['50DMA'].iloc[-1],
                df['Close'].iloc[-1] > df['200DMA'].iloc[-1],
                s_r_label == "✅ Near Support – Good Entry",
                df['VolumeSpike'].iloc[-1]
            ])
            buzz = detect_buzz_signals(df, single_ticker)
            expected_return = 18
            final_score, verdict = calculate_final_score(
                f_score, entry_score, expected_return, buzz['Buzz Verdict'], s_r_label
            )
            st.markdown(f"### 🧾 Final Verdict: {verdict}")
            st.write(f"**Smart Score:** {final_score}")
            st.write(f"**Fundamental Score:** {f_score} ({f_verdict})")
            st.write(f"**Entry Score:** {entry_score} (RSI: {rsi:.2f}, MACD Diff: {macd_diff:.2f})")
            st.write(f"**Support-Resistance Zone:** {s_r_label} (S: ₹{support}, R: ₹{resistance})")
            st.write(f"**Buzz Check:** {buzz['Buzz Verdict']}")
        except Exception as e:
            st.error(f"❌ Could not analyze {single_ticker}: {e}")
