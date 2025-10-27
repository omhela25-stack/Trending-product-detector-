import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="AI Trend Analyzer", layout="wide", page_icon="📊")

st.markdown(
    """
    <h1 style="text-align:center; color:#4F8BF9;">🤖 AI-Powered Google Trends Dashboard</h1>
    <p style="text-align:center;">Discover what's trending and where it's headed — powered by LSTM AI prediction.</p>
    """,
    unsafe_allow_html=True
)

# ------------------------ USER INPUT ------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    default_keywords = "smartwatch, wireless earbuds, sneakers, perfume, power bank"
    keywords = st.text_input("Enter product keywords (comma-separated):", default_keywords)
    keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
    region = st.selectbox("🌍 Region", ["Worldwide", "IN", "US"], index=1)
    timeframe = st.selectbox("🕒 Timeframe", ["today 3-m", "today 12-m"], index=0)
    future_days = st.slider("🔮 Predict Future Days", 3, 14, 7)
    st.markdown("---")
    st.info("Tip: Use short, popular keywords for better results!")

# ------------------------ BACKEND HELPERS ------------------------
@st.cache_data(ttl=3600)
def get_google_trends(keywords, timeframe="today 3-m", region="IN"):
    """Fetch Google Trends data safely with retries."""
    pytrends = TrendReq()
    trends_data = pd.DataFrame()

    for attempt in range(3):
        try:
            pytrends.build_payload(keywords, timeframe=timeframe, geo=region)
            trends_data = pytrends.interest_over_time()
            if not trends_data.empty and "isPartial" in trends_data.columns:
                trends_data = trends_data.drop(columns=["isPartial"])
            break
        except Exception:
            time.sleep(2)
            continue

    return trends_data


def predict_trend_lstm(series, future_steps=7, seq_len=10, epochs=3, batch_size=4):
    """Predict short-term trend using LSTM."""
    if len(series) < seq_len + 2:
        return np.array([])

    series = series.fillna(method="ffill").fillna(method="bfill")
    series = series[-(seq_len * 3):]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(25, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    predictions = []
    current_seq = X[-1]
    for _ in range(future_steps):
        pred = model.predict(current_seq.reshape(1, seq_len, 1), verbose=0)[0, 0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


def trend_direction(values):
    """Classify trend direction."""
    if len(values) < 2:
        return "⚪ Unknown"
    diff = values[-1] - values[0]
    if diff > 5:
        return "🟢 Rising"
    elif diff < -5:
        return "🔴 Falling"
    else:
        return "🟡 Stable"

# ------------------------ MAIN DISPLAY ------------------------
trend_summary = []

for keyword in keywords_list:
    with st.spinner(f"Fetching and analyzing data for **{keyword}**..."):
        st.markdown(f"### 🔍 Keyword: **{keyword.capitalize()}**")

        trends_df = get_google_trends([keyword], timeframe=timeframe, region=region)

        if trends_df.empty:
            st.warning(f"No Google Trends data found for '{keyword}'.")
            continue

        st.line_chart(trends_df[keyword], height=200)

        pred = predict_trend_lstm(trends_df[keyword], future_steps=future_days)
        if pred.size > 0:
            st.line_chart(pd.DataFrame({f"{keyword} - Predicted Trend": pred}))
            direction = trend_direction(pred)
            st.write(f"**Trend Outlook:** {direction}")
        else:
            st.info("Not enough data for prediction.")
            direction = "⚪ Unknown"

        trend_summary.append({
            "Keyword": keyword,
            "Latest Value": trends_df[keyword].iloc[-1],
            "Predicted Next": pred[-1] if pred.size > 0 else np.nan,
            "Trend": direction
        })

st.markdown("---")

# ------------------------ SUMMARY TABLE ------------------------
if trend_summary:
    st.subheader("📊 Summary of All Keywords")
    summary_df = pd.DataFrame(trend_summary)
    summary_df["Predicted Change (%)"] = ((summary_df["Predicted Next"] - summary_df["Latest Value"]) / summary_df["Latest Value"] * 100).round(2)
    summary_df = summary_df.sort_values(by="Predicted Change (%)", ascending=False)
    st.dataframe(summary_df.reset_index(drop=True), use_container_width=True)

    top_keyword = summary_df.iloc[0]["Keyword"]
    st.success(f"🔥 **Most Promising Trend:** `{top_keyword.upper()}` — showing strong upward potential!")
else:
    st.info("Enter keywords to begin your trend analysis.")
