import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="AI Trending Product Detector", layout="wide")
st.title("🤖 AI-Powered Trending Product Dashboard (No API Key Needed)")

# ------------------------ USER INPUT ------------------------
default_keywords = "smartwatch, wireless earbuds, sneakers, perfume, power bank"
keywords = st.text_input("Enter product keywords (comma-separated):", default_keywords)
keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
region = st.selectbox("Region", ["Worldwide", "IN", "US"], index=1)
timeframe = st.selectbox("Timeframe", ["today 3-m", "today 12-m"], index=0)

# ------------------------ GOOGLE TRENDS ------------------------
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

# ------------------------ LSTM TREND PREDICTION ------------------------
def predict_trend_lstm(series, future_steps=7, seq_len=14, epochs=5, batch_size=8):
    """Predict short-term trend using LSTM."""
    if len(series) < seq_len + 2:
        return np.array([])

    series = series[-(seq_len * 3):]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(25, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    predictions = []
    current_seq = X[-1]
    for _ in range(future_steps):
        pred = model.predict(current_seq.reshape(1, seq_len, 1), verbose=0)[0, 0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# ------------------------ DISPLAY RESULTS ------------------------
for keyword in keywords_list:
    st.subheader(f"📈 Keyword: {keyword}")

    trends_df = get_google_trends([keyword], timeframe=timeframe, region=region)

    if trends_df.empty:
        st.warning(f"No Google Trends data found for '{keyword}' in {region}.")
        continue

    st.line_chart(trends_df[keyword], height=200)

    pred = predict_trend_lstm(trends_df[keyword])
    if pred.size > 0:
        st.line_chart(pd.DataFrame({f"{keyword} - Predicted Trend": pred}))
    else:
        st.info("Not enough data for prediction.")

st.success("✅ Analysis Complete — AI Predictions Generated Successfully!")
