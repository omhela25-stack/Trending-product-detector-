# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="AI Amazon Trending Detector", layout="wide")
st.title("🤖 AI-Powered Amazon Trending Product Dashboard (Lightweight Version)")

# ------------------------ USER INPUT ------------------------
default_keywords = "smartwatch, wireless earbuds, sneakers, perfume, power bank"
keywords = st.text_input("Enter product keywords (comma-separated):", default_keywords)
keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
region = st.selectbox("Region", ["Worldwide", "IN", "US"], index=1)
timeframe = st.selectbox("Timeframe", ["today 3-m", "today 12-m"], index=0)
st.sidebar.title("Rainforest API Key")
api_key = st.sidebar.text_input("API Key", type="password")

refresh_interval = st.sidebar.number_input("Auto-refresh interval (minutes)", min_value=1, value=10)
top_products_count = st.sidebar.number_input("Number of top products per keyword", min_value=1, max_value=10, value=5)

# ------------------------ TEST API KEY ------------------------
def test_api_key(api_key):
    """Quickly verify if API key works."""
    try:
        test_url = "https://api.rainforestapi.com/request"
        params = {
            "api_key": api_key,
            "type": "search",
            "amazon_domain": "amazon.in",
            "search_term": "test"
        }
        r = requests.get(test_url, params=params, timeout=10)
        data = r.json()
        if "request_info" in data and data["request_info"].get("success"):
            return True
        return False
    except Exception:
        return False

if not api_key:
    st.warning("⚠️ Please enter your Rainforest API key in the sidebar to start.")
    st.stop()

with st.spinner("🔍 Verifying your API key..."):
    if test_api_key(api_key):
        st.success("✅ API key verified successfully!")
    else:
        st.error("❌ Invalid or expired API key. Please check it and try again.")
        st.stop()

# ------------------------ CACHED AMAZON API HELPER ------------------------
@st.cache_data(ttl=3600)
def get_amazon_products(api_key: str, keyword: str, country_domain: str = "amazon.in", max_results: int = 5):
    api_url = "https://api.rainforestapi.com/request"
    params = {
        "api_key": api_key,
        "type": "search",
        "amazon_domain": country_domain,
        "search_term": keyword,
        "page": 1
    }
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        if "search_results" not in data:
            return pd.DataFrame()
        items = data["search_results"][:max_results]
        df = pd.DataFrame([{
            "Title": item.get("title"),
            "Price": float(item.get("price", {}).get("raw", "0").replace("₹","").replace(",","")) if item.get("price") else np.nan,
            "Rating": float(item.get("rating", 0)),
            "Reviews": int(item.get("ratings_total", 0)),
            "Link": item.get("link")
        } for item in items])
        return df
    except Exception as e:
        print(f"⚠️ Error fetching Amazon data for '{keyword}': {e}")
        return pd.DataFrame()

# ------------------------ FAST LSTM HELPER ------------------------
def predict_trend_lstm_fast(series, future_steps=7, seq_len=14, epochs=10):
    series = series[-(seq_len*3):]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1,1))

    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(25, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X,
