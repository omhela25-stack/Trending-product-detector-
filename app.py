import streamlit as st
import pandas as pd
import numpy as np
import requests
from pytrends.request import TrendReq
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Amazon Trending Detector", layout="wide")
st.title("🤖 AI-Powered Amazon Trending Product Dashboard (LSTM)")

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Settings")
keywords = st.sidebar.text_input(
    "Enter product keywords (comma-separated)", 
    "smartphone, laptop, headphones, smartwatch, shoes, camera"
)
keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]

region = st.sidebar.selectbox("Region", ["Worldwide", "IN", "US"], index=1)
timeframe = st.sidebar.selectbox("Timeframe", ["today 3-m", "today 12-m"], index=0)
top_products_count = st.sidebar.slider("Top products to display:", 1, 10, 5)
SERPAPI_KEY = st.sidebar.text_input("SerpApi API Key", type="password")

if not SERPAPI_KEY:
    st.warning("⚠️ Enter your SerpApi key to fetch products")
    st.stop()

# ---------------- HELPERS ----------------
@st.cache_data(ttl=3600)
def fetch_amazon_products(keyword, num_results=5):
    url = "https://serpapi.com/search"
    params = {
        "engine": "amazon",
        "amazon_domain": "amazon.in",
        "q": keyword,
        "api_key": SERPAPI_KEY,
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        products = []
        for result in data.get("organic_results", [])[:num_results]:
            products.append({
                "title": result.get("title", "No title"),
                "price": result.get("price", "N/A"),
                "thumbnail": result.get("thumbnail", ""),
                "link": result.get("link", "#")
            })
        return products
    except Exception as e:
        st.error(f"Error fetching products: {e}")
        return []

@st.cache_data(ttl=3600)
def get_google_trends(keyword, timeframe="today 3-m", region="IN"):
    pytrends = TrendReq()
    trends_data = pd.DataFrame()
    for attempt in range(3):
        try:
            pytrends.build_payload([keyword], timeframe=timeframe, geo=region)
            trends_data = pytrends.interest_over_time()
            if not trends_data.empty and "isPartial" in trends_data.columns:
                trends_data = trends_data.drop(columns=["isPartial"])
            break
        except Exception:
            time.sleep(2)
            continue
    return trends_data

# ---------------- LSTM PREDICTION ----------------
def predict_trend_lstm(series, future_steps=7, seq_len=14, epochs=10, batch_size=4):
    """Predict trend using LSTM."""
    if len(series) < seq_len + 2:
        return np.array([])

    data = series.values[-(seq_len*3):].reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i,0])
        y.append(scaled[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1],1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y
