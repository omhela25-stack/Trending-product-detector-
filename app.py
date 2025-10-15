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
st.title("🤖 AI-Powered Amazon Trending Product Dashboard")

# ------------------------ USER INPUT ------------------------
default_keywords = "smartwatch, wireless earbuds, sneakers, perfume, power bank"
keywords = st.text_input("Enter product keywords (comma-separated):", default_keywords)
keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
region = st.selectbox("Region", ["Worldwide", "IN", "US"], index=1)
timeframe = st.selectbox("Timeframe", ["today 3-m", "today 12-m"], index=0)
st.sidebar.title("Rainforest API Key")
api_key = st.sidebar.text_input("API Key", type="password")
top_products_count = st.sidebar.number_input("Number of top products per keyword", min_value=1, max_value=10, value=5)

# ------------------------ TEST API KEY ------------------------
def test_api_key(api_key):
    try:
        r = requests.get(
            "https://api.rainforestapi.com/request",
            params={"api_key": api_key, "type": "search", "amazon_domain": "amazon.in", "search_term": "test"},
            timeout=10
        )
        data = r.json()
        return "request_info" in data and data["request_info"].get("success")
    except:
        return False

if not api_key:
    st.warning("⚠️ Enter your Rainforest API key.")
    st.stop()
elif not test_api_key(api_key):
    st.error("❌ Invalid API key.")
    st.stop()
else:
    st.success("✅ API key verified!")

# ------------------------ AMAZON API HELPER ------------------------
@st.cache_data(ttl=3600)
def get_amazon_products(api_key, keyword, country_domain="amazon.in", max_results=5):
    try:
        r = requests.get(
            "https://api.rainforestapi.com/request",
            params={"api_key": api_key, "type": "search", "amazon_domain": country_domain, "search_term": keyword, "page": 1},
        )
        r.raise_for_status()
        data = r.json()
        if "search_results" not in data: return pd.DataFrame()
        items = data["search_results"][:max_results]
        df = pd.DataFrame([{
            "Title": item.get("title"),
            "Price": float(item.get("price", {}).get("raw","0").replace("₹","").replace(",","")) if item.get("price") else np.nan,
            "Rating": float(item.get("rating",0)),
            "Reviews": int(item.get("ratings_total",0)),
            "Link": item.get("link")
        } for item in items])
        return df
    except:
        return pd.DataFrame()

# ------------------------ LSTM TREND PREDICTION ------------------------
def predict_trend_lstm(series, future_steps=7, seq_len=14, epochs=10, batch_size=8):
    series = series[-(seq_len*3):]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series.values.reshape(-1,1))
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i,0])
        y.append(scaled_data[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential()
    model.add(LSTM(25, input_shape=(X.shape[1],1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    predictions = []
    current_seq = X[-1]
    for _ in range(future_steps):
        pred = model.predict(current_seq.reshape(1, seq_len, 1), verbose=0)[0,0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)
    return scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()

# ------------------------ GOOGLE TRENDS ------------------------
@st.cache_data(ttl=3600)
def get_google_trends(keywords, timeframe="today 3-m", region="IN"):
    pytrends = TrendReq()
    trends_data = pd.DataFrame()
    try:
        pytrends.build_payload(keywords, timeframe=timeframe, geo=region)
        trends_data = pytrends.interest_over_time()
        if not trends_data.empty and "isPartial" in trends_data.columns:
            trends_data = trends_data.drop(columns=["isPartial"])
    except:
        pass
    return trends_data

# ------------------------ DISPLAY RESULTS ------------------------
for keyword in keywords_list:
    st.subheader(f"📈 Keyword: {keyword}")
    trends_df = get_google_trends([keyword], timeframe=timeframe, region=region)
    if not trends_df.empty:
        st.line_chart(trends_df[keyword])
        pred = predict_trend_lstm(trends_df[keyword])
        st.line_chart(pd.DataFrame({f"{keyword} - predicted": pred}))
    amazon_df = get_amazon_products(api_key, keyword, max_results=top_products_count)
    if not amazon_df.empty:
        st.write(f"🛒 Top {top_products_count} Amazon products for '{keyword}':")
        st.dataframe(amazon_df)
    else:
        st.info(f"No products found for '{keyword}'.")
