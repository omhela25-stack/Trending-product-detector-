# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="AI Amazon Trending Detector", layout="wide")
st.title("🤖 AI-Powered Amazon Trending Product Dashboard")

# ------------------------ USER INPUT ------------------------
default_keywords = "smartwatch, wireless earbuds, sneakers, perfume, power bank"
keywords = st.text_input("Enter product keywords (comma-separated):", default_keywords)
keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
region = st.selectbox("Region", ["Worldwide", "IN", "US"], index=1)
timeframe = st.selectbox("Timeframe", ["today 3-m", "today 12-m"], index=0)

st.sidebar.title("🔑 Rainforest API Key")
api_key = st.sidebar.text_input("API Key", type="password")
top_products_count = st.sidebar.number_input("Number of top products per keyword", min_value=1, max_value=10, value=5)

# ------------------------ TEST API KEY ------------------------
def test_api_key(api_key):
    try:
        r = requests.get(
            "https://api.rainforestapi.com/request",
            params={
                "api_key": api_key,
                "type": "search",
                "amazon_domain": "amazon.in",
                "search_term": "test",
            },
            timeout=10
        )
        data = r.json()
        return "request_info" in data and data["request_info"].get("success")
    except:
        return False

if not api_key:
    st.warning("⚠️ Please enter your Rainforest API key in the sidebar.")
    st.stop()
elif not test_api_key(api_key):
    st.error("❌ Invalid or inactive API key. Please check your Rainforest account.")
    st.stop()
else:
    st.success("✅ API key verified successfully!")

# ------------------------ AMAZON API HELPER ------------------------
@st.cache_data(ttl=3600)
def get_amazon_products(api_key, keyword, country_domain="amazon.in", max_results=5):
    try:
        r = requests.get(
            "https://api.rainforestapi.com/request",
            params={
                "api_key": api_key,
                "type": "search",
                "amazon_domain": country_domain,
                "search_term": keyword,
                "page": 1
            },
            timeout=20
        )
        r.raise_for_status()
        data = r.json()
        if "search_results" not in data:
            return pd.DataFrame()

        items = data["search_results"][:max_results]
        clean_items = []
        for item in items:
            price_raw = item.get("price", {}).get("raw")
            if price_raw:
                try:
                    price = float(price_raw.replace("₹", "").replace(",", "").split()[0])
                except:
                    price = np.nan
            else:
                price = np.nan

            clean_items.append({
                "Title": item.get("title"),
                "Price (₹)": price,
                "Rating": float(item.get("rating", 0)),
                "Reviews": int(item.get("ratings_total", 0)),
                "Link": item.get("link")
            })

        return pd.DataFrame(clean_items)
    except Exception as e:
        print("Amazon API error:", e)
        return pd.DataFrame()

# ------------------------ GOOGLE TRENDS ------------------------
@st.cache_data(ttl=3600)
def get_google_trends(keywords, timeframe="today 3-m", region="IN"):
    pytrends = TrendReq()
    trends_data = pd.DataFrame()
    for attempt in range(3):
        try:
            pytrends.build
