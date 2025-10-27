# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from pytrends.request import TrendReq
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time, math

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Amazon Trending Detector", layout="wide", page_icon="🛍️")

# ---------------- API KEY (hardcoded for test only) ----------------
SERPAPI_KEY = "4427e6d1d612ec487682027e5fc7ac384c21317cecd0fe503d785c10c6c6595c"  # ⚠️ Private, for local use only

# ---------------- SIDEBAR / SETTINGS ----------------
st.sidebar.header("Settings & Data Sources")

with st.sidebar.expander("Keywords & Data"):
    default_keywords = "smartwatch, wireless earbuds, sneakers"
    keywords_input = st.text_input("Enter keywords (comma-separated):", default_keywords)
    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

with st.sidebar.expander("Google Trends"):
    region = st.selectbox("Region", ["Worldwide", "IN", "US"], index=1)
    timeframe = st.selectbox("Timeframe", ["today 3-m", "today 12-m"], index=0)
    future_days = st.slider("Predict future days (LSTM)", 3, 21, 7)

with st.sidebar.expander("Amazon Settings"):
    amazon_domain = st.selectbox("Amazon domain", ["amazon.in", "amazon.com", "amazon.co.uk"], index=0)
    results_per_keyword = st.number_input("Products per keyword", 1, 10, 5)

# ---------------- HELPERS ----------------
@st.cache_data(ttl=3600)
def get_google_trends(keyword_list, timeframe="today 3-m", geo="IN"):
    pytrends = TrendReq(hl='en-US', tz=330)
    for _ in range(3):
        try:
            pytrends.build_payload(keyword_list, timeframe=timeframe, geo=geo if geo != "Worldwide" else "")
            df = pytrends.interest_over_time()
            if not df.empty and "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            return df
        except Exception:
            time.sleep(1)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_amazon_serp(keyword, serpapi_key, amazon_domain="amazon.in", num_results=5):
    params = {
        "engine": "amazon",
        "q": keyword,
        "api_key": serpapi_key,
        "amazon_domain": amazon_domain,
        "num": num_results
    }
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        usage_info = data.get("search_metadata", {}).get("credits_used", None)
        results = data.get("organic_results", []) or data.get("search_results", [])
        rows = []
        for item in results[:num_results]:
            rows.append({
                "keyword": keyword,
                "title": item.get("title") or item.get("product_title") or "Unnamed Product",
                "price_raw": item.get("price") or item.get("price_text") or "N/A",
                "rating": item.get("rating", 0),
                "reviews": item.get("reviews", 0),
                "image": item.get("thumbnail") or item.get("image"),
                "link": item.get("link")
            })
        return pd.DataFrame(rows), usage_info
    except Exception:
        return pd.DataFrame(), None

def sample_amazon_products(keyword, n=5):
    imgs = [
        "https://m.media-amazon.com/images/I/61m0lZtZfYL._AC_UL320_.jpg",
        "https://m.media-amazon.com/images/I/71S8U9VzLTL._AC_UL320_.jpg",
        "https://m.media-amazon.com/images/I/61D4Y1qQnTL._AC_UL320_.jpg",
        "https://m.media-amazon.com/images/I/81e4D1Q6+eL._AC_UL320_.jpg"
    ]
    return pd.DataFrame([{
        "keyword": keyword,
        "title": f"{keyword.title()} Model {chr(65+i)}",
        "price_raw": f"₹{np.random.randint(799,9999)}",
        "rating": round(np.random.uniform(3.5, 5.0), 1),
        "reviews": np.random.randint(100, 10000),
        "image": imgs[i % len(imgs)],
        "link": "#"
    } for i in range(n)])

def build_lstm_prediction(series, future_steps=7, seq_len=14, epochs=6, batch_size=8):
    series = series.dropna()
    if len(series) < seq_len + 2:
