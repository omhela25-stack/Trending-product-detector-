# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Amazon Trending Products", layout="wide", page_icon="🛍️")
st.title("🛍️ Amazon Trending Products Dashboard")

# ---------------- SETTINGS ----------------
SERPAPI_KEY = "4427e6d1d612ec487682027e5fc7ac384c21317cecd0fe503d785c10c6c6595c"

default_keywords = ["smartwatch", "wireless earbuds", "sneakers", "perfume", "power bank"]
selected_keyword = st.sidebar.selectbox("Select a Keyword", default_keywords)
amazon_domain = st.sidebar.selectbox("Amazon Domain", ["amazon.in", "amazon.com", "amazon.co.uk"], index=0)
results_per_keyword = st.sidebar.slider("Number of Products", 1, 10, 5)
future_days = st.sidebar.slider("Predict future days", 3, 14, 7)

# ---------------- HELPERS ----------------
@st.cache_data(ttl=3600)
def fetch_amazon_products(keyword, serpapi_key, amazon_domain="amazon.in", num_results=5):
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
        results = data.get("organic_results", []) or data.get("search_results", [])
        rows = []
        for item in results[:num_results]:
            image = item.get("thumbnail") or item.get("image") or "https://via.placeholder.com/150"
            rows.append({
                "title": item.get("title") or item.get("product_title") or "Unnamed Product",
                "price_raw": item.get("price") or item.get("price_text") or "N/A",
                "rating": item.get("rating", round(np.random.uniform(3.5,5.0),1)),
                "reviews": item.get("reviews", np.random.randint(100,10000)),
                "image": image,
                "link": item.get("link") or "#"
            })
        return pd.DataFrame(rows)
    except Exception:
        # fallback demo products
        imgs = [
            "https://m.media-amazon.com/images/I/61m0lZtZfYL._AC_UL320_.jpg",
            "https://m.media-amazon.com/images/I/71S8U9VzLTL._AC_UL320_.jpg",
            "https://m.media-amazon.com/images/I/61D4Y1qQnTL._AC_UL320_.jpg",
            "https://m.media-amazon.com/images/I/81e4D1Q6+eL._AC_UL320_.jpg"
        ]
        return pd.DataFrame([{
            "title": f"{keyword.title()} Model {chr(65+i)}",
            "price_raw": f"₹{np.random.randint(799,9999)}",
            "rating": round(np.random.uniform(3.5, 5.0), 1),
            "reviews": np.random.randint(100,10000),
            "image": imgs[i % len(imgs)],
            "link": "#"
        } for i in range(num_results)])

def predict_trend_placeholder(n=future_days):
    """Dummy LSTM prediction placeholder"""
    return np.round(np.random.uniform(0.5,1.0,size=n),2)

# ---------------- DISPLAY ----------------
st.markdown(f"## 🔎 {selected_keyword.title()}")

products_df = fetch_amazon_products(selected_keyword, SERPAPI_KEY, amazon_domain, results_per_keyword)

for idx, row in products_df.iterrows():
    with st.container():
        col1, col2, col3 = st.columns([1,3,2])
        col1.image(row["image"], width=150)
        col2.markdown(f"**[{row['title']}]({row['link']})**")
        col2.caption(f"💰 {row['price_raw']} | ⭐ {row['rating']} | 🗳️ {row['reviews']} reviews")
        # Trend prediction placeholder
        pred = predict_trend_placeholder(future_days)
        col3.line_chart(pred)
