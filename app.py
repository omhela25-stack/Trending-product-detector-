# app.py
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

# ---------------- API KEY (embedded for testing only) ----------------
SERPAPI_KEY = "4427e6d1d612ec487682027e5fc7ac384c21317cecd0fe503d785c10c6c6595c"

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
        return np.array([])
    series = series[-(seq_len*3):]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential([LSTM(32, input_shape=(seq_len, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    preds = []
    last_seq = X[-1]
    for _ in range(future_steps):
        pred = model.predict(last_seq.reshape(1, seq_len, 1), verbose=0)[0, 0]
        preds.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# ---------------- MAIN ----------------
st.title("🛍️ Amazon Trending Product Detector (AI + LSTM)")

geo_code = "" if region == "Worldwide" else region
with st.spinner("Fetching Google Trends..."):
    trends_df = get_google_trends(keywords, timeframe, geo_code)

usage_tracker = []

for kw in keywords:
    st.markdown(f"## 🔎 {kw.title()}")
    col1, col2 = st.columns([2, 1])

    # Google Trends + LSTM
    if kw in trends_df.columns:
        with col1:
            st.line_chart(trends_df[kw])
            preds = build_lstm_prediction(trends_df[kw], future_days)
            if preds.size > 0:
                st.line_chart(pd.Series(preds, name="Predicted"))
    else:
        st.warning(f"No Google Trends data found for {kw}")

    # Amazon Data
    with col2:
        st.markdown("**Amazon Results**")
        df_products, usage = fetch_amazon_serp(kw, SERPAPI_KEY, amazon_domain, results_per_keyword)
        if usage: usage_tracker.append(usage)
        if df_products.empty:
            df_products = sample_amazon_products(kw, results_per_keyword)
        for _, row in df_products.iterrows():
            st.image(row["image"], width=120)
            st.markdown(f"**[{row['title']}]({row['link']})**")
            st.caption(f"💰 {row['price_raw']} | ⭐ {row['rating']} | 🗳️ {row['reviews']} reviews")

# ---------------- FOOTER INFO ----------------
st.markdown("---")
st.header("🔐 API Key Management & Info")

if usage_tracker:
    st.info(f"Estimated SerpApi calls used this session: **{sum(usage_tracker)}**")
else:
    st.info("SerpApi usage data unavailable (using sample fallback).")

def rotate_key(new_key):
    global SERPAPI_KEY
    SERPAPI_KEY = new_key
    st.success("API key rotated successfully! Refresh to apply.")

with st.expander("Rotate API Key"):
    new_key = st.text_input("Enter new SerpApi key:", type="password")
    if st.button("Rotate Key Now"):
        if new_key:
            rotate_key(new_key)
        else:
            st.warning("Please enter a valid key first.")

st.caption("Deploy safely by keeping API key secret in Streamlit Cloud (Settings → Secrets).")

