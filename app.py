import streamlit as st
import pandas as pd
import numpy as np
import requests
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import time

# -------------------- Streamlit UI Config --------------------
st.set_page_config(page_title="AI-Powered Amazon Trending Product Dashboard", layout="wide")

# Sidebar: API and Controls
st.sidebar.title("🔑 Rainforest API Key")
api_key = st.sidebar.text_input("API Key", type="password")

st.sidebar.number_input("Auto-refresh interval (minutes)", 1, 60, 10)
top_n = st.sidebar.number_input("Number of top products per keyword", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.info("Powered by Amazon Product Data + Google Trends")

# -------------------- Main Page --------------------
st.title("🤖 AI-Powered Amazon Trending Product Dashboard (Lightweight Version)")

keywords = st.text_input("Enter product keywords (comma-separated):", "smartwatch, wireless earbuds, sneakers, perfume, power bank")
region = st.selectbox("Region", ["IN", "US", "UK", "CA", "DE"])
timeframe = st.selectbox("Timeframe", ["today 3-m", "today 12-m", "now 7-d", "all"])

if not api_key:
    st.error("Enter your Rainforest API key in the sidebar!")
    st.stop()

# -------------------- Function: Amazon Trending Fetch --------------------
def get_amazon_trending_products(keyword, api_key, top_n=5):
    """Fetch trending products for a given keyword using Rainforest API"""
    url = "https://api.rainforestapi.com/request"
    params = {
        "api_key": api_key,
        "type": "search",
        "amazon_domain": f"amazon.{region.lower()}",
        "search_term": keyword,
        "sort_by": "featured",
    }

    try:
        r = requests.get(url, params=params)
        data = r.json()
        products = []

        for item in data.get("search_results", [])[:top_n]:
            products.append({
                "title": item.get("title"),
                "price": item.get("price", {}).get("value", "N/A"),
                "link": item.get("link"),
                "rating": item.get("rating", "N/A"),
                "reviews": item.get("reviews", {}).get("total_reviews", "N/A")
            })
        return pd.DataFrame(products)
    except Exception as e:
        st.warning(f"API Error: {e}")
        return pd.DataFrame()

# -------------------- Function: Google Trends Data --------------------
def get_trend_data(keyword, timeframe, region):
    pytrends = TrendReq()
    pytrends.build_payload([keyword], timeframe=timeframe, geo=region)
    data = pytrends.interest_over_time()
    if not data.empty:
        return data[keyword]
    return pd.Series(dtype=float)

# -------------------- Analysis --------------------
keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
trend_data = {}

for keyword in keywords_list:
    with st.spinner(f"Analyzing {keyword}..."):
        trend_series = get_trend_data(keyword, timeframe, region)
        if not trend_series.empty:
            trend_data[keyword] = trend_series

if trend_data:
    st.subheader("📈 Google Trends Data (Past Popularity)")
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, v in trend_data.items():
        ax.plot(v.index, v.values, label=k)
    ax.set_title("Interest Over Time")
    ax.legend()
    st.pyplot(fig)

# -------------------- Amazon Results --------------------
st.subheader("🛒 Trending Products on Amazon")
for keyword in keywords_list:
    st.markdown(f"### 🔍 {keyword.title()}")
    df = get_amazon_trending_products(keyword, api_key, top_n)
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.warning(f"No data found for **{keyword}**.")

st.success("✅ Trend analysis complete! Dashboard ready for exploration.")
