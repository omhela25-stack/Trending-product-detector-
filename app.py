import streamlit as st
import pandas as pd
import numpy as np
import requests
from pytrends.request import TrendReq
from sklearn.linear_model import LinearRegression
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Amazon Trending Detector", layout="wide")
st.title("🤖 AI-Powered Amazon Trending Product Dashboard")

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

def predict_trend(series, future_steps=7):
    if len(series) < 2:
        return np.array([])
    y = series.values
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(y), len(y) + future_steps).reshape(-1, 1)
    pred = model.predict(future_X)
    return pred

# ---------------- RUN PREDICTION ----------------
if st.button("▶️ Predict Trends & Fetch Products"):
    for keyword in keywords_list:
        st.subheader(f"📈 Keyword: {keyword}")

        # Google Trends
        trends_df = get_google_trends(keyword, timeframe=timeframe, region=region)
        if not trends_df.empty:
            st.line_chart(trends_df[keyword], height=250)
            pred = predict_trend(trends_df[keyword])
            if len(pred) > 0:
                st.line_chart(pd.DataFrame({f"{keyword} - Predicted Trend": pred}), height=200)
        else:
            st.warning(f"No Google Trends data found for '{keyword}' in {region}.")

        # Amazon Products
        products = fetch_amazon_products(keyword, num_results=top_products_count)
        if products:
            st.write(f"🛒 Top {top_products_count} Amazon products for '{keyword}':")
            cols_disp = st.columns(3)
            for idx, p in enumerate(products):
                with cols_disp[idx % 3]:
                    if p["thumbnail"]:
                        st.image(p["thumbnail"], use_column_width=True)
                    st.markdown(f"**{p['title']}**")
                    st.markdown(f"💰 Price: {p['price']}")
                    st.markdown(f"[View on Amazon]({p['link']})", unsafe_allow_html=True)
        else:
            st.info("No products found.")
