import streamlit as st
import pandas as pd
import numpy as np
import requests
from pytrends.request import TrendReq
from sklearn.linear_model import LinearRegression
import time

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="AI Amazon Trending Detector", layout="wide")
st.title("🤖 AI-Powered Amazon Trending Product Dashboard")

# ------------------------ CATEGORY SELECTION ------------------------
categories = ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Shoes", "Camera"]
st.subheader("Select Product Category:")

cols = st.columns(3)
category_selected = None
for idx, cat in enumerate(categories):
    if cols[idx % 3].button(cat):
        category_selected = cat

# ------------------------ REGION & SETTINGS ------------------------
region = st.selectbox("Region", ["Worldwide", "IN", "US"], index=1)
timeframe = st.selectbox("Timeframe", ["today 3-m", "today 12-m"], index=0)
top_products_count = st.slider("Number of top products to show:", min_value=1, max_value=10, value=5)

# ------------------------ SERPAPI CONFIG ------------------------
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY") or ""

if not SERPAPI_KEY:
    st.warning("⚠️ Add your SerpApi API key in Streamlit secrets.")
    st.stop()

# ------------------------ HELPER FUNCTIONS ------------------------
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
    y = series.values
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(y), len(y)+future_steps).reshape(-1, 1)
    pred = model.predict(future_X)
    return pred

# ------------------------ RUN PREDICTION ------------------------
if category_selected:
    if st.button("▶️ Predict Trends & Fetch Products"):
        st.subheader(f"📈 Category: {category_selected}")

        # Google Trends
        trends_df = get_google_trends(category_selected, timeframe=timeframe, region=region)
        if not trends_df.empty:
            st.line_chart(trends_df[category_selected], height=250, use_container_width=True)
            pred = predict_trend(trends_df[category_selected])
            st.line_chart(pd.DataFrame({f"{category_selected} - Predicted Trend": pred}), height=200, use_container_width=True)
        else:
            st.warning("No Google Trends data available.")

        # Amazon Products
        products = fetch_amazon_products(category_selected, num_results=top_products_count)
        if products:
            st.write(f"🛒 Top {top_products_count} Amazon products for '{category_selected}':")
            cols_disp = st.columns(3)
            for idx, p in enumerate(products):
                with cols_disp[idx % 3]:
                    st.image(p["thumbnail"], use_column_width=True)
                    st.markdown(f"**{p['title']}**")
                    st.markdown(f"💰 Price: {p['price']}")
                    st.markdown(f"[View on Amazon]({p['link']})", unsafe_allow_html=True)
        else:
            st.info("No products found.")
