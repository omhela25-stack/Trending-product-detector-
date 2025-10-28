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

# ------------------------ USER INPUT ------------------------
default_keywords = "smartwatch, wireless earbuds, sneakers, perfume, power bank"
keywords = st.text_input("Enter product keywords (comma-separated):", default_keywords)
keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
region = st.selectbox("Region", ["Worldwide", "IN", "US"], index=1)
timeframe = st.selectbox("Timeframe", ["today 3-m", "today 12-m"], index=0)

st.sidebar.title("Rainforest API Key")
api_key = st.sidebar.text_input("API Key", type="password")
top_products_count = st.sidebar.number_input(
    "Number of top products per keyword", min_value=1, max_value=10, value=5
)

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
            timeout=10,
        )
        data = r.json()
        return "request_info" in data and data["request_info"].get("success", False)
    except Exception:
        return False

if not api_key:
    st.warning("⚠️ Enter your Rainforest API key.")
    st.stop()
elif not test_api_key(api_key):
    st.error("❌ Invalid API key. Please check and try again.")
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
                "page": 1,
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()

        if "search_results" not in data:
            return pd.DataFrame()

        items = data["search_results"][:max_results]
        rows = []
        for item in items:
            price_raw = item.get("price", {}).get("raw")
            if price_raw:
                try:
                    price = float(price_raw.replace("₹","").replace(",","").split()[0])
                except Exception:
                    price = np.nan
            else:
                price = np.nan

            rows.append({
                "Title": item.get("title"),
                "Price (INR)": price,
                "Rating": float(item.get("rating",0)),
                "Reviews": int(item.get("ratings_total",0)),
                "Link": item.get("link"),
            })

        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ------------------------ LINEAR TREND PREDICTION ------------------------
def predict_trend_linear(series, future_steps=7):
    y = series.values
    if len(y) < 2:
        return np.array([y[-1] if len(y) else 0]*future_steps)
    X = np.arange(len(y)).reshape(-1,1)
    model = LinearRegression().fit(X, y)
    X_future = np.arange(len(y), len(y)+future_steps).reshape(-1,1)
    return model.predict(X_future)

# ------------------------ GOOGLE TRENDS ------------------------
@st.cache_data(ttl=3600)
def get_google_trends(keywords, timeframe="today 3-m", region="IN"):
    pytrends = TrendReq()
    trends_data = pd.DataFrame()
    for _ in range(3):
        try:
            pytrends.build_payload(keywords, timeframe=timeframe, geo=region if region!="Worldwide" else "")
            trends_data = pytrends.interest_over_time()
            if not trends_data.empty and "isPartial" in trends_data.columns:
                trends_data = trends_data.drop(columns=["isPartial"])
            break
        except Exception:
            time.sleep(1)
    return trends_data

# ------------------------ DISPLAY RESULTS ------------------------
if st.button("Run Analysis"):
    if not keywords_list:
        st.warning("Enter at least one keyword.")
    for keyword in keywords_list:
        st.subheader(f"📈 Keyword: {keyword}")

        trends_df = get_google_trends([keyword], timeframe=timeframe, region=region)
        if trends_df.empty:
            st.warning(f"No Google Trends data for '{keyword}' in {region}.")
        else:
            st.line_chart(trends_df[keyword])

            pred = predict_trend_linear(trends_df[keyword])
            st.line_chart(pd.DataFrame({f"{keyword} - predicted": pred}))

        amazon_df = get_amazon_products(api_key, keyword, max_results=top_products_count)
        if not amazon_df.empty:
            st.write(f"🛒 Top {top_products_count} Amazon products for '{keyword}':")
            st.dataframe(amazon_df)
        else:
            st.info(f"No products found for '{keyword}'.")
