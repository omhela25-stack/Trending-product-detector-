import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from sklearn.linear_model import LinearRegression
import time

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="AI Trending Product Detector", layout="wide")
st.title("🤖 AI-Powered Trending Product Dashboard (No API Key Needed)")

# ------------------------ USER INPUT ------------------------
default_keywords = "smartwatch, wireless earbuds, sneakers, perfume, power bank"
keywords = st.text_input("Enter product keywords (comma-separated):", default_keywords)
keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
region = st.selectbox("Region", ["Worldwide", "IN", "US"], index=1)
timeframe = st.selectbox("Timeframe", ["today 3-m", "today 12-m"], index=0)

# ------------------------ GOOGLE TRENDS ------------------------
@st.cache_data(ttl=3600)
def get_google_trends(keywords, timeframe="today 3-m", region="IN"):
    pytrends = TrendReq()
    trends_data = pd.DataFrame()
    for attempt in range(3):
        try:
            pytrends.build_payload(keywords, timeframe=timeframe, geo=region)
            trends_data = pytrends.interest_over_time()
            if not trends_data.empty and "isPartial" in trends_data.columns:
                trends_data = trends_data.drop(columns=["isPartial"])
            break
        except Exception:
            time.sleep(2)
            continue
    return trends_data

# ------------------------ SIMPLE LINEAR FORECAST ------------------------
def predict_trend_linear(series, future_steps=7):
    """Predict short-term trend using linear regression."""
    if len(series) < 2:
        return []

    y = series.values
    X = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    X_future = np.arange(len(y), len(y) + future_steps).reshape(-1, 1)
    y_future = model.predict(X_future)
    return y_future

# ------------------------ DISPLAY RESULTS ------------------------
for keyword in keywords_list:
    st.subheader(f"📈 Keyword: {keyword}")

    trends_df = get_google_trends([keyword], timeframe=timeframe, region=region)

    if trends_df.empty:
        st.warning(f"No Google Trends data found for '{keyword}' in {region}.")
        continue

    st.line_chart(trends_df[keyword], height=200)

    pred = predict_trend_linear(trends_df[keyword])
    if len(pred) > 0:
        st.line_chart(pd.DataFrame({f"{keyword} - Predicted Trend": pred}))
    else:
        st.info("Not enough data for prediction.")

st.success("✅ Analysis Complete — AI Predictions Generated Successfully!")
