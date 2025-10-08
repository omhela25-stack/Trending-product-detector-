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

refresh_interval = st.sidebar.number_input("Auto-refresh interval (minutes)", min_value=1, value=10)
top_products_count = st.sidebar.number_input("Number of top products per keyword", min_value=1, max_value=10, value=3)

# ------------------------ CACHED AMAZON API HELPER ------------------------
@st.cache_data(ttl=3600)
def get_amazon_products(api_key: str, keyword: str, country_domain: str = "amazon.in", max_results: int = 5):
    api_url = "https://api.rainforestapi.com/request"
    params = {
        "api_key": api_key,
        "type": "search",
        "amazon_domain": country_domain,
        "search_term": keyword,
        "page": 1
    }
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        if "search_results" not in data:
            return pd.DataFrame()
        items = data["search_results"][:max_results]
        df = pd.DataFrame([{
            "Title": item.get("title"),
            "Price": item.get("price", {}).get("raw", "N/A"),
            "Rating": item.get("rating", "N/A"),
            "Reviews": item.get("ratings_total", "N/A"),
            "Link": item.get("link")
        } for item in items])
        return df
    except Exception as e:
        print(f"⚠️ Error fetching Amazon data for '{keyword}': {e}")
        return pd.DataFrame()

# ------------------------ LSTM HELPER ------------------------
def predict_trend_lstm(series, future_steps=7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1,1))
    seq_len = min(14, len(series)-1)
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1],1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=50, batch_size=1, verbose=0)

    last_seq = scaled_data[-seq_len:].reshape(1, seq_len, 1)
    predictions = []
    for _ in range(future_steps):
        pred = model.predict(last_seq, verbose=0)
        predictions.append(pred[0,0])
        last_seq = np.append(last_seq[:,1:,:], [[pred]], axis=1)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
    return predictions.flatten()

# ------------------------ MAIN LOOP WITH AUTO-REFRESH ------------------------
pytrends = TrendReq(hl='en-US', tz=330)

while True:
    if not api_key:
        st.error("Enter your Rainforest API key in the sidebar!")
        st.stop()

    st.subheader("📈 Historical Google Trends Data")
    try:
        pytrends.build_payload(kw_list=keywords_list, geo="" if region=="Worldwide" else region, timeframe=timeframe)
        trend_data = pytrends.interest_over_time().drop(columns=['isPartial'], errors='ignore')
        st.line_chart(trend_data)
    except Exception as e:
        st.error(f"Error fetching Google Trends data: {e}")
        st.stop()

    # ------------------------ LSTM FORECAST ------------------------
    st.subheader("🤖 Predicted Trend for Next 7 Days (AI Forecast)")
    predicted_growth = {}
    combined_plot_data = pd.DataFrame()

    for kw in trend_data.columns:
        series = trend_data[kw]
        future_pred = predict_trend_lstm(series, future_steps=7)
        predicted_growth[kw] = future_pred.mean() - series.tail(7).mean()
        combined_series = pd.Series(list(series[-14:])+list(future_pred),
                                    index=pd.date_range(end=pd.Timestamp.today()+pd.Timedelta(days=7), periods=21))
        combined_plot_data[kw] = combined_series

    # ------------------------ COMBINED CURRENT VS PREDICTED CHART ------------------------
    st.subheader("📊 Current vs Predicted Trend Growth Chart")
    fig, ax = plt.subplots(figsize=(10,5))
    for kw in combined_plot_data.columns:
        ax.plot(combined_plot_data.index, combined_plot_data[kw], label=kw)
    ax.set_title("Google Trends: Current vs Predicted")
    ax.set_ylabel("Interest (0-100)")
    ax.legend()
    st.pyplot(fig)

    # ------------------------ AI PREDICTED TOP KEYWORDS WITH COLOR-CODED ARROWS ------------------------
    trending_keywords = pd.Series(predicted_growth).sort_values(ascending=False).head(5)
    st.subheader("🔥 AI Predicted Top Trending Keywords")
    colored_keywords = []
    for kw, growth in trending_keywords.items():
        if growth > 0:
            colored_keywords.append(f"🟢 {kw} (+{growth:.2f})")
        else:
            colored_keywords.append(f"🔴 {kw} ({growth:.2f})")
    st.markdown("<br>".join(colored_keywords), unsafe_allow_html=True)

    # ------------------------ AMAZON TOP PRODUCTS PER KEYWORD ------------------------
    st.subheader("🛒 Top Rising Amazon Products")
    all_products = []
    for kw in trending_keywords.index:
        st.markdown(f"### 🏷️ {kw.title()}")
        df = get_amazon_products(api_key, kw, max_results=top_products_count)
        if not df.empty:
            # Show only top N products
            st.dataframe(df)
            all_products.append(df)
        else:
            st.warning(f"No products found for '{kw}'.")

    # ------------------------ CSV DOWNLOAD ------------------------
    if all_products:
        combined_df = pd.concat(all_products, ignore_index=True)
        csv_data = combined_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download AI Predicted Products CSV", data=csv_data,
                           file_name="ai_trending_products.csv", mime="text/csv")

    st.markdown("---")
    st.caption(f"Made with ❤️ using Streamlit, Google Trends, Rainforest API, and AI (LSTM forecasting). Auto-refresh every {refresh_interval} minutes.")

    # Wait for auto-refresh interval
    st.info(f
