import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="AI Amazon Trending Detector", layout="wide")
st.title("🤖 AI-Powered Amazon Trending Product Dashboard")

# ------------------------ SIDEBAR CONFIG ------------------------
categories = ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Shoes", "Camera"]
st.sidebar.header("Settings")
selected_categories = st.sidebar.multiselect("Choose product categories:", categories, default=["Smartphone"])
future_days = st.sidebar.slider("Days to predict trend for:", 1, 14, 7)
seq_len = st.sidebar.slider("LSTM sequence length:", 5, 30, 14)
serp_api_key = st.sidebar.text_input("SerpApi API Key", type="password")
predict_btn = st.sidebar.button("Predict Trends")

# ------------------------ LSTM TREND PREDICTION ------------------------
def generate_sales_data(n=60):
    """Simulate dummy sales data."""
    np.random.seed(42)
    base = np.linspace(50, 200, n)
    noise = np.random.normal(0, 5, n)
    return base + noise

def lstm_predict(series, future_steps=7, seq_len=14, epochs=10, batch_size=4):
    if len(series) < seq_len + 2:
        return np.array([])

    data = np.array(series).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i + seq_len])
        y.append(scaled[i + seq_len])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    predictions = []
    current_seq = X[-1]
    for _ in range(future_steps):
        pred = model.predict(current_seq.reshape(1, seq_len, 1), verbose=0)[0, 0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# ------------------------ SERPAPI AMAZON FETCH ------------------------
@st.cache_data(ttl=3600)
def fetch_amazon_products_serp(keyword, api_key, num_results=5):
    url = "https://serpapi.com/search"
    params = {
        "engine": "amazon",
        "amazon_domain": "amazon.in",
        "q": keyword,
        "api_key": api_key
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        products = []
        for item in data.get("organic_results", [])[:num_results]:
            products.append({
                "title": item.get("title", "No Title"),
                "price": item.get("price", "N/A"),
                "thumbnail": item.get("thumbnail", "https://via.placeholder.com/150"),
                "link": item.get("link", "#")
            })
        return products
    except Exception:
        return []

# ------------------------ MAIN APP ------------------------
if predict_btn:
    if not serp_api_key:
        st.error("⚠️ Enter your SerpApi API key in the sidebar to fetch real products.")
    else:
        for category in selected_categories:
            st.subheader(f"📈 Category: {category}")

            # Simulate sales/trend data
            sales_data = generate_sales_data()
            st.line_chart(sales_data, height=200, use_container_width=True)

            # Predict trend
            pred = lstm_predict(sales_data, future_steps=future_days, seq_len=seq_len)
            if pred.size > 0:
                st.line_chart(pred, height=200, use_container_width=True)
            else:
                st.info("Not enough data for prediction.")

            # Fetch real Amazon products
            products = fetch_amazon_products_serp(category, serp_api_key, num_results=5)
            if products:
                cols = st.columns(len(products))
                for col, p in zip(cols, products):
                    with col:
                        st.image(p["thumbnail"], use_column_width=True)
                        st.write(p["title"])
                        st.write(f"💰 Price: {p['price']}")
                        st.markdown(f"[View Product]({p['link']})")
            else:
                st.info(f"No products found for '{category}'.")
