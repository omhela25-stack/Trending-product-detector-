import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="AI Amazon Trending Detector", layout="wide")
st.title("🤖 AI-Powered Amazon Trending Product Dashboard")

# Sidebar
categories = ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Shoes", "Camera"]
st.sidebar.header("Select Categories")
selected_categories = st.sidebar.multiselect("Choose product categories:", categories, default=["Smartphone"])
future_days = st.sidebar.slider("Days to predict trend for:", 1, 14, 7)
seq_len = st.sidebar.slider("LSTM sequence length:", 5, 30, 14)
serp_api_key = st.sidebar.text_input("SerpApi API Key", type="password")
predict_btn = st.sidebar.button("Predict Trends")

# LSTM
def lstm_predict(series, future_steps=7, seq_len=14, epochs=10, batch_size=4):
    data = np.array(series).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i + seq_len])
        y.append(scaled[i + seq_len])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(seq_len,1)))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    predictions = []
    current_seq = X[-1]
    for _ in range(future_steps):
        pred = model.predict(current_seq.reshape(1, seq_len, 1), verbose=0)[0,0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)
    return scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()

# Fetch Amazon via SerpApi
def fetch_amazon_products_serp(keyword, api_key, num_results=5):
    url = "https://serpapi.com/search"
    params = {"engine":"amazon","amazon_domain":"amazon.in","q":keyword,"api_key":api_key}
    try:
        r = requests.get(url, params=params, timeout=10).json()
        products = []
        for item in r.get("organic_results", [])[:num_results]:
            products.append({
                "title": item.get("title","No title"),
                "price": item.get("price","N/A"),
                "thumbnail": item.get("thumbnail","https://via.placeholder.com/150"),
                "link": item.get("link","#")
            })
        return products
    except:
        return []

# Main
if predict_btn:
    if not serp_api_key:
        st.error("Enter your SerpApi API key in the sidebar.")
    else:
        for cat in selected_categories:
            st.subheader(f"📈 Category: {cat}")

            # Dummy trend
            sales = np.linspace(50,200,60) + np.random.normal(0,5,60)
            st.line_chart(sales, height=200)

            pred = lstm_predict(sales, future_steps=future_days, seq_len=seq_len)
            st.line_chart(pred, height=200)

            # Real products
            products = fetch_amazon_products_serp(cat, serp_api_key, num_results=5)
            if products:
                cols = st.columns(len(products))
                for col, p in zip(cols, products):
                    with col:
                        st.image(p["thumbnail"], use_column_width=True)
                        st.write(p["title"])
                        st.write(f"💰 Price: {p['price']}")
                        st.markdown(f"[View Product]({p['link']})")
            els
