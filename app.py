import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="AI Amazon Trending Detector", layout="wide")
st.title("🤖 AI-Powered Amazon Trending Product Dashboard")

# ------------------------ USER INPUT ------------------------
categories = ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Shoes", "Camera"]
st.sidebar.header("Select Categories")
selected_categories = st.sidebar.multiselect("Choose product categories:", categories, default=["Smartphone"])

st.sidebar.header("Prediction Settings")
future_days = st.sidebar.slider("Days to predict trend for:", 1, 14, 7)
seq_len = st.sidebar.slider("LSTM sequence length:", 5, 30, 14)

predict_btn = st.sidebar.button("Predict Trends")

# ------------------------ HELPER FUNCTIONS ------------------------
def generate_sales_data(n=60):
    """Simulate dummy sales data."""
    np.random.seed(42)
    base = np.linspace(50, 200, n)
    noise = np.random.normal(0, 5, n)
    return base + noise

def lstm_predict(series, future_steps=7, seq_len=14, epochs=10, batch_size=4):
    """Predict trend using LSTM."""
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

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(seq_len, 1)))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    predictions = []
    current_seq = X[-1]
    for _ in range(future_steps):
        pred = model.predict(current_seq.reshape(1, seq_len, 1), verbose=0)[0, 0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def fetch_amazon_products(keyword, num_results=5):
    """Fetch dummy Amazon product data (replace with real API if available)."""
    products = []
    for i in range(num_results):
        products.append({
            "title": f"{keyword} Product {i+1}",
            "price": np.random.randint(1000, 5000),
            "thumbnail": "https://via.placeholder.com/150",
            "link": "#"
        })
    return products

# ------------------------ MAIN DISPLAY ------------------------
if predict_btn:
    for keyword in selected_categories:
        st.subheader(f"📈 Category: {keyword}")

        # Simulate trend data
        sales_data = generate_sales_data()
        st.line_chart(sales_data, height=200, use_container_width=True)

        # Predict trend
        pred = lstm_predict(sales_data, future_steps=future_days, seq_len=seq_len)
        if pred.size > 0:
            st.line_chart(pred, height=200, use_container_width=True)
        else:
            st.info("Not enough data to predict trend.")

        # Display products
        products = fetch_amazon_products(keyword, num_results=5)
        cols = st.columns(5)
        for col, p in zip(cols, products):
            with col:
                st.image(p["thumbnail"], use_column_width=True)
                st.write(p["title"])
                st.write(f"💰 Price: ₹{p['price']}")
                st.markdown(f"[View Product]({p['link']})")

# ------------------------ TEAM INFO ------------------------
# ------------------------ MAIN DISPLAY ------------------------
if predict_btn:
    for keyword in selected_categories:
        st.subheader(f"📈 Category: {keyword}")

        # Simulate trend data
        sales_data = generate_sales_data()
        st.line_chart(sales_data, height=200, use_container_width=True)

        # Predict trend
        pred = lstm_predict(sales_data, future_steps=future_days, seq_len=seq_len)
        if pred.size > 0:
            st.line_chart(pred, height=200, use_container_width=True)
        else:
            st.info("Not enough data to predict trend.")

        # ------------------------ TEAM INFO (IN-BETWEEN) ------------------------
        st.markdown("### 👥 Project Team")
        cols = st.columns(4)
        team_data = [
            {"Name": "Om", "Role": "AI Model Development"},
            {"Name": "Swati", "Role": "Feature Engineering"},
            {"Name": "Jyoti", "Role": "Frontend Development"},
            {"Name": "Srishti", "Role": "Frontend Development"}
        ]
        for col, member in zip(cols, team_data):
            with col:
                st.markdown(f"**{member['Name']}**")
                st.markdown(f"*{member['Role']}*")

       
