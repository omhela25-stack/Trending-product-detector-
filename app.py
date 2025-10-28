import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
import time

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="AI Amazon Trending Detector", layout="wide")

# ------------------------ MARQUEE ------------------------
st.markdown(
    "<marquee style='color:orange; font-size:18px;'>Capstone Project Group 32 — In Progress Yet to be Completed</marquee>",
    unsafe_allow_html=True
)

st.title("🤖 AI-Powered Amazon Trending Product Dashboard")

# ------------------------ SIDEBAR ------------------------
categories = ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Shoes", "Camera"]
st.sidebar.header("Select Categories")
selected_categories = st.sidebar.multiselect("Choose product categories:", categories, default=["Smartphone"])

st.sidebar.header("Prediction Settings")
future_days = st.sidebar.slider("Days to predict trend for:", 1, 14, 7)
seq_len = st.sidebar.slider("LSTM sequence length:", 5, 30, 14)
predict_btn = st.sidebar.button("Predict Trends")

# ------------------------ HELPER FUNCTIONS ------------------------
def generate_sales_data(n=60):
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

def fetch_amazon_products(keyword, num_results=9):
    products = []
    for i in range(num_results):
        # Simulating realistic product names
        product_names = [
            "Apple iPhone 16", "Samsung Galaxy S24", "Xiaomi 11", 
            "OnePlus 12", "Realme X5", "Sony WH-1000XM5", 
            "Apple Watch Series 9", "Nike Air Max 2025", "Canon EOS R6"
        ]
        products.append({
            "title": product_names[i % len(product_names)],
            "price": np.random.randint(1000, 5000),
            "thumbnail": "https://via.placeholder.com/150",
            "link": "#"
        })
    return products

# ------------------------ MAIN DISPLAY ------------------------
if predict_btn:
    for keyword in selected_categories:
        st.subheader(f"📈 Category: {keyword}")

        # Simulate sales data
        sales_data = generate_sales_data()
        pred = lstm_predict(sales_data, future_steps=future_days, seq_len=seq_len)

        # Layout: graphs left, products right
        col1, col2 = st.columns([3, 1])

        with col1:
            # Historical sales graph
            fig_sales = go.Figure()
            fig_sales.add_trace(go.Scatter(
                x=list(range(1, len(sales_data)+1)),
                y=sales_data,
                mode='lines+markers',
                name='Historical Sales'
            ))
            fig_sales.update_layout(
                title=f"{keyword} - Historical Sales Trend",
                xaxis_title="Days",
                yaxis_title="Units Sold",
                template="plotly_dark",
                height=300
            )
            st.plotly_chart(fig_sales, use_container_width=True)

            # Predicted trend graph
            if pred.size > 0:
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=list(range(1, len(pred)+1)),
                    y=pred,
                    mode='lines+markers',
                    name='Predicted Sales'
                ))
                fig_pred.update_layout(
                    title=f"{keyword} - Predicted Trend for Next {future_days} Days",
                    xaxis_title="Future Days",
                    yaxis_title="Predicted Units",
                    template="plotly_dark",
                    height=300
                )
                st.plotly_chart(fig_pred, use_container_width=True)
            else:
                st.info("Not enough data to predict trend.")

        with col2:
            # Display products numbered
            products = fetch_amazon_products(keyword)
            st.write(f"🛒 Top Products for '{keyword}':")
            for idx, p in enumerate(products, start=1):
                st.markdown(f"**{idx}. {p['title']}**")
                st.image(p["thumbnail"], use_column_width=True)
                st.write(f"💰 Price: ₹{p['price']}")
                st.markdown(f"[View Product]({p['link']})")

# ------------------------ TEAM CREDITS ------------------------
st.markdown("---")
st.markdown("<h4 style='text-align:center; color:lightblue;'>Team: OM (AI Model) | Swati (Feature Engineering) | Jyoti (Frontend) | Srishti (Frontend)</h4>", unsafe_allow_html=True)
