import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random

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

# ------------------------ TEAM ------------------------
def show_team():
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("<center>**OM**<br>AI Model</center>", unsafe_allow_html=True)
    col2.markdown("<center>**Swati**<br>Feature Engineering</center>", unsafe_allow_html=True)
    col3.markdown("<center>**Jyoti**<br>Frontend</center>", unsafe_allow_html=True)
    col4.markdown("<center>**Srishti**<br>Frontend</center>", unsafe_allow_html=True)

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

# Realistic products per category
product_names = {
    "Smartphone": ["Apple iPhone 16", "Samsung Galaxy S24", "Xiaomi 13", "OnePlus 12", "Google Pixel 8",
                   "Realme GT 3", "Vivo X90", "Oppo Find X6", "Motorola Edge 40", "Sony Xperia 1 V"],
    "Laptop": ["MacBook Pro 16", "Dell XPS 15", "HP Spectre x360", "Lenovo ThinkPad X1", "Asus ROG Zephyrus",
               "Acer Swift 3", "MSI Stealth 15", "Apple MacBook Air M2", "Razer Blade 16", "LG Gram 17"],
    "Headphones": ["Sony WH-1000XM5", "Bose 700", "Apple AirPods Max", "JBL Live Pro", "Sennheiser Momentum",
                   "Beats Studio 4", "Anker Soundcore Life Q35", "AKG N700NC", "Bang & Olufsen Beoplay H95", "Audio-Technica ATH-M50X"],
    "Smartwatch": ["Apple Watch Series 9", "Samsung Galaxy Watch 6", "Fitbit Versa 4", "Garmin Fenix 7",
                   "Amazfit GTR 4", "Huawei Watch GT 4", "TicWatch Pro 5", "Suunto 9 Peak", "Withings ScanWatch", "Fossil Gen 6"],
    "Shoes": ["Nike Air Max 2025", "Adidas Ultraboost 22", "Puma RS-X", "Reebok Classic", "New Balance 550",
              "Skechers D'Lites", "Converse Chuck Taylor", "Vans Old Skool", "Asics Gel-Kayano", "Under Armour HOVR"],
    "Camera": ["Canon EOS R6", "Nikon Z7 II", "Sony A7 IV", "Fujifilm X-T5", "Panasonic Lumix S5",
               "Olympus OM-D E-M1 Mark III", "Leica Q2", "Canon EOS R5", "Nikon Z6 II", "Sony RX100 VII"]
}

def generate_products(category, num_products=8):
    products = []
    names_list = product_names.get(category, [f"{category} Product {i+1}" for i in range(num_products)])
    for i in range(num_products):
        name = names_list[i % len(names_list)]
        sales = generate_sales_data()
        pred_demand = lstm_predict(sales, future_steps=future_days, seq_len=seq_len)
        avg_pred = pred_demand.mean() if pred_demand.size > 0 else np.random.randint(50,200)
        products.append({
            "Name": name,
            "Price": np.random.randint(1000, 50000),
            "Predicted Demand": round(avg_pred,2),
            "Thumbnail": "https://via.placeholder.com/150",
            "Link": "#"
        })
    # Sort by predicted demand descending
    products = sorted(products, key=lambda x: x["Predicted Demand"], reverse=True)
    return products

# ------------------------ MAIN DISPLAY ------------------------
if predict_btn:
    for category in selected_categories:
        st.subheader(f"📈 Category: {category}")
        products = generate_products(category)

        # Display products in columns (4 per row)
        cols_per_row = 4
        for i in range(0, len(products), cols_per_row):
            row_products = products[i:i+cols_per_row]
            cols = st.columns(len(row_products))
            for col, p in zip(cols, row_products):
                with col:
                    st.image(p["Thumbnail"], use_column_width=True)
                    st.markdown(f"**{p['Name']}**")
                    st.markdown(f"💰 Price: ₹{p['Price']}")
                    st.markdown(f"📊 Predicted Demand: {p['Predicted Demand']}")
                    st.markdown(f"[View Product]({p['Link']})")

# ------------------------ TEAM SECTION ------------------------
st.markdown("---")
st.subheader("🛠️ Team Behind This Project")
show_team()
