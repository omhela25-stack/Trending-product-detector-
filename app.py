import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time

# ------------------------ SLIDING BANNER ------------------------
st.markdown(
    """
    <div style="overflow:hidden; white-space:nowrap;">
        <marquee behavior="scroll" direction="left" scrollamount="6" style="font-size:20px; color:orange;">
            🚧 Capstone Project Group 32 -- In Progress, Yet to be Completed 🚧
        </marquee>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------ PAGE HEADER ------------------------
st.title("🤖 AI-Powered Amazon Trending Product Dashboard")

# ------------------------ USER INPUT ------------------------
categories = ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Shoes", "Camera"]
st.sidebar.header("Select Categories")
selected_categories = st.sidebar.multiselect("Choose product categories:", categories, default=["Smartphone"])

st.sidebar.header("Prediction Settings")
future_days = st.sidebar.slider("Days to predict trend for:", 1, 14, 7)
seq_len = st.sidebar.slider("LSTM sequence length:", 5, 30, 14)
predict_btn = st.sidebar.button("Predict Trends")

# ------------------------ TEAM MEMBERS ------------------------
st.markdown("---")
st.markdown(
    """
    <div style="display:flex; justify-content:space-around; font-weight:bold; font-size:16px;">
        <div>OM | AI Model</div>
        <div>SWATI | Feature Engineering</div>
        <div>JYOTI | Frontend</div>
        <div>Srishti | Frontend</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

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

def fetch_amazon_products(keyword, num_results=5):
    product_names = {
        "Smartphone": ["Apple iPhone 16", "Samsung Galaxy S24", "Xiaomi 13 Pro", "OnePlus 12", "Google Pixel 9"],
        "Laptop": ["MacBook Pro 16", "Dell XPS 15", "HP Spectre x360", "Lenovo ThinkPad X1", "Asus ZenBook 14"],
        "Headphones": ["Sony WH-1000XM5", "Bose QC45", "Apple AirPods Max", "Sennheiser HD 450", "JBL Live 650BT"],
        "Smartwatch": ["Apple Watch Series 9", "Samsung Galaxy Watch 6", "Fitbit Sense 2", "Garmin Venu 3", "Fossil Gen 7"],
        "Shoes": ["Nike Air Max 2025", "Adidas Ultraboost 22", "Puma RS-X", "Reebok Nano X3", "New Balance 990v6"],
        "Camera": ["Canon EOS R6", "Nikon Z8", "Sony A7 IV", "Fujifilm X-T5", "Panasonic Lumix S5"]
    }
    products = []
    names = product_names.get(keyword, [f"{keyword} Product {i+1}" for i in range(num_results)])
    for idx, name in enumerate(names[:num_results]):
        products.append({
            "title": f"{idx+1}. {name}",
            "price": np.random.randint(30000, 150000) if keyword in ["Smartphone", "Laptop"] else np.random.randint(2000, 50000),
            "thumbnail": "https://via.placeholder.com/150",
            "link": "#"
        })
    return products

# ------------------------ MAIN DISPLAY ------------------------
if predict_btn:
    for keyword in selected_categories:
        st.subheader(f"📈 Category: {keyword}")

        # Left: Charts, Right: Products
        chart_col, product_col = st.columns([3, 1])

        # Generate sales data
        sales_data = generate_sales_data()
        days = np.arange(1, len(sales_data)+1)

        # Actual Sales Trend
        with chart_col:
            df_actual = pd.DataFrame({"Day": days, "Actual Sales": sales_data}).set_index("Day")
            st.line_chart(df_actual, height=250, use_container_width=True)
            st.write("**Actual Sales Trend**")
            st.write("X-axis: Day, Y-axis: Sales")

        # Predicted Trend
        pred = lstm_predict(sales_data, future_steps=future_days, seq_len=seq_len)
        with chart_col:
            if pred.size > 0:
                df_pred = pd.DataFrame({"Day": np.arange(len(sales_data)+1, len(sales_data)+1+len(pred)), 
                                        "Predicted Sales": pred}).set_index("Day")
                st.line_chart(df_pred, height=250, use_container_width=True)
                st.write("**Predicted Sales Trend**")
                st.write("X-axis: Day, Y-axis: Predicted Sales")
            else:
                st.info("Not enough data to predict trend.")

        # Display products below charts on left side
        with chart_col:
            st.write(f"🛒 Top Products for {keyword}:")
            products = fetch_amazon_products(keyword, num_results=5)
            for p in products:
                st.write(f"{p['title']} - 💰 Price: ₹{p['price']}")
