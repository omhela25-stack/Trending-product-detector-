import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="AI Amazon Trending Detector", layout="wide")

# ------------------------ SLIDING BANNER ------------------------
st.markdown("""
<div style="overflow:hidden; white-space:nowrap; font-size:18px; color:blue; margin-bottom:20px;">
<marquee behavior="scroll" direction="left" scrollamount="6">Capstone Project Group 32 -- In Progress Yet To Be Completed</marquee>
</div>
""", unsafe_allow_html=True)

# ------------------------ HEADER ------------------------
st.title("🤖 AI-Powered Amazon Trending Product Dashboard")

# ------------------------ TEAM CREDITS ------------------------
col1, col2, col3, col4 = st.columns(4)
col1.markdown("**Om** - AI Model")
col2.markdown("**Swati** - Feature Engineering")
col3.markdown("**Jyoti** - Frontend")
col4.markdown("**Srishti** - Frontend")

st.markdown("---")

# ------------------------ SIDEBAR USER INPUT ------------------------
categories = ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Shoes", "Camera"]
st.sidebar.header("Select Categories")
selected_categories = st.sidebar.multiselect("Choose product categories:", categories, default=["Smartphone"])

st.sidebar.header("Prediction Settings")
future_days = st.sidebar.slider("Days to predict trend for:", 1, 14, 7)
seq_len = st.sidebar.slider("LSTM sequence length:", 5, 30, 14)
top_n_products = st.sidebar.slider("Number of products to display:", 3, 10, 5)
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

    data = np.array(series).reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled)-seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len])
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
        pred = model.predict(current_seq.reshape(1,seq_len,1), verbose=0)[0,0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    smoothed = pd.Series(predictions).rolling(2, min_periods=1).mean().values
    return scaler.inverse_transform(np.array(smoothed).reshape(-1,1)).flatten()

def fetch_amazon_products(keyword, num_results=5):
    product_names = {
        "Smartphone": ["Apple iPhone 16","Samsung S24","Xiaomi 13","OnePlus 12","Realme GT","Google Pixel 8","Vivo X90","Oppo Find X6","Motorola Edge","Nokia X100"],
        "Laptop": ["Dell XPS 15","MacBook Pro 16","HP Spectre","Lenovo ThinkPad","Asus ZenBook","Acer Swift","MSI Stealth","Razer Blade","Huawei MateBook","Samsung Galaxy Book"],
        "Headphones": ["Sony WH-1000XM5","Bose QC45","Beats Studio Pro","Sennheiser HD 450","JBL Live Pro","Audio-Technica ATH-M50","Philips Fidelio","AKG N700","Skullcandy Crusher","Bang&Olufsen Beoplay"],
        "Smartwatch": ["Apple Watch 9","Samsung Galaxy Watch 6","Fitbit Versa 4","Garmin Venu 3","Amazfit GTR 4","Fossil Gen 7","Huawei Watch GT 4","Suunto 9 Peak","TicWatch Pro 5","Mobvoi TicWatch E4"],
        "Shoes": ["Nike Air Max 2025","Adidas Ultraboost 24","Puma RS-X","Reebok Nano X3","New Balance 990v6","ASICS Gel-Kayano","Converse Chuck 70","Vans Old Skool","Jordan Air 1","Skechers GoRun"],
        "Camera": ["Canon EOS R7","Nikon Z9","Sony A7 IV","Fujifilm X-H2","Panasonic Lumix S5","Olympus OM-D E-M10","Leica Q3","Pentax K-3","Sigma fp L","Hasselblad X1D II"]
    }
    products = []
    for i in range(num_results):
        products.append({
            "title": product_names.get(keyword,[f"{keyword} Product {j+1}" for j in range(num_results)])[i],
            "price": np.random.randint(1000,5000),
            "thumbnail": "https://via.placeholder.com/150",
            "link": "#"
        })
    return products

# ------------------------ MAIN DISPLAY ------------------------
if predict_btn:
    for keyword in selected_categories:
        st.subheader(f"📈 Category: {keyword}")

        sales_data = generate_sales_data()
        pred = lstm_predict(sales_data, future_steps=future_days, seq_len=seq_len)

        trend_col, product_col = st.columns([2,1])

        # ---------------- TREND GRAPH ----------------
        with trend_col:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=sales_data, mode='lines+markers', name='Actual Sales'))
            if pred.size > 0:
                fig.add_trace(go.Scatter(y=pred, mode='lines+markers', name='Predicted Trend'))
            fig.update_layout(
                title=f"{keyword} Sales Trend",
                xaxis_title="Days",
                yaxis_title="Sales Units",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---------------- PRODUCT DISPLAY ----------------
        with product_col:
            st.write(f"🛒 Top {top_n_products} Products for '{keyword}':")
            products = fetch_amazon_products(keyword, num_results=top_n_products)
            for idx, p in enumerate(products, start=1):
                # Tooltip-style hover info
                st.markdown(f"**{idx}. {p['title']}**")
                st.image(p["thumbnail"], use_column_width=True)
                st.markdown(f"Price: ₹{p['price']}")
                if pred.size > 0:
                    predicted_value = round(pred[min(idx-1, len(pred)-1)],2)
                    st.markdown(f"Predicted Trend Units: {predicted_value}")
                st.markdown(f"[View Product]({p['link']})")
