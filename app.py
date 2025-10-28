import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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
num_products = st.sidebar.slider("Number of top products to show:", 3, 10, 5)

predict_btn = st.sidebar.button("Predict Trends")

# ------------------------ HELPER FUNCTIONS ------------------------
def generate_sales_data(n=60):
    """Simulate dummy sales trend data."""
    np.random.seed(42)
    base = np.linspace(50, 200, n)
    noise = np.random.normal(0, 10, n)
    return base + noise

def lstm_predict(series, future_steps=7, seq_len=14, epochs=5, batch_size=4):
    """Predict product demand trend using LSTM."""
    if len(series) < seq_len + 2:
        return np.array([])

    data = np.array(series).reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i + seq_len])
        y.append(scaled[i + seq_len])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1],1)

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

def generate_products(category, num_products=5):
    """Generate dummy product data per category with random prices & predicted demand."""
    products = []
    for i in range(num_products):
        sales = generate_sales_data()
        pred_demand = lstm_predict(sales, future_steps=future_days, seq_len=seq_len)
        avg_pred = pred_demand.mean() if pred_demand.size>0 else np.random.randint(50,200)
        products.append({
            "Name": f"{category} Product {i+1}",
            "Price (₹)": np.random.randint(1000, 100000),
            "Predicted Demand": round(avg_pred,2)
        })
    # Sort by predicted demand descending
    products = sorted(products, key=lambda x: x["Predicted Demand"], reverse=True)
    return pd.DataFrame(products)

# ------------------------ MAIN DISPLAY ------------------------
if predict_btn:
    for category in selected_categories:
        st.subheader(f"📈 Top Trending Products - {category}")

        products_df = generate_products(category, num_products=num_products)
        st.dataframe(products_df, use_container_width=True)

    # ------------------------ TEAM INFO (SIDE-BY-SIDE) ------------------------
    st.markdown("---")
    st.markdown("### 👥 Project Team")
    team_cols = st.columns(4)
    team_data = [
        {"Name": "Om", "Role": "AI Model Development", "Color": "#FFB347", "Icon": "🤖"},
        {"Name": "Swati", "Role": "Feature Engineering", "Color": "#77DD77", "Icon": "🛠️"},
        {"Name": "Jyoti", "Role": "Frontend Development", "Color": "#89CFF0", "Icon": "💻"},
        {"Name": "Srishti", "Role": "Frontend Development", "Color": "#FF6961", "Icon": "🎨"}
    ]
    for col, member in zip(team_cols, team_data):
        with col:
            st.markdown(
                f"<div style='background-color:{member['Color']}; padding:15px; border-radius:10px; text-align:center'>"
                f"<h3>{member['Icon']} {member['Name']}</h3>"
                f"<p><i>{member['Role']}</i></p>"
                f"</div>",
                unsafe_allow_html=True
            )
