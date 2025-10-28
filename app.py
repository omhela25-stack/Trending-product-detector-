import streamlit as st
import requests
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# 🔑 SerpApi Key
# ----------------------------
SERPAPI_KEY = "4427e6d1d612ec487682027e5fc7ac384c21317cecd0fe503d785c10c6c6595c"

# ----------------------------
# 📦 Fetch Amazon Products via SerpApi
# ----------------------------
def fetch_amazon_products(keyword, num_results=6):
    url = "https://serpapi.com/search"
    params = {
        "engine": "amazon",
        "amazon_domain": "amazon.in",
        "q": keyword,
        "api_key": SERPAPI_KEY,
    }

    response = requests.get(url, params=params)
    data = response.json()

    products = []
    for result in data.get("organic_results", [])[:num_results]:
        products.append({
            "title": result.get("title", "No title"),
            "price": result.get("price", "N/A"),
            "thumbnail": result.get("thumbnail", ""),
            "link": result.get("link", "#")
        })
    return products

# ----------------------------
# 🧠 Dummy Sales Trend Simulation + LSTM Prediction
# ----------------------------
def generate_sales_data(n=60):
    np.random.seed(42)
    base = np.linspace(50, 200, n)
    noise = np.random.normal(0, 5, n)
    return base + noise

def lstm_predict(sales):
    data = np.array(sales).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    seq_len = 5
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=4, verbose=0)

    next_input = scaled[-seq_len:].reshape(1, seq_len, 1)
    pred = model.predict(next_input, verbose=0)
    pred_inv = scaler.inverse_transform(pred)
    return float(pred_inv[0][0])

# ----------------------------
# 🎨 Streamlit UI
# ----------------------------
st.set_page_config(page_title="Amazon Trending Predictor", layout="wide")

st.title("🛍️ Amazon Trending Product Predictor")

keyword = st.selectbox(
    "Select Keyword:",
    ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Shoes", "Camera"]
)

if st.button("Fetch & Predict"):
    with st.spinner("Fetching products and predicting trends..."):
        products = fetch_amazon_products(keyword)
        if not products:
            st.warning("No products found. Try again later or with another keyword.")
        else:
            for p in products:
                sales_data = generate_sales_data()
                try:
                    prediction = lstm_predict(sales_data)
                    trend_text = f"📈 Predicted next demand: **{prediction:.2f} units**"
                except Exception as e:
                    trend_text = f"⚠️ Prediction unavailable ({e})"

                st.markdown("---")
                col1, col2 = st.columns([1, 3])
                with col1:
                    if p["thumbnail"]:
                        st.image(p["thumbnail"], use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/200", use_container_width=True)
                with col2:
                    st.subheader(p["title"])
                    st.write(f"💰 **Price:** {p['price']}")
                    st.write(trend_text)
                    st.link_button("View on Amazon", p["link"])
