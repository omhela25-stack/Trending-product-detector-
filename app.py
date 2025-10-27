# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from pytrends.request import TrendReq
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time
import math
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Amazon Trending Detector", layout="wide", page_icon="🛍️")

# ---------------- STYLES ----------------
st.markdown(
    """
    <style>
    .product-card {
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }
    .small-muted { color: #6b7280; font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Settings & Data Sources")

with st.sidebar.expander("Keywords & Data"):
    default_keywords = "smartwatch, wireless earbuds, sneakers"
    keywords_input = st.text_input("Enter keywords (comma-separated):", default_keywords)
    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

with st.sidebar.expander("Google Trends (pytrends)"):
    region = st.selectbox("Region (pytrends geo)", ["Worldwide", "IN", "US"], index=1)
    timeframe = st.selectbox("Timeframe (pytrends)", ["today 3-m", "today 12-m"], index=0)
    future_days = st.slider("Predict future days (LSTM)", min_value=3, max_value=21, value=7)

with st.sidebar.expander("Amazon Data (SerpApi - optional)"):
    st.markdown(
        "If you provide a **SerpApi** key the app will fetch live Amazon results.\n"
        "If left blank, the app will use sample data for demo purposes."
    )
    serpapi_key = st.text_input("SerpApi API Key (optional)", type="password")
    amazon_domain = st.selectbox("Amazon domain", ["amazon.in", "amazon.com", "amazon.co.uk"], index=0)
    results_per_keyword = st.number_input("Products per keyword", min_value=1, max_value=10, value=5)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ — LSTM predictions + Amazon product cards")

# ---------------- HELPERS ----------------
@st.cache_data(ttl=3600)
def get_google_trends(keyword_list, timeframe="today 3-m", geo="IN"):
    """Return a DataFrame (interest_over_time) for keywords from pytrends."""
    pytrends = TrendReq(hl='en-US', tz=360)
    for attempt in range(3):
        try:
            pytrends.build_payload(keyword_list, timeframe=timeframe, geo=geo if geo != "Worldwide" else "")
            df = pytrends.interest_over_time()
            if not df.empty and "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            return df
        except Exception as e:
            time.sleep(1)
            continue
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_amazon_serp(keyword, serpapi_key, amazon_domain="amazon.in", num_results=5):
    """
    Uses SerpApi to fetch Amazon results.
    API reference: https://serpapi.com/
    Example: https://serpapi.com/search.json?engine=amazon&q=smartwatch&api_key=YOUR_KEY&amazon_domain=amazon.com
    """
    if not serpapi_key:
        return pd.DataFrame()

    params = {
        "engine": "amazon",
        "q": keyword,
        "api_key": serpapi_key,
        "amazon_domain": amazon_domain,
        "num": num_results
    }
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        results = data.get("organic_results", []) or data.get("search_results", [])
        rows = []
        for item in results[:num_results]:
            title = item.get("title") or item.get("product_title") or item.get("name")
            price = item.get("price") or item.get("price_text") or item.get("price_string") or None
            # Normalize price to a number if possible
            price_num = None
            if price:
                try:
                    # remove non-digit except dot and comma
                    cleaned = ''.join(ch for ch in price if ch.isdigit() or ch in ".,")
                    cleaned = cleaned.replace(",", "")
                    price_num = float(cleaned)
                except:
                    price_num = None
            rating = None
            try:
                rating = float(item.get("rating", item.get("stars", 0)))
            except:
                rating = None
            reviews = None
            try:
                reviews = int(item.get("ratings_total", item.get("reviews", 0) or 0))
            except:
                reviews = None
            image = item.get("thumbnail") or item.get("image") or item.get("product_image")
            link = item.get("link") or item.get("product_link")
            rows.append({
                "keyword": keyword,
                "title": title,
                "price_raw": price,
                "price": price_num,
                "rating": rating if rating is not None else np.nan,
                "reviews": reviews if reviews is not None else np.nan,
                "image": image,
                "link": link
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def sample_amazon_products(keyword, n=5):
    """Generate realistic sample product rows for demo purposes."""
    sample_images = [
        "https://m.media-amazon.com/images/I/61m0lZtZfYL._AC_UL320_.jpg",
        "https://m.media-amazon.com/images/I/71S8U9VzLTL._AC_UL320_.jpg",
        "https://m.media-amazon.com/images/I/61D4Y1jQnTL._AC_UL320_.jpg",
        "https://m.media-amazon.com/images/I/61f6k+IyNwL._AC_UL320_.jpg",
        "https://m.media-amazon.com/images/I/81e4D1Q6+eL._AC_UL320_.jpg"
    ]
    rows = []
    for i in range(n):
        rows.append({
            "keyword": keyword,
            "title": f"{keyword.title()} Model {['A','B','C','X','Pro'][i%5]}",
            "price_raw": f"₹{np.random.randint(799, 9999)}",
            "price": float(np.random.randint(799, 9999)),
            "rating": round(np.random.uniform(3.6, 5.0), 1),
            "reviews": int(np.random.randint(50, 8000)),
            "image": sample_images[i % len(sample_images)],
            "link": "https://www.amazon.example/product"
        })
    return pd.DataFrame(rows)

def compute_trend_score_for_product(product_row, keyword_trend_latest, predicted_trend_next):
    """
    Compute a composite trend score based on:
    - rating (1-5)
    - reviews (log scaled)
    - price (lower price gives slight boost)
    - current trend interest and predicted increase
    Returns score in 0-100.
    """
    rating = product_row.get("rating", np.nan)
    reviews = product_row.get("reviews", 0) or 0
    price = product_row.get("price", np.nan)

    # rating component (0-40)
    rating_score = 0
    if not math.isnan(rating):
        rating_score = max(0, min(40, (rating - 1) / 4 * 40))

    # reviews component (0-30) - log scale
    reviews_score = 0
    if reviews > 0:
        reviews_score = min(30, math.log10(reviews + 1) / 5 * 30)

    # price component (0-10) - prefer mid-range (not strictly necessary)
    price_score = 5
    if not math.isnan(price):
        # cheaper products slightly favored
        price_score = max(0, min(10, (1 / (1 + (price / 2000))) * 10))

    # trend components (0-20)
    trend_score_component = 0
    if not np.isnan(keyword_trend_latest) and not np.isnan(predicted_trend_next):
        # if predicted increases relative to current, give a boost
        try:
            change_pct = (predicted_trend_next - keyword_trend_latest) / (keyword_trend_latest + 1e-6)
            # clamp
            change_pct = max(-1, min(1, change_pct))
            trend_score_component = (change_pct + 1) / 2 * 20  # maps [-1,1] to [0,20]
        except:
            trend_score_component = 10

    total = rating_score + reviews_score + price_score + trend_score_component
    # normalize to 0-100
    total = max(0, min(100, total))
    return round(total, 1)

def build_lstm_prediction(series, future_steps=7, seq_len=14, epochs=6, batch_size=8):
    """Train a minimal LSTM on a 1-D series and predict future_steps ahead."""
    series = series.dropna()
    if len(series) < seq_len + 2:
        return np.array([])

    # use last seq_len*3 values for training for speed/stability
    series = series[-(seq_len * 3):]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    preds = []
    last_seq = X[-1]
    for _ in range(future_steps):
        p = model.predict(last_seq.reshape(1, seq_len, 1), verbose=0)[0, 0]
        preds.append(p)
        last_seq = np.append(last_seq[1:], [[p]], axis=0)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds

# ---------------- MAIN LOGIC ----------------
st.title("🛍️ Amazon Trending Product Detector — LSTM + Live/Product Cards")

if len(keywords) == 0:
    st.warning("Please enter at least one keyword in the sidebar.")
    st.stop()

col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### Summary & Downloads")
    info_text = (
        "- Enter keywords at left. Optionally provide a **SerpApi** key in the sidebar to fetch real Amazon results.\n"
        "- The app will fetch Google Trends for each keyword and train a lightweight LSTM to predict future interest.\n"
        "- Products are scored combining rating, reviews, price and predicted trend change."
    )
    st.info(info_text)
    st.markdown("---")

# Fetch trends for all keywords at once for efficiency
with st.spinner("Fetching Google Trends..."):
    geo_code = "" if region == "Worldwide" else region
    trends_df = get_google_trends(keywords, timeframe=timeframe, geo=geo_code)

# Prepare a place to collect all product rows
all_products = []

for kw in keywords:
    st.markdown(f"## 🔎 Keyword: **{kw}**")
    with st.container():
        # Column layout: left for graphs, right for products
        gcol, pcol = st.columns([1.3, 1])

        # Get trends series for this keyword
        series = pd.Series(dtype=float)
        if not trends_df.empty and kw in trends_df.columns:
            series = trends_df[kw]
            with gcol:
                st.markdown("**Trend (Google search interest)**")
                st.line_chart(series, height=220)
        else:
            with gcol:
                st.warning("Google Trends data not available for this keyword.")

        # LSTM prediction
        with gcol:
            if not series.empty:
                with st.spinner("Training LSTM and predicting..."):
                    preds = build_lstm_prediction(series, future_steps=future_days, seq_len=14, epochs=6, batch_size=8)
                if preds.size > 0:
                    future_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=len(preds), freq='D')
                    pred_series = pd.Series(preds, index=future_index)
                    st.markdown("**Predicted interest (next days)**")
                    st.line_chart(pred_series)
                    latest_trend_val = series.iloc[-1]
                    predicted_next = float(preds[-1])
                else:
                    st.info("Not enough historical data for LSTM prediction.")
                    latest_trend_val = np.nan
                    predicted_next = np.nan
            else:
                latest_trend_val = np.nan
                predicted_next = np.nan

        # Fetch Amazon products (live or sample)
        with pcol:
            st.markdown("**Products**")
            if serpapi_key:
                with st.spinner("Fetching Amazon results via SerpApi..."):
                    df_products = fetch_amazon_serp(kw, serpapi_key, amazon_domain=amazon_domain, num_results=results_per_keyword)
                if df_products.empty:
                    st.warning("No live results returned; falling back to sample products.")
                    df_products = sample_amazon_products(kw, n=results_per_keyword)
            else:
                df_products = sample_amazon_products(kw, n=results_per_keyword)

            # compute scores and display cards
            product_cards = []
