# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import time

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="AI Amazon Trending Detector", layout="wide")
st.title("🤖 AI-Powered Amazon Trending Product Dashboard (Lightweight Version)")

# ------------------------ USER INPUT ------------------------
default_keywords = "smartwatch, wireless earbuds, sneakers, perfume, power bank"
keywords = st.text_input("Enter product keywords (comma-separated):", default_keywords)
keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
region = st.selectbox("Region", ["Worldwide", "IN", "US"], index=1)
timeframe = st.selectbox("Timeframe", ["today 3-m", "today 12-m"], index=0)
st.sidebar.title("Rainforest API Key")
api_key = st.sidebar.text_input("API Key", type="password")

refresh_interval = st.sidebar.number_input("Auto-refresh interval (minutes)", min_value=1, value=10)
top_products_count = st.sidebar.number_input("Number of top products per keyword", min_value=1, max_value=10, value=5)

# ------------------------ TEST API KEY ------------------------
def test_api_key(api_key):
    try:
        test_url = "https://api.rainforestapi.com/request"
        params = {
            "api_key": api_key,
            "type": "search",
            "amazon_domain": "amazon.in",
            "search_term": "test"
        }
        r = requests.get(test_url, params=params, timeout=10)
        data = r.json()
        if "request_info" in data and data["request_info"].get("success"):
            return True
        return False
    except Exception:
        return False

if not api_key:
    st.warning("⚠️ Please enter your Rainforest API key in the sidebar to start.")
    st.stop()

with st.spinner("🔍 Verifying your API key..."):
    if test_api_key(api_key):
        st.success("✅ API key verified successfully!")
    else:
        st.error("❌ Invalid or expired API key. Please check it and try again.")
        st.stop()

# ------------------------ AMAZON API HELPER ------------------------
@st.cache_data(ttl=3600)
def get_amazon_products(api_key: str, keyword: str, country_domain: str = "amazon.in", max_results: int = 5):
    api_url = "https://api.rainforestapi.com/request"
    params = {
        "api_key": api_key,
        "type": "search",
        "amazon_domain": country_domain,
        "search_term": keyword,
        "page": 1
    }
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        if "search_results" not in data:
            return pd.DataFrame()
        items = data["search_results"][:max_results]
        df = pd.DataFrame([{
            "Title": item.get("title"),
            "Price": float(item.get("price", {}).get("raw", "0").replace("₹","").replace(",","")) if item.get("price") else np.nan,
            "Rating": float(item.get("rating", 0)),
            "Reviews": int(item.get("ratings_total", 0)),
            "Link": item.get("link")
        } for item in items])
        return df
    except Exception as e:
        print(f"⚠️ Error fetching Amazon data for '{keyword}': {e}")
        return pd.DataFrame()

# ------------------------ LSTM TREND PREDICTION ------------------------
def predict_trend_lstm(series, future_steps=7, seq_len=14, epochs=10, batch_size=8):
    series = series[-(seq_len*3):]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1,1))

    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(25, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    predictions = []
    last_seq = X[-1]
    current_seq = last_seq
    for _ in range(future_steps):
        pred = model.predict(current_seq.reshape(1, seq_len, 1), verbose=0)[0,0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()
    return predictions

# ------------------------ GOOGLE TRENDS ------------------------
@st.cache_data(ttl=3600)
def get_google_trends(keywords, timeframe="today 3-m", region="IN"):
    pytrends = TrendReq()
    trends_data = pd.DataFrame()
    try:
        pytrends.build_payload(keywords, timeframe=timeframe, geo=region)
        trends_data = pytrends.interest_over_time()
        if not trends_data.empty and 'isPartial' in trends_data.columns:
            trends_data = trends_data.drop(columns=['isPartial'])
    except Exception as e:
        print(f"⚠️ Error fetching Google Trends: {e}")
    return trends_data

# ------------------------ DISPLAY RESULTS ------------------------
for keyword in keywords_list:
    st.subheader(f"📈 Keyword: {keyword}")

    trends_df = get_google_trends([keyword], timeframe=timeframe, region=region)
    if not trends_df.empty:
        st.line_chart(trends_df[keyword])
        pred = predict_trend_lstm(trends_df[keyword])
        st.line_chart(pd.DataFrame({f"{keyword} - predicted": pred}))

    amazon_df = get_amazon_products(api_key, keyword, max_results=top_products_count)
    if not amazon_df.empty:
        st.write(f"🛒 Top {top_products_count} Amazon products for '{keyword}':")
        st.dataframe(amazon_df)
    else:
        st.info(f"No products found for '{keyword}'.")

# ------------------------ AUTO-REFRESH ------------------------
st.info(f"🔄 Auto-refresh every {refresh_interval} minutes.")
time.sleep(refresh_interval*60)
st.experimental_rerun()
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import time

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="AI Amazon Trending Detector", layout="wide")
st.title("🤖 AI-Powered Amazon Trending Product Dashboard (Lightweight Version)")

# ------------------------ USER INPUT ------------------------
default_keywords = "smartwatch, wireless earbuds, sneakers, perfume, power bank"
keywords = st.text_input("Enter product keywords (comma-separated):", default_keywords)
keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
region = st.selectbox("Region", ["Worldwide", "IN", "US"], index=1)
timeframe = st.selectbox("Timeframe", ["today 3-m", "today 12-m"], index=0)
st.sidebar.title("Rainforest API Key")
api_key = st.sidebar.text_input("API Key", type="password")

refresh_interval = st.sidebar.number_input("Auto-refresh interval (minutes)", min_value=1, value=10)
top_products_count = st.sidebar.number_input("Number of top products per keyword", min_value=1, max_value=10, value=5)

# ------------------------ TEST API KEY ------------------------
def test_api_key(api_key):
    try:
        test_url = "https://api.rainforestapi.com/request"
        params = {
            "api_key": api_key,
            "type": "search",
            "amazon_domain": "amazon.in",
            "search_term": "test"
        }
        r = requests.get(test_url, params=params, timeout=10)
        data = r.json()
        if "request_info" in data and data["request_info"].get("success"):
            return True
        return False
    except Exception:
        return False

if not api_key:
    st.warning("⚠️ Please enter your Rainforest API key in the sidebar to start.")
    st.stop()

with st.spinner("🔍 Verifying your API key..."):
    if test_api_key(api_key):
        st.success("✅ API key verified successfully!")
    else:
        st.error("❌ Invalid or expired API key. Please check it and try again.")
        st.stop()

# ------------------------ AMAZON API HELPER ------------------------
@st.cache_data(ttl=3600)
def get_amazon_products(api_key: str, keyword: str, country_domain: str = "amazon.in", max_results: int = 5):
    api_url = "https://api.rainforestapi.com/request"
    params = {
        "api_key": api_key,
        "type": "search",
        "amazon_domain": country_domain,
        "search_term": keyword,
        "page": 1
    }
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        if "search_results" not in data:
            return pd.DataFrame()
        items = data["search_results"][:max_results]
        df = pd.DataFrame([{
            "Title": item.get("title"),
            "Price": float(item.get("price", {}).get("raw", "0").replace("₹","").replace(",","")) if item.get("price") else np.nan,
            "Rating": float(item.get("rating", 0)),
            "Reviews": int(item.get("ratings_total", 0)),
            "Link": item.get("link")
        } for item in items])
        return df
    except Exception as e:
        print(f"⚠️ Error fetching Amazon data for '{keyword}': {e}")
        return pd.DataFrame()

# ------------------------ LSTM TREND PREDICTION ------------------------
def predict_trend_lstm(series, future_steps=7, seq_len=14, epochs=10, batch_size=8):
    series = series[-(seq_len*3):]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1,1))

    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(25, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    predictions = []
    last_seq = X[-1]
    current_seq = last_seq
    for _ in range(future_steps):
        pred = model.predict(current_seq.reshape(1, seq_len, 1), verbose=0)[0,0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()
    return predictions

# ------------------------ GOOGLE TRENDS ------------------------
@st.cache_data(ttl=3600)
def get_google_trends(keywords, timeframe="today 3-m", region="IN"):
    pytrends = TrendReq()
    trends_data = pd.DataFrame()
    try:
        pytrends.build_payload(keywords, timeframe=timeframe, geo=region)
        trends_data = pytrends.interest_over_time()
        if not trends_data.empty and 'isPartial' in trends_data.columns:
            trends_data = trends_data.drop(columns=['isPartial'])
    except Exception as e:
        print(f"⚠️ Error fetching Google Trends: {e}")
    return trends_data

# ------------------------ DISPLAY RESULTS ------------------------
for keyword in keywords_list:
    st.subheader(f"📈 Keyword: {keyword}")

    trends_df = get_google_trends([keyword], timeframe=timeframe, region=region)
    if not trends_df.empty:
        st.line_chart(trends_df[keyword])
        pred = predict_trend_lstm(trends_df[keyword])
        st.line_chart(pd.DataFrame({f"{keyword} - predicted": pred}))

    amazon_df = get_amazon_products(api_key, keyword, max_results=top_products_count)
    if not amazon_df.empty:
        st.write(f"🛒 Top {top_products_count} Amazon products for '{keyword}':")
        st.dataframe(amazon_df)
    else:
        st.info(f"No products found for '{keyword}'.")

# ------------------------ AUTO-REFRESH ------------------------
st.info(f"🔄 Auto-refresh every {refresh_interval} minutes.")
time.sleep(refresh_interval*60)
st.experimental_rerun()
