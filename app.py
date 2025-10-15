import streamlit as st
import pandas as pd
import numpy as np
import requests
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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

# ------------------------ CACHED AMAZON API HELPER ------------------------
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

# ------------------------ SIMPLE TREND PREDICTOR (LINEAR REGRESSION) ------------------------
def predict_trend_linear(series, future_steps=7):
    model = LinearRegression()
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    model.fit(X, y)
    future_X = np.arange(len(series), len(series) + future_steps).reshape(-1, 1)
    predictions = model.predict(future_X)
    return predictions

# ------------------------ MAIN APP ------------------------
pytrends = TrendReq(hl='en-US', tz=330)

if not api_key:
    st.error("Enter your Rainforest API key in the sidebar!")
    st.stop()

try:
    pytrends.build_payload(kw_list=keywords_list, geo="" if region=="Worldwide" else region, timeframe=timeframe)
    trend_data = pytrends.interest_over_time().drop(columns=['isPartial'], errors='ignore')
    st.subheader("📈 Historical Google Trends Data")
    st.line_chart(trend_data)
except Exception as e:
    st.error(f"Error fetching Google Trends data: {e}")
    st.stop()

# ------------------------ FORECASTING ------------------------
st.subheader("🤖 Predicted Trend for Next 7 Days (ML Forecast)")
predicted_growth = {}
combined_plot_data = pd.DataFrame()

for kw in trend_data.columns:
    series = trend_data[kw]
    future_pred = predict_trend_linear(series, future_steps=7)
    predicted_growth[kw] = future_pred.mean() - series.tail(7).mean()
    combined_series = pd.Series(list(series[-14:]) + list(future_pred),
                                index=pd.date_range(end=pd.Timestamp.today() + pd.Timedelta(days=7), periods=21))
    combined_plot_data[kw] = combined_series

# ------------------------ PLOT ------------------------
st.subheader("📊 Current vs Predicted Trend Growth Chart")
fig, ax = plt.subplots(figsize=(10,5))
for kw in combined_plot_data.columns:
    ax.plot(combined_plot_data.index, combined_plot_data[kw], label=kw)
ax.axvspan(pd.Timestamp.today(), combined_plot_data.index[-1], color='orange', alpha=0.1, label="Predicted")
ax.set_title("Google Trends: Current + Predicted")
ax.set_ylabel("Interest (0-100)")
ax.legend()
st.pyplot(fig)

# ------------------------ TOP KEYWORDS ------------------------
trending_keywords = pd.Series(predicted_growth).sort_values(ascending=False).head(5)
st.subheader("🔥 Predicted Top Trending Keywords")
colored_keywords = [f"🟢 {kw} (+{growth:.2f})" if growth > 0 else f"🔴 {kw} ({growth:.2f})"
                    for kw, growth in trending_keywords.items()]
st.markdown("<br>".join(colored_keywords), unsafe_allow_html=True)

# ------------------------ AMAZON PRODUCTS ------------------------
st.subheader("🛒 Top Amazon Products for Trending Keywords")
all_products = []

for kw in trending_keywords.index:
    st.markdown(f"### 🏷️ {kw.title()}")
    df = get_amazon_products(api_key, kw, max_results=top_products_count)
    if not df.empty:
        all_products.append(df)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df)
        with col2:
            plt.figure(figsize=(5,4))
            plt.scatter(df['Price'], df['Rating'], s=df['Reviews'], c='green', alpha=0.6)
            plt.xlabel("Price")
            plt.ylabel("Rating")
            plt.title(f"Price vs Rating ({kw.title()})")
            st.pyplot(plt)
        value_picks = df.sort_values(['Rating','Price'], ascending=[False,True]).head(3)
        st.markdown("**💎 Top 3 Value Picks:**")
        for _, row in value_picks.iterrows():
            st.markdown(f"- [{row['Title']}]({row['Link']}) | ₹{row['Price']} | ⭐ {row['Rating']} | {row['Reviews']} reviews")
    else:
        st.warning(f"No products found for '{kw}'.")

if all_products:
    combined_df = pd.concat(all_products, ignore_index=True)
    csv_data = combined_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predicted Products CSV", data=csv_data,
                       file_name="trending_products.csv", mime="text/csv")

st.caption(f"Made with ❤️ using Streamlit, Google Trends, Rainforest API, and AI forecasting. Auto-refresh disabled in lightweight mode.")
