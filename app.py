# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Amazon Trending Detector", layout="wide")
st.title("🤖 AI-Powered Amazon Trending Product Dashboard")

# ---------------- USER INPUT ----------------
default_keywords = "smartwatch, wireless earbuds, sneakers, perfume, power bank"
keywords = st.text_input("Enter product keywords (comma-separated):", default_keywords)
keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
region = st.selectbox("Region", ["Worldwide", "IN", "US"], index=1)
timeframe = st.selectbox("Timeframe", ["today 3-m", "today 12-m"], index=0)
st.sidebar.title("Rainforest API Key (optional for demo)")
api_key = st.sidebar.text_input("API Key", type="password")

refresh_btn = st.sidebar.button("🔄 Refresh / Detect now")
refresh_interval = st.sidebar.number_input("Auto-refresh interval (minutes) — use Refresh button instead", min_value=1, value=10)
top_products_count = st.sidebar.number_input("Number of top products per keyword", min_value=1, max_value=10, value=5)

st.info("Tip: Enter your Rainforest API key in the sidebar to fetch real Amazon listings. If left empty, product fetch will be skipped.")

# ---------------- AMAZON API HELPER (cached) ----------------
@st.cache_data(ttl=3600)
def get_amazon_products(api_key: str, keyword: str, country_domain: str = "amazon.in", max_results: int = 5):
    """
    Returns DataFrame with columns: Title, Price (float or NaN), Rating (float), Reviews (int), Link
    """
    if not api_key:
        return pd.DataFrame()  # no API key -> return empty
    api_url = "https://api.rainforestapi.com/request"
    params = {
        "api_key": api_key,
        "type": "search",
        "amazon_domain": country_domain,
        "search_term": keyword,
        "page": 1
    }
    try:
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if "search_results" not in data:
            return pd.DataFrame()
        items = data["search_results"][:max_results]
        rows = []
        for item in items:
            # Price parsing (robust)
            price_raw = item.get("price", {}).get("raw")
            price_val = np.nan
            if price_raw:
                # remove currency symbols and non-numeric chars
                p = re.sub(r"[^\d.]", "", str(price_raw))
                try:
                    price_val = float(p) if p else np.nan
                except:
                    price_val = np.nan
            rating = item.get("rating") or 0
            try:
                rating = float(rating)
            except:
                rating = 0.0
            reviews = item.get("ratings_total") or 0
            try:
                reviews = int(reviews)
            except:
                reviews = 0
            rows.append({
                "Title": item.get("title"),
                "Price": price_val,
                "Rating": rating,
                "Reviews": reviews,
                "Link": item.get("link")
            })
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        st.warning(f"Amazon fetch failed for '{keyword}': {e}")
        return pd.DataFrame()

# ---------------- FAST LSTM HELPER ----------------
def predict_trend_lstm_fast(series, future_steps=7, seq_len=14, epochs=12):
    """
    Fast LSTM forecast. Returns numpy array of length future_steps.
    If input series too short, raises ValueError.
    """
    if len(series) < seq_len + 2:
        raise ValueError("Not enough history for LSTM (need at least seq_len+2 points).")

    series = series[-(seq_len*3):]  # use recent window
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1,1))

    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Small model for speed
    model = Sequential()
    model.add(LSTM(25, return_sequences=False, input_shape=(X.shape[1],1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=1, verbose=0)

    last_seq = scaled_data[-seq_len:].reshape(1, seq_len, 1)
    predictions = []
    for _ in range(future_steps):
        pred = model.predict(last_seq, verbose=0)
        predictions.append(pred[0,0])
        last_seq = np.append(last_seq[:,1:,:], [[pred]], axis=1)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()
    return predictions

# ---------------- MAIN ACTION ----------------
# Trigger detection on page load or when Refresh button pressed
do_run = True if refresh_btn else True  # always run once when user loads the page

if do_run:
    # Build Google Trends payload
    pytrends = TrendReq(hl='en-US', tz=330)
    if not keywords_list:
        st.error("Please enter at least one keyword.")
        st.stop()

    # Fetch trends
    st.subheader("📈 Historical Google Trends Data")
    try:
        pytrends.build_payload(kw_list=keywords_list, geo="" if region=="Worldwide" else region, timeframe=timeframe)
        trend_data = pytrends.interest_over_time()
        if trend_data.empty:
            st.warning("No Google Trends data returned for these keywords.")
            trend_data = pd.DataFrame()
        else:
            # drop isPartial if present
            if 'isPartial' in trend_data.columns:
                trend_data = trend_data.drop(columns=['isPartial'])
            # show trend chart
            plt.clf()
            st.line_chart(trend_data)
    except Exception as e:
        st.error(f"Error fetching Google Trends data: {e}")
        st.stop()

    if trend_data.empty:
        st.info("No trend data to analyze. Change keywords/timeframe.")
        st.stop()

    # LSTM predictions per keyword (safe guards)
    st.subheader("🤖 Predicted Trend for Next 7 Days (AI Forecast)")
    predicted_growth = {}
    combined_plot_data = pd.DataFrame()

    for kw in trend_data.columns:
        series = trend_data[kw].dropna()
        try:
            future_pred = predict_trend_lstm_fast(series, future_steps=7)
            predicted_growth[kw] = float(np.nanmean(future_pred) - np.nanmean(series.tail(7)))
            combined_series = pd.Series(list(series.tail(14)) + list(future_pred),
                                        index=pd.date_range(end=pd.Timestamp.today() + pd.Timedelta(days=7), periods=len(series.tail(14)) + 7))
            combined_plot_data[kw] = combined_series
        except Exception as e:
            # Not enough data for LSTM — fallback to simple projection (use last mean)
            st.warning(f"Skipping LSTM for '{kw}' — {e}. Using simple projection instead.")
            recent_mean = series.tail(7).mean() if len(series) >= 7 else series.mean()
            future_pred = np.repeat(recent_mean, 7)
            predicted_growth[kw] = 0.0
            combined_series = pd.Series(list(series.tail(14)) + list(future_pred),
                                        index=pd.date_range(end=pd.Timestamp.today() + pd.Timedelta(days=7), periods=len(series.tail(14)) + 7))
            combined_plot_data[kw] = combined_series

    # Combined plot
    st.subheader("📊 Current vs Predicted Trend Chart")
    plt.clf()
    fig, ax = plt.subplots(figsize=(10,5))
    for kw in combined_plot_data.columns:
        ax.plot(combined_plot_data.index, combined_plot_data[kw], label=kw)
    ax.axvspan(pd.Timestamp.today(), combined_plot_data.index[-1], color='orange', alpha=0.08, label="Predicted")
    ax.set_title("Google Trends: Current + Predicted")
    ax.set_ylabel("Interest (0-100)")
    ax.legend()
    st.pyplot(fig)

    # Top predicted keywords
    trending_keywords = pd.Series(predicted_growth).sort_values(ascending=False).head(5)
    st.subheader("🔥 AI Predicted Top Trending Keywords")
    colored_keywords = []
    for kw, growth in trending_keywords.items():
        arrow = "🟢" if growth > 0 else "🔴"
        colored_keywords.append(f"{arrow} {kw} ({growth:+.2f})")
    st.markdown("<br>".join(colored_keywords), unsafe_allow_html=True)

    # Amazon product fetch + scatter + value picks
    st.subheader("🛒 Top Amazon Products & Value Analysis")
    all_products = []

    for kw in trending_keywords.index:
        st.markdown(f"### 🏷️ {kw.title()}")
        df = get_amazon_products(api_key, kw, max_results=top_products_count)
        if df.empty:
            if not api_key:
                st.info("No API key provided — product fetch skipped. Provide Rainforest API key in sidebar to fetch real listings.")
            else:
                st.warning(f"No products found for '{kw}'.")
            continue

        # show table and scatter side-by-side
        all_products.append(df)
        col1, col2 = st.columns([1,1])
        with col1:
            st.dataframe(df)
        with col2:
            plt.clf()
            plt.figure(figsize=(5,4))
            # handle NaN prices by dropping them for scatter
            scatter_df = df.dropna(subset=['Price'])
            if not scatter_df.empty:
                plt.scatter(scatter_df['Price'], scatter_df['Rating'], s=np.clip(scatter_df['Reviews'], 10, 100), c='green', alpha=0.6)
                plt.xlabel("Price")
                plt.ylabel("Rating")
                plt.title(f"Price vs Rating ({kw.title()})")
                st.pyplot(plt)
            else:
                st.info("Price data not available for scatter plot.")

        # Top value picks
        value_picks = df.copy()
        # fill NaNs so sorting works
        value_picks['Price'] = value_picks['Price'].fillna(value_picks['Price'].max() if not value_picks['Price'].isna().all() else 0)
        value_picks = value_picks.sort_values(['Rating', 'Price'], ascending=[False, True]).head(3)
        st.markdown("**💎 Top 3 Value Picks:**")
        for _, row in value_picks.iterrows():
            price_str = "N/A" if pd.isna(row['Price']) else f"₹{row['Price']:.2f}"
            st.markdown(f"- [{row['Title']}]({row['Link']}) | {price_str} | ⭐ {row['Rating']} | {row['Reviews']} reviews")

    # CSV download
    if all_products:
        combined_df = pd.concat(all_products, ignore_index=True)
        csv_data = combined_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download AI Predicted Products CSV", data=csv_data,
                           file_name="ai_trending_products.csv", mime="text/csv")

    st.caption("Made with ❤️ using Streamlit, Google Trends, Rainforest API, and AI (Fast LSTM). Use the Refresh button in the sidebar to re-run.")
    # If user pressed Refresh, rerun to fetch fresh data
    if refresh_btn:
        st.experimental_rerun()
