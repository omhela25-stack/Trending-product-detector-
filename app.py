import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import warnings

# ---------------------------------------------------------
# 1. GLOBAL CONFIGURATION (MUST BE FIRST)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Group 32 - Capstone Super Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 2. CUSTOM CSS & STYLE
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* App background & text */
    .stApp {
        background-color: #0E1117;
        color: #E6E6FA;
        font-family: 'Roboto', 'Trebuchet MS', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #11111a;
        color: #ffffff;
    }

    /* Header */
    h1, h2, h3 {
        color: #00FFFF !important;
        font-family: 'Roboto', 'Trebuchet MS', sans-serif;
        font-weight: bold;
    }

    /* Metrics */
    [data-testid="stMetricValue"] { color: #39FF14 !important; font-size: 24px; }
    [data-testid="stMetricDelta"] { color: #FFD700 !important; font-size: 14px; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #FF4B4B, #FF9900);
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }

    /* Card like containers */
    .card {
        background:#1E1E2F;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Marquee ticker at top */
    .marquee-container {
        width: 100%;
        background-color: #1E1E2F;
        color: #FFD700;
        padding: 8px;
        white-space: nowrap;
        overflow: hidden;
        border-bottom: 2px solid #333;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    .marquee-content {
        display: inline-block;
        padding-left: 100%;
        animation: marquee 25s linear infinite;
        font-family: monospace;
        font-size: 16px;
    }
    @keyframes marquee {
        0%   { transform: translate(0, 0); }
        100% { transform: translate(-100%, 0); }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS & CACHING
# ---------------------------------------------------------

@st.cache_data(ttl=300)
def get_stock_marquee_data():
    """Fetches live prices for the top banner safely."""
    tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "RELIANCE.NS", "TCS.NS"]
    try:
        # Fetch data
        df = yf.download(tickers, period="1d", progress=False)
        
        # Check if data is empty
        if df.empty:
            return "Market Data Loading..."
            
        # Handle MultiIndex (Price, Ticker) vs Single Index
        if isinstance(df.columns, pd.MultiIndex):
            closes = df['Close'].iloc[-1]
            text_parts = []
            for ticker in tickers:
                if ticker in closes:
                    val = closes[ticker]
                    if pd.notnull(val):
                        text_parts.append(f"{ticker}: ${val:.2f}")
            return "  •  ".join(text_parts)
        else:
            return "Data structure mismatch (Try refreshing)"
    except Exception as e:
        return f"Market Data Unavailable"

def get_api_key():
    """Safely retrieves API key or returns None."""
    try:
        return st.secrets["api_keys"]["serpapi"]
    except:
        return None

# ---------------------------------------------------------
# 4. MODULE A: AMAZON TREND DETECTOR (LSTM SIMULATION)
# ---------------------------------------------------------
def generate_sales_data(n=90):
    np.random.seed(42)
    base = np.linspace(50, 200, n)
    noise = np.random.normal(0, 10, n)
    return base + noise

def fetch_amazon_products(keyword, num_results=6):
    products = []
    # Simulated product data for demo
    product_names = [
        f"{keyword} Pro Max", f"{keyword} Ultra", f"{keyword} Air", 
        f"{keyword} Lite", f"{keyword} X", f"{keyword} SE"
    ]
    for i in range(num_results):
        products.append({
            "title": product_names[i % len(product_names)],
            "price": np.random.randint(50, 2000),
            "thumbnail": f"https://source.unsplash.com/random/200x200?{keyword}&sig={i}",
            "link": "#"
        })
    return products

def render_amazon_dashboard():
    st.title("📦 AI-Powered Amazon Trending Detector")
    st.markdown("### Predict future product trends using AI")

    # Sidebar inputs specific to this module
    categories = ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Shoes", "Camera"]
    
    col_conf, col_main = st.columns([1, 3])
    
    with col_conf:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("⚙️ Settings")
        selected_cat = st.selectbox("Product Category", categories)
        future_days = st.slider("Prediction Horizon (Days)", 1, 30, 14)
        run_trend = st.button("🚀 Predict Trends")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_main:
        if run_trend:
            with st.spinner(f"Analyzing trends for {selected_cat}..."):
                # 1. Simulate Historical Data
                sales_data = generate_sales_data(90)
                dates = pd.date_range(end=pd.Timestamp.now(), periods=90)
                
                # 2. LSTM Simulation (Linear Projection for Stability)
                # Calculating trend slope
                slope = (sales_data[-1] - sales_data[-30]) / 30
                last_val = sales_data[-1]
                
                future_dates = pd.date_range(start=dates[-1], periods=future_days+1)[1:]
                # Adding volatility to prediction
                future_noise = np.random.normal(0, 5, future_days)
                predictions = [last_val + (slope * i) + future_noise[i-1] for i in range(1, future_days + 1)]
                
                # 3. Visualization
                col_chart, col_prods = st.columns([2, 1])
                
                with col_chart:
                    st.subheader(f"📈 Sales Forecast: {selected_cat}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dates, y=sales_data, mode='lines', name='Historical', line=dict(color='#00FFFF')))
                    fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines+markers', name='Predicted', line=dict(color='#FFA500', dash='dash')))
                    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_prods:
                    st.subheader("🛒 Top Picks")
                    products = fetch_amazon_products(selected_cat, 4)
                    for p in products:
                        with st.container():
                            c1, c2 = st.columns([1, 2])
                            with c1:
                                st.image(p['thumbnail'], use_column_width=True)
                            with c2:
                                st.write(f"**{p['title']}**")
                                st.caption(f"Price: ${p['price']}")
                                st.progress(np.random.randint(60, 95))

# ---------------------------------------------------------
# 5. MODULE B: STOCK PRICE PREDICTOR (RANDOM FOREST)
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_stock_history(ticker, period="1Y"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty: return pd.DataFrame()
        df = df.reset_index()
        # Flatten MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

def prepare_features(df, ma1=20, ma2=50):
    df['MA_Short'] = df['Close'].rolling(ma1).mean()
    df['MA_Long'] = df['Close'].rolling(ma2).mean()
    df['Return'] = df['Close'].pct_change()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Target: Next Day Close
    df['Target'] = df['Close'].shift(-1)
    
    # Lag Features
    for lag in [1, 2, 3]:
        df[f'Lag_{lag}'] = df['Close'].shift(lag)
        
    return df.dropna()

def render_stock_predictor():
    st.title("📈 Pro Stock Predictor")
    st.markdown("### Technical Analysis & Machine Learning Forecast")
    
    # Sidebar
    st.sidebar.subheader("Stock Config")
    ticker_input = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()
    period = st.sidebar.select_slider("History Period", ["3mo", "6mo", "1y", "2y", "5y"], value="1y")
    ma_short = st.sidebar.number_input("Short MA", 10, 50, 20)
    ma_long = st.sidebar.number_input("Long MA", 50, 200, 50)
    
    if st.button("Analyze Stock"):
        with st.spinner(f"Fetching data for {ticker_input}..."):
            df = fetch_stock_history(ticker_input, period)
            
            if df.empty:
                st.error("No data found. Please check ticker symbol.")
                return
            
            # Process Data
            data = prepare_features(df, ma_short, ma_long)
            
            if len(data) < 50:
                st.error("Not enough data points for training. Select a longer period.")
                return

            # Display Key Metrics
            current_price = df['Close'].iloc[-1]
            current_rsi = data['RSI'].iloc[-1]
            current_vol = df['Volume'].iloc[-1]
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Price", f"${current_price:.2f}")
            m2.metric("RSI (14)", f"{current_rsi:.2f}")
            m3.metric("Volume", f"{current_vol:,.0f}")
            
            # Machine Learning
            features = ['Open', 'High', 'Low', 'Volume', 'MA_Short', 'MA_Long', 'RSI', 'Lag_1', 'Lag_2', 'Lag_3']
            X = data[features]
            y = data['Target']
            
            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            
            # Predict Next Day
            last_row = X.iloc[[-1]]
            next_pred = model.predict(last_row)[0]
            
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <h3>🔮 Tomorrow's Prediction</h3>
                <h1 style="color: #39FF14;">${next_pred:.2f}</h1>
                <p>Model R² Score: {score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualizations
            tab1, tab2 = st.tabs(["Price Chart", "Model Performance"])
            
            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df[f'MA_Short'], line=dict(color='orange', width=1), name=f'MA {ma_short}'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df[f'MA_Long'], line=dict(color='blue', width=1), name=f'MA {ma_long}'))
                fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Plot Actual vs Predicted for Test Set
                test_preds = model.predict(X_test)
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(y=y_test.values, mode='lines', name='Actual', line=dict(color='cyan')))
                fig_test.add_trace(go.Scatter(y=test_preds, mode='lines', name='Predicted', line=dict(color='magenta', dash='dot')))
                fig_test.update_layout(title="Backtest Results", template="plotly_dark", height=400)
                st.plotly_chart(fig_test, use_container_width=True)
                
                # Feature Importance
                st.bar_chart(pd.Series(model.feature_importances_, index=features))

# ---------------------------------------------------------
# 6. MODULE C: PRODUCT CATALOG (SERPAPI + MOCK)
# ---------------------------------------------------------
# Complete catalog dictionary
CATALOG_DATA = {
    "Smartphones": [
        {"name": "iPhone 15 Pro", "desc": "Titanium design, A17 Pro chip."},
        {"name": "Samsung S24 Ultra", "desc": "Galaxy AI, 200MP camera."},
        {"name": "Google Pixel 8 Pro", "desc": "Best AI camera features."},
        {"name": "OnePlus 12", "desc": "Fastest charging flagship."}
    ],
    "Laptops": [
        {"name": "MacBook Air M3", "desc": "Thin, light, powerful."},
        {"name": "Dell XPS 14", "desc": "Premium Windows experience."},
        {"name": "ASUS ROG Zephyrus", "desc": "Top tier gaming laptop."},
        {"name": "Lenovo ThinkPad X1", "desc": "Best for business."}
    ],
    "Headphones": [
        {"name": "Sony WH-1000XM5", "desc": "Industry leading ANC."},
        {"name": "Bose QC Ultra", "desc": "Comfort king."},
        {"name": "AirPods Max", "desc": "Apple ecosystem integration."},
        {"name": "Sennheiser Momentum 4", "desc": "Audiophile grade sound."}
    ]
}

def render_product_catalog():
    st.title("🛒 Live Product Catalog")
    st.markdown("### Compare Prices Across Retailers")
    
    api_key = get_api_key()
    
    col_sel, col_content = st.columns([1, 3])
    
    with col_sel:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        cat_select = st.radio("Category", list(CATALOG_DATA.keys()))
        st.markdown('</div>', unsafe_allow_html=True)
        
        if not api_key:
            st.info("ℹ️ Demo Mode Active. Add `serpapi` key to secrets for live data.")
    
    with col_content:
        for item in CATALOG_DATA[cat_select]:
            with st.expander(f"🔹 {item['name']}", expanded=True):
                st.caption(item['desc'])
                
                if api_key:
                    # LIVE FETCH
                    try:
                        with st.spinner("Fetching live prices..."):
                            params = {"engine": "google_shopping", "q": item['name'], "api_key": api_key, "num": 3}
                            res = requests.get("https://serpapi.com/search.json", params=params).json()
                            
                            if "shopping_results" in res:
                                cols = st.columns(3)
                                for idx, offer in enumerate(res["shopping_results"][:3]):
                                    with cols[idx]:
                                        st.markdown(f"**{offer.get('source')}**")
                                        st.markdown(f"### {offer.get('price')}")
                                        st.markdown(f"[View Deal]({offer.get('link')})")
                            else:
                                st.warning("No live offers found at the moment.")
                    except:
                        st.error("API connection failed.")
                else:
                    # MOCK DATA
                    c1, c2, c3 = st.columns(3)
                    base_price = np.random.randint(500, 1500)
                    
                    with c1:
                        st.markdown("**Amazon**")
                        st.markdown(f"### ${base_price}")
                        st.button("Buy on Amazon", key=f"amz_{item['name']}")
                    with c2:
                        st.markdown("**BestBuy**")
                        st.markdown(f"### ${base_price - 20}")
                        st.button("Buy on BestBuy", key=f"bb_{item['name']}")
                    with c3:
                        st.markdown("**Walmart**")
                        st.markdown(f"### ${base_price + 15}")
                        st.button("Buy on Walmart", key=f"wm_{item['name']}")

# ---------------------------------------------------------
# 7. MAIN APP ROUTER & NAVIGATION
# ---------------------------------------------------------

# Display the Top Marquee
marquee_text = get_stock_marquee_data()
st.markdown(f"""
<div class="marquee-container">
    <div class="marquee-content">
        {marquee_text}
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation Layout
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to Module", ["Amazon Trends", "Stock Predictor", "Product Catalog"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Group 32")
st.sidebar.info("""
**Team Members:**
- **Om** (AI Models)
- **Swati** (Feature Eng)
- **Jyoti** (Frontend)
- **Srishti** (Frontend)
""")

# Routing
if page == "Amazon Trends":
    render_amazon_dashboard()
elif page == "Stock Predictor":
    render_stock_predictor()
elif page == "Product Catalog":
    render_product_catalog()
