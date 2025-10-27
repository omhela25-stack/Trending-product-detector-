import requests
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# ----------------------------
# 🔑 SerpApi Key (replace with your own)
# ----------------------------
SERPAPI_KEY = "4427e6d1d612ec487682027e5fc7ac384c21317cecd0fe503d785c10c6c6595c"

# ----------------------------
# ⚙️ Dash App Setup
# ----------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server  # for Render deployment

# ----------------------------
# 📦 Helper: Fetch Amazon Products via SerpApi
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
# 🧠 Helper: Dummy sales trend simulation + LSTM prediction
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
# 🎨 App Layout
# ----------------------------
app.layout = dbc.Container([
    html.H2("🛍️ Amazon Trending Product Predictor", className="text-center my-4"),

    html.Div([
        html.Label("Select Keyword:"),
        dcc.Dropdown(
            id="keyword",
            options=[
                {"label": "Smartphone", "value": "smartphone"},
                {"label": "Laptop", "value": "laptop"},
                {"label": "Headphones", "value": "headphones"},
                {"label": "Smartwatch", "value": "smartwatch"},
                {"label": "Shoes", "value": "shoes"},
                {"label": "Camera", "value": "camera"},
            ],
            value="smartphone",
            clearable=False
        ),
        html.Br(),
        dbc.Button("Fetch & Predict", id="analyze", color="primary", className="mb-4"),
    ], className="text-center"),

    dbc.Row(id="output", className="gy-4")
])

# ----------------------------
# 🔁 Callback
# ----------------------------
@app.callback(
    Output("output", "children"),
    Input("analyze", "n_clicks"),
    State("keyword", "value"),
    prevent_initial_call=True
)
def analyze_products(n, keyword):
    products = fetch_amazon_products(keyword)
    cards = []

    for p in products:
        try:
            sales_data = generate_sales_data()
            prediction = lstm_predict(sales_data)
            trend_text = f"Predicted next demand: {prediction:.2f} units"
        except Exception as e:
            trend_text = f"Prediction unavailable ({e})"

        card = dbc.Col(
            dbc.Card([
                dbc.CardImg(src=p["thumbnail"], top=True, style={"height": "250px", "object-fit": "contain"}),
                dbc.CardBody([
                    html.H5(p["title"], className="card-title"),
                    html.P(f"💰 Price: {p['price']}", className="card-text"),
                    html.P(trend_text, className="text-info"),
                    dbc.Button("View on Amazon", href=p["link"], target="_blank", color="success", className="mt-2")
                ])
            ], style={"height": "100%"}), width=4
        )
        cards.append(card)

    return dbc.Row(cards)

# ----------------------------
# 🚀 Run Server
# ----------------------------
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# ----------------------------
# 🔑 SerpApi Key (replace with your own)
# ----------------------------
SERPAPI_KEY = "4427e6d1d612ec487682027e5fc7ac384c21317cecd0fe503d785c10c6c6595c"

# ----------------------------
# ⚙️ Dash App Setup
# ----------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server  # for Render deployment

# ----------------------------
# 📦 Helper: Fetch Amazon Products via SerpApi
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
# 🧠 Helper: Dummy sales trend simulation + LSTM prediction
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
# 🎨 App Layout
# ----------------------------
app.layout = dbc.Container([
    html.H2("🛍️ Amazon Trending Product Predictor", className="text-center my-4"),

    html.Div([
        html.Label("Select Keyword:"),
        dcc.Dropdown(
            id="keyword",
            options=[
                {"label": "Smartphone", "value": "smartphone"},
                {"label": "Laptop", "value": "laptop"},
                {"label": "Headphones", "value": "headphones"},
                {"label": "Smartwatch", "value": "smartwatch"},
                {"label": "Shoes", "value": "shoes"},
                {"label": "Camera", "value": "camera"},
            ],
            value="smartphone",
            clearable=False
        ),
        html.Br(),
        dbc.Button("Fetch & Predict", id="analyze", color="primary", className="mb-4"),
    ], className="text-center"),

    dbc.Row(id="output", className="gy-4")
])

# ----------------------------
# 🔁 Callback
# ----------------------------
@app.callback(
    Output("output", "children"),
    Input("analyze", "n_clicks"),
    State("keyword", "value"),
    prevent_initial_call=True
)
def analyze_products(n, keyword):
    products = fetch_amazon_products(keyword)
    cards = []

    for p in products:
        try:
            sales_data = generate_sales_data()
            prediction = lstm_predict(sales_data)
            trend_text = f"Predicted next demand: {prediction:.2f} units"
        except Exception as e:
            trend_text = f"Prediction unavailable ({e})"

        card = dbc.Col(
            dbc.Card([
                dbc.CardImg(src=p["thumbnail"], top=True, style={"height": "250px", "object-fit": "contain"}),
                dbc.CardBody([
                    html.H5(p["title"], className="card-title"),
                    html.P(f"💰 Price: {p['price']}", className="card-text"),
                    html.P(trend_text, className="text-info"),
                    dbc.Button("View on Amazon", href=p["link"], target="_blank", color="success", className="mt-2")
                ])
            ], style={"height": "100%"}), width=4
        )
        cards.append(card)

    return dbc.Row(cards)

# ----------------------------
# 🚀 Run Server
# ----------------------------
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
