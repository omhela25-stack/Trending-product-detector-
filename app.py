# app.py
# app.py
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import requests

# ---------------- SETTINGS ----------------
SERPAPI_KEY = "4427e6d1d612ec487682027e5fc7ac384c21317cecd0fe503d785c10c6c6595c"
keywords_list = ["smartwatch", "wireless earbuds", "sneakers", "perfume", "power bank"]
amazon_domain = "amazon.in"
results_per_keyword = 5
future_days = 7

# ---------------- HELPER FUNCTIONS ----------------
def fetch_amazon_products(keyword, serpapi_key, amazon_domain="amazon.in", num_results=5):
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
            image = item.get("thumbnail") or \
                    (item.get("product_images")[0]["link"] if "product_images" in item and item["product_images"] else "https://via.placeholder.com/150")
            rows.append({
                "title": item.get("title") or item.get("product_title") or "Unnamed Product",
                "price_raw": item.get("price") or item.get("price_text") or "N/A",
                "rating": item.get("rating", round(np.random.uniform(3.5,5.0),1)),
                "reviews": item.get("reviews", np.random.randint(100,10000)),
                "image": image,
                "link": item.get("link") or "#"
            })
        return pd.DataFrame(rows)
    except Exception:
        # fallback demo products
        imgs = [
            "https://m.media-amazon.com/images/I/61m0lZtZfYL._AC_UL320_.jpg",
            "https://m.media-amazon.com/images/I/71S8U9VzLTL._AC_UL320_.jpg",
            "https://m.media-amazon.com/images/I/61D4Y1qQnTL._AC_UL320_.jpg",
            "https://m.media-amazon.com/images/I/81e4D1Q6+eL._AC_UL320_.jpg"
        ]
        return pd.DataFrame([{
            "title": f"{keyword.title()} Model {chr(65+i)}",
            "price_raw": f"₹{np.random.randint(799,9999)}",
            "rating": round(np.random.uniform(3.5,5.0),1),
            "reviews": np.random.randint(100,10000),
            "image": imgs[i % len(imgs)],
            "link": "#"
        } for i in range(num_results)])

def predict_trend_placeholder(n=future_days):
    return np.round(np.random.uniform(0.5,1.0,size=n),2)

# ---------------- DASH APP ----------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("🛍️ Amazon Trending Products Dashboard", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Select Keyword:"),
            dcc.Dropdown(
                id="keyword-dropdown",
                options=[{"label": k.title(), "value": k} for k in keywords_list],
                value=keywords_list[0],
                clearable=False
            )
        ], width=4)
    ], className="mb-4"),

    dbc.Row(id="products-container")
], fluid=True)

# ---------------- CALLBACK ----------------
from dash.dependencies import Input, Output

@app.callback(
    Output("products-container", "children"),
    [Input("keyword-dropdown", "value")]
)
def update_products(keyword):
    products_df = fetch_amazon_products(keyword, SERPAPI_KEY, amazon_domain, results_per_keyword)
    cards = []
    for _, row in products_df.iterrows():
        pred = predict_trend_placeholder(future_days)
        card = dbc.Card(
            [
                dbc.Row([
                    dbc.Col(html.Img(src=row["image"], style={"width":"100%","height":"200px","object-fit":"contain"}), width=3),
                    dbc.Col([
                        html.H5(html.A(row["title"], href=row["link"], target="_blank")),
                        html.P(f"💰 {row['price_raw']} | ⭐ {row['rating']} | 🗳️ {row['reviews']} reviews")
                    ], width=5),
                    dbc.Col(dcc.Graph(
                        figure={
                            "data":[{"y": pred, "type":"line", "name":"Prediction"}],
                            "layout":{"height":200, "margin":{"l":20,"r":20,"t":20,"b":20}}
                        }
                    ), width=4)
                ])
            ], className="mb-4 p-2", style={"box-shadow":"0 2px 5px rgba(0,0,0,0.1)"}
        )
        cards.append(dbc.Col(card, width=12))
    return cards

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run_server(debug=True)
