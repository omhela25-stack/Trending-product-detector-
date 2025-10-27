# Amazon Trending Product Predictor

A Dash web app that:
- Fetches top Amazon products using SerpApi
- Predicts product demand using an LSTM model
- Displays live product images, prices, and trend predictions

### 🚀 Deploy on Render
1. Push this repo to GitHub
2. Create new Web Service on Render
3. Select your GitHub repo
4. Use build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app.server`
6. Done ✅
