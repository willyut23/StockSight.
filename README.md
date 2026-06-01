# 📈 StockSight — ML-Powered Stock Price Forecasting

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

**StockSight** is a lightweight, single‑file web application that fetches historical stock data, trains a random forest model on‑the‑fly, and predicts future closing prices (1‑day or 5‑day horizon).  
It comes with a modern, responsive dashboard built right into the API — no separate frontend needed.

> 🧪 **For educational purposes only. This is not financial advice.**

---

## ✨ Features

- 🔮 **One‑click forecasts** for any US stock or crypto pair (e.g. AAPL, TSLA, BTC-USD)
- 📊 **Interactive chart** – historical prices + predicted trajectory (Plotly)
- 🤖 **Random Forest model** trained on lagged prices, volume, rolling means, and volatility
- 📈 **Confidence score** – R² on a hold‑out validation set
- 💾 **In‑memory cache** – data stays fresh for 1 hour; repeated calls are instant
- 🧭 **Quick‑pick buttons** for popular tickers
- 🌗 **Dark, elegant UI** – fully embedded, no extra build step
- ⚡ **FastAPI backend** with automatic input validation and error handling

---

## 🛠️ Tech Stack

| Layer        | Technology |
|--------------|------------|
| Web framework| FastAPI (Python) |
| Data         | yfinance, pandas, numpy |
| Machine Learning | scikit-learn (RandomForestRegressor) |
| Frontend     | Vanilla HTML/CSS/JS + Plotly CDN |
| Server       | Uvicorn (ASGI) |

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

    # Clone the repository
    git clone https://github.com/willyut23/StockSight-..git
    cd StockSight-.

    # Create a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate      # Windows: venv\Scripts\activate

    # Install dependencies
    pip install -r requirements.txt

**requirements.txt** (create this file if missing):

    fastapi
    uvicorn
    yfinance
    pandas
    numpy
    scikit-learn

---

## 🚀 Usage

Start the server:

    uvicorn app:app --reload

Then open your browser and go to **http://127.0.0.1:8000** — the full dashboard will appear.

### How to predict
1. Type a ticker symbol (e.g. `MSFT`) in the input field or click a quick‑pick button.
2. Choose a forecast horizon (`1 Day` or `5 Days`).
3. Click **Forecast →**.

The chart, prediction table, and R² confidence gauge will update immediately.

---

## 📡 API Endpoints

| Method | Endpoint    | Description |
|--------|-------------|-------------|
| `GET`  | `/`         | Serves the embedded frontend |
| `GET`  | `/health`   | API health check + list of cached tickers |
| `POST` | `/predict`  | Run a prediction |

### `POST /predict`

**Request body** (JSON):

    {
      "ticker": "AAPL",
      "horizon": 5
    }

- `ticker` (string, required) – Yahoo Finance ticker symbol.
- `horizon` (integer, 1 or 5) – number of business days to forecast.

**Response** (200):

    {
      "ticker": "AAPL",
      "horizon": 5,
      "r2": 0.872,
      "hist_dates": ["2024-01-02", "..."],
      "hist_prices": [185.64, "..."],
      "future_dates": ["2024-01-12", "..."],
      "future_prices": [192.33, "..."],
      "weekend_warning": false,
      "cached": false,
      "data_points": 124
    }

Errors return appropriate HTTP status codes (400, 422, 503) with a `detail` field.

---

## 📁 Project Structure

    StockSight-./
    ├── app.py              # Main application – API + embedded HTML
    ├── requirements.txt    # Python dependencies
    ├── README.md
    └── assets/             # (optional) screenshots, GIFs

> The entire project lives in a single file (`app.py`) for maximum simplicity.

---

## 🧠 How the Model Works

1. **Data fetching** – 6 months of daily OHLCV via `yfinance`.  
2. **Feature engineering** – 10 lagged close prices, 3 lagged volumes, 5/10/20‑day moving averages, rolling standard deviation, and daily high‑low range.  
3. **Training** – 80/20 time‑series split (no shuffling). Features are standardised.  
4. **Model** – `RandomForestRegressor` (100 trees, max_depth=6).  
5. **Iterative forecasting** – For multi‑day horizon, the model predicts one step ahead, then uses the predicted price to create the next day’s input features.  
6. **Confidence** – R² score on the validation set is displayed to help you gauge reliability.

---

## ⚠️ Important Notes

- The predictions are **purely statistical** and should **never** be used for real trading decisions.  
- Weekend/holiday extrapolation is a rough approximation — the dashboard warns you when the forecast window includes non‑trading days.  
- The model is retrained on every request; for a production system you’d want to persist models and update them incrementally.  
- Data is cached for 1 hour to avoid hitting Yahoo Finance rate limits.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to improve.  
Some ideas:
- Add more ticker validation (NASDAQ, NYSE, crypto).
- Include technical indicators like RSI or MACD as features.
- Persist trained models to disk.
- Deploy with Docker or cloud providers.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

- **Author:** William Lowry
- **GitHub:** [@willyut23](https://github.com/willyut23)
- **Email:** WilliamLowry341@outlook.com
---

*Built with ❤️ for curious minds who love finance and machine learning.*
