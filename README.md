# StockSight

**ML-powered stock price forecasting**

![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?style=flat-square&logo=fastapi)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## What It Does

Fetches live stock data, trains an ML model (Random Forest), and predicts closing prices 1 or 5 days ahead.

---

## One File

Everything is in **`app.py`** — backend, ML, and frontend.

```bash
pip install fastapi uvicorn yfinance pandas numpy scikit-learn plotly
uvicorn app:app --reload
Then open http://localhost:8000

Features
Feature	Description
1-Day Forecast	Predicts next trading day's close
5-Day Forecast	Predicts next 5 trading days
Interactive Chart	Historical + forecasted prices
R² Score	Model confidence (0–1 scale)
Caching	No redundant API calls for 1 hour
10+ Quick Picks	AAPL, TSLA, NVDA, MSFT, etc.
Requirements
Python 3.8+

Internet connection

No API keys needed

Disclaimer
For educational purposes only. Not financial advice.

License
MIT
