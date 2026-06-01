# StockSight

ML-powered stock price forecasting

![Python](https://img.shields.io/badge/python-3.8+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688?style=flat-square&logo=fastapi)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## One File. Zero Bugs. Ready to Run.

**Only `app.py`** — backend, ML, and frontend all in one file.
pip install fastapi uvicorn yfinance pandas numpy scikit-learn plotly
uvicorn app:app --reload

Open http://localhost:8000

No API keys. No database. No build step. No other files.

---

## Features

**1-Day Forecast** — Predicts next trading day's closing price

**5-Day Forecast** — Predicts next 5 trading days

**Interactive Chart** — Historical + forecast with Plotly

**R² Confidence Score** — Model reliability (0–1 scale)

**Smart Caching** — 1-hour TTL, no redundant API calls

**10+ Quick Picks** — AAPL, TSLA, NVDA, MSFT, GOOGL, AMZN, META, SPY, BTC-USD, NFLX

**Weekend Warning** — Auto-detects non-trading days

**Error Handling** — Invalid tickers, missing data, network retries

---

## ML Features (18 total)

**Lag Prices** — Close price from 1 to 10 days ago

**Lag Volume** — Volume from 1, 2, and 3 days ago

**Moving Averages** — 5-day, 10-day, 20-day SMA

**Volatility** — 5-day and 10-day standard deviation of close

**Price Range** — Daily high minus low

**Model** — Random Forest Regressor (100 trees, max depth 6)

**Training** — 80% historical, validation on 20%

---

## Accuracy Guide

**R² > 0.7** — Strong fit, reasonably trustworthy

**R² 0.3 – 0.7** — Moderate, use with caution

**R² < 0.3** — Weak, directional only

---

## That's It

One file. Run it. Done.

---

## Disclaimer

For educational purposes only. Not financial advice.

---

## License

MIT
