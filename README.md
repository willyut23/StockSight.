# StockSight

**ML-powered stock price forecasting**

![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?style=flat-square&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## What It Does

Fetches live stock data from Yahoo Finance, trains a Random Forest model on 20+ features, and predicts closing prices 1 or 5 trading days ahead. All in a single Python file with an interactive web interface.

---

## One File. Zero Bugs. Ready to Run.

Everything is in **`app.py`** — FastAPI backend, ML pipeline, and Plotly frontend.

```bash
# Install
pip install fastapi uvicorn yfinance pandas numpy scikit-learn plotly

# Run
uvicorn app:app --reload

# Open
http://localhost:8000
No API keys. No database. No build step.

Features
Feature	Description
1-Day Forecast	Predicts next trading day's closing price
5-Day Forecast	Predicts next 5 trading days
Interactive Chart	Historical prices + forecast with Plotly
R² Confidence Score	Model reliability on validation set (0–1 scale)
Smart Caching	1-hour TTL — no redundant API calls
10+ Quick Picks	AAPL, TSLA, NVDA, MSFT, GOOGL, AMZN, META, SPY, BTC-USD, NFLX
Weekend Warning	Auto-detects non-trading days and warns user
Error Handling	Invalid tickers, missing data, network retries — all covered
ML Features Used
The model uses 18 features from 6 months of daily data:

Category	Features
Lag Prices	Close price from 1 to 10 days ago
Lag Volume	Volume from 1, 2, and 3 days ago
Moving Averages	5-day, 10-day, 20-day SMA
Volatility	5-day and 10-day standard deviation of close
Price Range	Daily high - low
Model: Random Forest Regressor (100 trees, max depth 6)

How Accurate Is It?
R² Range	Interpretation
> 0.7	Strong fit — reasonably trustworthy
0.3 – 0.7	Moderate — use with caution
< 0.3	Weak — directional guidance only
The model is trained on 80% of data and validated on the remaining 20%.

Project Structure
text
stocksight/
├── app.py              # Single file — backend + frontend + ML
├── requirements.txt    # Dependencies (optional)
└── README.md           # This file
app.py contains:

FastAPI routes (/, /health, /predict)

Data fetching with yfinance + caching

Feature engineering (18 columns)

Random Forest training + iterative forecasting

Embedded HTML/CSS/JS frontend with Plotly

Error Handling (Zero-Bug Design)
Scenario	Handling
Invalid ticker	Clear error message, no crash
Less than 30 days of data	Graceful fail with explanation
Network failure	1 retry, then user-friendly error
NaN / infinite values	Auto-interpolated or dropped
Weekend prediction	Warning label + extrapolation
Rate limiting	Retry logic built-in
Requirements
Python 3.8+

Internet connection (for Yahoo Finance)

No API keys, no registration

Dependencies:

text
fastapi>=0.95.0
uvicorn>=0.21.0
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.14.0
Screenshots
(Run the app and see for yourself)

Clean, minimalist interface

Dark-themed chart with historical + forecast lines

Prediction table with daily changes

R² confidence bar with color coding

Limitations
Not a trading signal — educational only

Simple model (no LSTM, no transformers)

No fundamental data (earnings, news, sentiment)

Weekend forecasts are extrapolated, not market-open predictions

Disclaimer
For educational purposes only. Past performance does not guarantee future results. Do not use this for real trading or investment decisions. The author assumes no liability for financial losses.

License
MIT — free to use, modify, and distribute.

Author
Built with FastAPI, scikit-learn, and yfinance. One file, zero complexity.
