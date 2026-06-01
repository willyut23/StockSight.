# Run with: pip install -r requirements.txt && uvicorn app:app --reload

import json
import logging
import time
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Prediction MVP", version="1.0.0")

# In-memory cache: { ticker: { "data": df, "fetched_at": timestamp } }
_cache: dict = {}
CACHE_TTL_SECONDS = 3600  # 1 hour


# ─────────────────────────────────────────────
#  Pydantic models
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    ticker: str
    horizon: int = 1  # 1 or 5


# ─────────────────────────────────────────────
#  Data helpers
# ─────────────────────────────────────────────
def _get_cached_data(ticker: str) -> Optional[pd.DataFrame]:
    entry = _cache.get(ticker)
    if entry and (time.time() - entry["fetched_at"]) < CACHE_TTL_SECONDS:
        logger.info(f"Cache hit for {ticker}")
        return entry["data"]
    return None


def _fetch_data(ticker: str) -> pd.DataFrame:
    cached = _get_cached_data(ticker)
    if cached is not None:
        return cached

    end = datetime.today()
    start = end - timedelta(days=185)  # ~6 months with buffer

    last_exc = None
    for attempt in range(2):
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if df.empty:
                raise ValueError(f"No data returned for ticker '{ticker}'. It may be invalid or delisted.")
            break
        except ValueError:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                logger.warning(f"Fetch attempt 1 failed for {ticker}: {exc}. Retrying…")
                time.sleep(1)
            else:
                raise RuntimeError(
                    f"Network error fetching data for '{ticker}' after 2 attempts: {exc}"
                ) from exc

    # Normalise columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Drop rows with all NaN, then interpolate remaining
    df.dropna(how="all", inplace=True)
    df.interpolate(method="time", inplace=True)
    df.dropna(inplace=True)  # drop any residual leading NaNs

    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} trading days available for '{ticker}' — need at least 30. "
            "Try a more liquid ticker."
        )

    _cache[ticker] = {"data": df, "fetched_at": time.time()}
    logger.info(f"Fetched {len(df)} rows for {ticker}")
    return df


# ─────────────────────────────────────────────
#  Feature engineering
# ─────────────────────────────────────────────
LAG_DAYS = 10


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()

    # Lagged close prices
    for i in range(1, LAG_DAYS + 1):
        feat[f"close_lag_{i}"] = feat["Close"].shift(i)

    # Lagged volume
    for i in range(1, 4):
        feat[f"vol_lag_{i}"] = feat["Volume"].shift(i)

    # Rolling averages
    feat["ma5"] = feat["Close"].rolling(5).mean()
    feat["ma10"] = feat["Close"].rolling(10).mean()
    feat["ma20"] = feat["Close"].rolling(20).mean()

    # Rolling std (volatility proxy)
    feat["std5"] = feat["Close"].rolling(5).std()
    feat["std10"] = feat["Close"].rolling(10).std()

    # Price range
    feat["hl_range"] = feat["High"] - feat["Low"]

    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.dropna(inplace=True)

    return feat


def _feature_columns(feat: pd.DataFrame) -> list:
    exclude = {"Open", "High", "Low", "Close", "Volume"}
    return [c for c in feat.columns if c not in exclude]


# ─────────────────────────────────────────────
#  Model training & prediction
# ─────────────────────────────────────────────
def _train_and_predict(df: pd.DataFrame, horizon: int) -> dict:
    feat = _build_features(df)
    cols = _feature_columns(feat)

    X = feat[cols].values
    y = feat["Close"].values

    # Train / val split (80/20)
    split = max(int(len(X) * 0.8), len(X) - 30)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    val_preds = model.predict(X_val_s)
    r2 = float(r2_score(y_val, val_preds))
    r2 = max(-1.0, min(1.0, r2))  # clamp

    # Iterative future prediction
    # We'll extend the feature row day-by-day
    last_row = feat.iloc[-1].copy()
    close_history = list(feat["Close"].values)
    volume_history = list(feat["Volume"].values)
    high_history = list(feat["High"].values)
    low_history = list(feat["Low"].values)

    future_prices = []
    future_dates = []
    last_date = feat.index[-1]

    for step in range(horizon):
        next_date = last_date + timedelta(days=1)
        # Skip weekends (simple approximation — not calendar-aware)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

        # Build synthetic next row
        new_close_lags = [close_history[-(i)] for i in range(1, LAG_DAYS + 1)]
        new_vol_lags = [volume_history[-(i)] for i in range(1, 4)]

        close_arr = np.array(close_history)
        new_ma5 = float(np.mean(close_arr[-5:])) if len(close_arr) >= 5 else float(np.mean(close_arr))
        new_ma10 = float(np.mean(close_arr[-10:])) if len(close_arr) >= 10 else float(np.mean(close_arr))
        new_ma20 = float(np.mean(close_arr[-20:])) if len(close_arr) >= 20 else float(np.mean(close_arr))
        new_std5 = float(np.std(close_arr[-5:])) if len(close_arr) >= 5 else 0.0
        new_std10 = float(np.std(close_arr[-10:])) if len(close_arr) >= 10 else 0.0
        new_hl_range = float(np.mean(np.array(high_history[-5:]) - np.array(low_history[-5:])))

        row = (
            new_close_lags
            + new_vol_lags
            + [new_ma5, new_ma10, new_ma20, new_std5, new_std10, new_hl_range]
        )
        row_arr = np.array(row, dtype=float)

        # Sanity check
        if np.any(np.isnan(row_arr)) or np.any(np.isinf(row_arr)):
            row_arr = np.nan_to_num(row_arr, nan=close_history[-1], posinf=close_history[-1], neginf=close_history[-1])

        row_s = scaler.transform(row_arr.reshape(1, -1))
        pred_price = float(model.predict(row_s)[0])
        pred_price = max(0.01, pred_price)  # floor at 1 cent

        future_prices.append(round(pred_price, 4))
        future_dates.append(next_date.strftime("%Y-%m-%d"))

        # Append to history for next iteration
        close_history.append(pred_price)
        volume_history.append(volume_history[-1])  # carry last volume
        high_history.append(pred_price * 1.005)
        low_history.append(pred_price * 0.995)
        last_date = next_date

    on_weekend = bool(last_date.weekday() >= 5)

    return {
        "r2": round(r2, 4),
        "future_dates": future_dates,
        "future_prices": future_prices,
        "on_weekend_warning": on_weekend,
    }


# ─────────────────────────────────────────────
#  API routes
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "cached_tickers": list(_cache.keys())}


@app.post("/predict")
def predict(req: PredictRequest):
    # Sanitize input
    ticker = req.ticker.strip().upper()
    if not ticker or not ticker.replace(".", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid ticker symbol.")
    if req.horizon not in (1, 5):
        raise HTTPException(status_code=400, detail="Horizon must be 1 or 5.")

    try:
        df = _fetch_data(ticker)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Pre-flight: ensure enough rows survive feature engineering
    # (need LAG_DAYS + rolling-window + train/val split)
    if len(df) < 30:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(df)} trading days available for '{ticker}' — need at least 30. "
                   "Try a more liquid ticker or a longer time range.",
        )

    try:
        result = _train_and_predict(df, req.horizon)
    except Exception as e:
        logger.exception(f"ML error for {ticker}")
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # Build chart data
    hist_dates = df.index.strftime("%Y-%m-%d").tolist()
    hist_prices = [round(float(p), 4) for p in df["Close"].tolist()]

    is_cached = (time.time() - _cache[ticker]["fetched_at"]) < CACHE_TTL_SECONDS

    return JSONResponse(
        content={
            "ticker": ticker,
            "horizon": req.horizon,
            "r2": result["r2"],
            "hist_dates": hist_dates,
            "hist_prices": hist_prices,
            "future_dates": result["future_dates"],
            "future_prices": result["future_prices"],
            "weekend_warning": result["on_weekend_warning"],
            "cached": is_cached,
            "data_points": len(df),
        }
    )


# ─────────────────────────────────────────────
#  Embedded frontend (single-file HTML)
# ─────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>StockSight — Price Forecasting</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: #f7f3ee;
    --surface: #faf7f3;
    --surface2: #f0ebe3;
    --border: #e0d8ce;
    --accent: #8b6f47;
    --accent-light: #c4a882;
    --text: #2c2218;
    --muted: #8a7a6a;
    --green: #4a7c59;
    --red: #9b3a2f;
    --font-display: 'Lora', serif;
    --font-body: 'DM Sans', sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    min-height: 100vh;
  }

  .wrapper {
    max-width: 860px;
    margin: 0 auto;
    padding: 3rem 1.5rem;
  }

  /* Header */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }
  header h1 {
    font-family: var(--font-display);
    font-size: 1.75rem;
    font-weight: 500;
    color: var(--text);
    letter-spacing: -0.01em;
  }
  header h1 em {
    font-style: italic;
    color: var(--accent);
  }
  .tagline {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.15rem;
  }
  .health-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--border);
    transition: background 0.4s;
  }
  .health-dot.ok { background: var(--green); }

  /* Quick picks */
  .quick-picks {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 1.5rem;
    align-items: center;
  }
  .qp-label {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-right: 0.25rem;
  }
  .quick-pick {
    padding: 0.2rem 0.6rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 3px;
    font-size: 0.73rem;
    font-family: var(--font-body);
    color: var(--muted);
    cursor: pointer;
    transition: all 0.15s;
  }
  .quick-pick:hover {
    background: var(--surface2);
    color: var(--accent);
    border-color: var(--accent-light);
  }

  /* Controls */
  .controls {
    display: grid;
    grid-template-columns: 1fr 160px auto;
    gap: 0.75rem;
    align-items: end;
    margin-bottom: 1.75rem;
  }
  @media (max-width: 580px) {
    .controls { grid-template-columns: 1fr 1fr; }
    .controls .predict-wrap { grid-column: 1 / -1; }
  }

  label {
    display: block;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 0.35rem;
  }

  .ticker-wrap { position: relative; }
  .ticker-wrap::before {
    content: '$';
    position: absolute;
    left: 11px; top: 50%;
    transform: translateY(-50%);
    color: var(--accent-light);
    font-size: 0.9rem;
    pointer-events: none;
    font-family: var(--font-display);
  }

  input[type="text"] {
    width: 100%;
    padding: 0.6rem 0.9rem 0.6rem 1.75rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text);
    font-family: var(--font-body);
    font-size: 0.95rem;
    font-weight: 500;
    outline: none;
    transition: border-color 0.2s;
  }
  input[type="text"]:focus { border-color: var(--accent-light); }

  select {
    width: 100%;
    padding: 0.6rem 0.85rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text);
    font-family: var(--font-body);
    font-size: 0.88rem;
    cursor: pointer;
    outline: none;
    appearance: none;
    -webkit-appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%238a7a6a' fill='none' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
    padding-right: 1.8rem;
  }

  .predict-wrap button {
    width: 100%;
    padding: 0.6rem 1.5rem;
    background: var(--accent);
    color: #faf7f3;
    border: none;
    border-radius: 5px;
    font-family: var(--font-display);
    font-size: 0.88rem;
    font-style: italic;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
    white-space: nowrap;
  }
  .predict-wrap button:hover { background: #7a5f38; }
  .predict-wrap button:active { transform: scale(0.98); }
  .predict-wrap button:disabled {
    background: var(--border);
    color: var(--muted);
    cursor: not-allowed;
  }

  /* Banners */
  .banner {
    padding: 0.75rem 1rem;
    border-radius: 5px;
    font-size: 0.82rem;
    margin-bottom: 1.25rem;
    border-left: 3px solid;
    display: none;
    line-height: 1.5;
  }
  .banner.visible { display: block; }
  .banner.error { background: #fdf0ee; border-color: #c87a6a; color: #7a3a2a; }
  .banner.warning { background: #fdf5e8; border-color: #c4a050; color: #7a5a18; }
  .banner.info { background: #f0ede8; border-color: var(--accent-light); color: var(--accent); }

  /* Loader */
  .loader {
    display: none;
    align-items: center;
    gap: 0.75rem;
    padding: 1.5rem 0;
    color: var(--muted);
    font-size: 0.82rem;
    font-style: italic;
    font-family: var(--font-display);
  }
  .loader.visible { display: flex; }
  .spinner {
    width: 16px; height: 16px;
    border: 1.5px solid var(--border);
    border-top-color: var(--accent-light);
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
    flex-shrink: 0;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Results */
  #results { display: none; }
  #results.visible { display: block; animation: fadeUp 0.35s ease; }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* Meta bar */
  .meta-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 1.25rem;
    align-items: center;
  }
  .pill {
    padding: 0.2rem 0.65rem;
    border-radius: 3px;
    font-size: 0.72rem;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--muted);
  }
  .pill.ticker {
    font-family: var(--font-display);
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--accent);
    border-color: var(--accent-light);
    background: #f5efe7;
    padding: 0.2rem 0.8rem;
  }
  .pill.good { color: var(--green); border-color: #a0c4a8; background: #f0f7f0; }
  .pill.bad  { color: var(--red);   border-color: #d4a0a0; background: #fdf0ee; }
  .pill.cached { color: var(--accent); border-color: var(--accent-light); }

  /* Chart */
  #chart {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 1.25rem;
    overflow: hidden;
  }

  /* Table */
  .pred-table-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 1.25rem;
  }
  .section-title {
    font-family: var(--font-display);
    font-size: 0.72rem;
    font-weight: 400;
    font-style: italic;
    color: var(--muted);
    padding: 0.8rem 1.1rem 0.55rem;
    border-bottom: 1px solid var(--border);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
  }
  th {
    text-align: left;
    padding: 0.45rem 1.1rem;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    background: var(--surface2);
    font-weight: 400;
  }
  td {
    padding: 0.55rem 1.1rem;
    border-top: 1px solid var(--border);
    color: var(--text);
  }
  td.price { font-weight: 500; color: var(--accent); }
  td.change.up   { color: var(--green); }
  td.change.down { color: var(--red); }
  tr:hover td { background: var(--surface2); }

  /* R² */
  .confidence-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
  }
  .r2-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 0.6rem;
  }
  .r2-bar-bg {
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 100px;
    overflow: hidden;
  }
  .r2-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.8s cubic-bezier(0.25, 1, 0.5, 1);
  }
  .r2-value {
    font-weight: 500;
    font-size: 0.9rem;
    min-width: 3.2rem;
    text-align: right;
    font-family: var(--font-display);
  }
  .r2-label {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.4rem;
    font-style: italic;
    font-family: var(--font-display);
  }

  footer {
    margin-top: 2.5rem;
    padding-top: 1.25rem;
    border-top: 1px solid var(--border);
    font-size: 0.7rem;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.4rem;
  }
</style>
</head>
<body>
<div class="wrapper">
  <header>
    <div>
      <h1>Stock<em>Sight</em></h1>
      <div class="tagline">ML-powered price forecasting</div>
    </div>
    <div class="health-dot" id="healthDot" title="API status"></div>
  </header>

  <div class="quick-picks" id="quickPicks">
    <span class="qp-label">Quick picks</span>
  </div>

  <div class="controls">
    <div>
      <label for="tickerInput">Ticker</label>
      <div class="ticker-wrap">
        <input type="text" id="tickerInput" placeholder="AAPL" maxlength="10" autocomplete="off" spellcheck="false"/>
      </div>
    </div>
    <div>
      <label for="horizonSelect">Horizon</label>
      <select id="horizonSelect">
        <option value="1">1 Day</option>
        <option value="5">5 Days</option>
      </select>
    </div>
    <div class="predict-wrap">
      <label>&nbsp;</label>
      <button id="predictBtn" onclick="runPrediction()">Forecast →</button>
    </div>
  </div>

  <div class="banner error" id="errorBanner"></div>
  <div class="banner warning" id="warningBanner"></div>
  <div class="banner info" id="infoBanner"></div>

  <div class="loader" id="loader">
    <div class="spinner"></div>
    <span id="loaderText">Fetching market data…</span>
  </div>

  <div id="results">
    <div class="meta-bar" id="metaBar"></div>
    <div id="chart"></div>
    <div class="pred-table-wrap">
      <div class="section-title">Predicted closing prices</div>
      <table id="predTable"></table>
    </div>
    <div class="confidence-section">
      <div class="section-title" style="padding:0;border:none;">Model confidence — R² on validation set</div>
      <div class="r2-row">
        <div class="r2-bar-bg"><div class="r2-bar-fill" id="r2Fill" style="width:0%"></div></div>
        <div class="r2-value" id="r2Value">—</div>
      </div>
      <div class="r2-label" id="r2Label"></div>
    </div>
  </div>

  <footer>
    <span>StockSight v1.0 · Random Forest · yfinance</span>
    <span>For educational use only — not financial advice.</span>
  </footer>
</div>

<script>
const TICKERS = ["AAPL","MSFT","TSLA","GOOGL","AMZN","META","NVDA","SPY","BTC-USD","NFLX"];

const qp = document.getElementById('quickPicks');
TICKERS.forEach(t => {
  const btn = document.createElement('button');
  btn.className = 'quick-pick';
  btn.textContent = t;
  btn.onclick = () => { document.getElementById('tickerInput').value = t; runPrediction(); };
  qp.appendChild(btn);
});

async function checkHealth() {
  try {
    const r = await fetch('/health');
    if (r.ok) document.getElementById('healthDot').className = 'health-dot ok';
  } catch {}
}
checkHealth();

document.getElementById('tickerInput').addEventListener('input', function() {
  const pos = this.selectionStart;
  this.value = this.value.toUpperCase().replace(/[^A-Z0-9.\-]/g,'');
  this.setSelectionRange(pos, pos);
});
document.getElementById('tickerInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') runPrediction();
});

function show(id) { document.getElementById(id).classList.add('visible'); }
function hide(id) { document.getElementById(id).classList.remove('visible'); }
function setBanner(id, msg) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.classList.add('visible');
}
function clearBanners() {
  ['errorBanner','warningBanner','infoBanner'].forEach(id => {
    document.getElementById(id).classList.remove('visible');
    document.getElementById(id).textContent = '';
  });
}

async function runPrediction() {
  const ticker = document.getElementById('tickerInput').value.trim().toUpperCase();
  const horizon = parseInt(document.getElementById('horizonSelect').value);

  if (!ticker) {
    setBanner('errorBanner', 'Please enter a ticker symbol (e.g. AAPL, TSLA, MSFT).');
    return;
  }

  clearBanners();
  hide('results');

  const btn = document.getElementById('predictBtn');
  btn.disabled = true;
  const loader = document.getElementById('loader');
  loader.classList.add('visible');
  document.getElementById('loaderText').textContent = `Fetching data for ${ticker}…`;

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker, horizon }),
    });

    const data = await resp.json();

    if (!resp.ok) {
      setBanner('errorBanner', `✖ ${data.detail || 'Unknown error. Please try again.'}`);
      return;
    }

    loader.classList.remove('visible');
    renderResults(data);

  } catch (err) {
    setBanner('errorBanner', `✖ Network error: ${err.message}. Is the server running?`);
  } finally {
    btn.disabled = false;
    loader.classList.remove('visible');
  }
}

function renderResults(data) {
  clearBanners();

  if (data.weekend_warning) {
    setBanner('warningBanner', '⚠ Prediction window includes weekend / non-trading days — prices are extrapolated, not market-open forecasts.');
  }
  if (data.cached) {
    setBanner('infoBanner', `ℹ Using cached data for ${data.ticker} (refreshes hourly). ${data.data_points} trading days loaded.`);
  }

  // Meta pills
  const r2Class = data.r2 >= 0.7 ? 'good' : data.r2 >= 0.3 ? 'neutral' : 'bad';
  const r2Label = data.r2 >= 0.7 ? 'Strong fit' : data.r2 >= 0.3 ? 'Moderate fit' : 'Weak fit';
  document.getElementById('metaBar').innerHTML = `
    <span class="pill ticker">${data.ticker}</span>
    <span class="pill neutral">${data.data_points} trading days</span>
    <span class="pill neutral">${data.horizon}-day horizon</span>
    <span class="pill ${r2Class}">R² ${data.r2} — ${r2Label}</span>
    ${data.cached ? '<span class="pill cached">⚡ cached</span>' : ''}
  `;

  // Chart
  const lastHistDate = data.hist_dates[data.hist_dates.length - 1];
  const bridgeDates = [lastHistDate, ...data.future_dates];
  const bridgePrices = [data.hist_prices[data.hist_prices.length - 1], ...data.future_prices];

  const traceHist = {
    x: data.hist_dates,
    y: data.hist_prices,
    type: 'scatter',
    mode: 'lines',
    name: 'Historical',
    line: { color: '#4da6ff', width: 1.5 },
    hovertemplate: '<b>%{x}</b><br>$%{y:.2f}<extra>Historical</extra>',
  };

  const traceFuture = {
    x: bridgeDates,
    y: bridgePrices,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Forecast',
    line: { color: '#00e5a0', width: 2, dash: 'dot' },
    marker: { size: 7, color: '#00e5a0', line: { color: '#020f08', width: 1.5 } },
    hovertemplate: '<b>%{x}</b><br>$%{y:.2f}<extra>Forecast</extra>',
  };

  const layout = {
    paper_bgcolor: '#0e1318',
    plot_bgcolor: '#0e1318',
    font: { family: 'DM Mono, monospace', color: '#5a7a8a', size: 11 },
    margin: { l: 55, r: 20, t: 20, b: 45 },
    xaxis: {
      gridcolor: '#1e2a35', zeroline: false,
      tickfont: { size: 10 },
    },
    yaxis: {
      gridcolor: '#1e2a35', zeroline: false,
      tickfont: { size: 10 },
      tickprefix: '$',
    },
    legend: {
      bgcolor: 'rgba(14,19,24,0.8)',
      bordercolor: '#1e2a35',
      borderwidth: 1,
      font: { size: 11 },
    },
    shapes: [{
      type: 'line',
      x0: lastHistDate, x1: lastHistDate,
      y0: 0, y1: 1, yref: 'paper',
      line: { color: 'rgba(255,255,255,0.12)', width: 1, dash: 'dash' },
    }],
    annotations: [{
      x: lastHistDate, y: 1, yref: 'paper',
      text: 'Today', showarrow: false,
      font: { size: 9, color: 'rgba(255,255,255,0.3)' },
      xanchor: 'left', xshift: 5, yshift: -2,
    }],
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: '#141b22',
      bordercolor: '#1e2a35',
      font: { family: 'DM Mono, monospace', size: 11 },
    },
  };

  Plotly.newPlot('chart', [traceHist, traceFuture], layout, {
    responsive: true,
    displayModeBar: false,
  });

  // Prediction table
  const lastClose = data.hist_prices[data.hist_prices.length - 1];
  let tableHTML = `<thead><tr>
    <th>#</th><th>Date</th><th>Predicted Close</th><th>Change vs Last</th>
  </tr></thead><tbody>`;

  data.future_prices.forEach((price, i) => {
    const delta = price - lastClose;
    const pct = ((delta / lastClose) * 100).toFixed(2);
    const sign = delta >= 0 ? '+' : '';
    const cls = delta >= 0 ? 'up' : 'down';
    tableHTML += `<tr>
      <td style="color:var(--muted)">${i + 1}</td>
      <td>${data.future_dates[i]}</td>
      <td class="price">$${price.toFixed(2)}</td>
      <td class="change ${cls}">${sign}$${Math.abs(delta).toFixed(2)} (${sign}${pct}%)</td>
    </tr>`;
  });
  tableHTML += '</tbody>';
  document.getElementById('predTable').innerHTML = tableHTML;

  // R² bar
  const pct = Math.max(0, Math.min(100, data.r2 * 100));
  const fill = document.getElementById('r2Fill');
  const color = data.r2 >= 0.7 ? '#00e5a0' : data.r2 >= 0.3 ? '#4da6ff' : '#ff4d6a';
  fill.style.width = pct + '%';
  fill.style.background = color;
  document.getElementById('r2Value').textContent = data.r2.toFixed(3);
  document.getElementById('r2Value').style.color = color;
  document.getElementById('r2Label').textContent =
    data.r2 >= 0.7 ? 'Model explains most variance in the validation set — reasonably trustworthy signal.'
    : data.r2 >= 0.3 ? 'Moderate explanatory power — treat predictions with appropriate caution.'
    : 'Low explanatory power — market may be highly volatile. Use as loose directional guidance only.';

  show('results');
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML)
