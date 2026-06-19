"""
CO2 Emission Analysis: FastAPI Application
Endpoints:
  GET  /health
  GET  /summary
  GET  /descriptive
  GET  /diagnostic
  GET  /forecast/{country}
  POST /forecast/batch
  GET  /predictive
  GET  /prescriptive
  GET  /countries
  GET  /country/{country}/history
"""

import os
import json
import logging
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# local imports
from data_processing import load_data, clean_data, engineer_features, get_country_series, get_top_emitters, get_summary_stats
from analysis import descriptive_analysis, diagnostic_analysis, predictive_summary, prescriptive_analysis
from arima_model import train_and_forecast, load_forecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CO2 Emission Analysis API",
    description=(
        "End-to-end CO2 emission analysis: descriptive, diagnostic, "
        "predictive (ARIMA), and prescriptive insights with MLflow tracking."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Data cache (loaded once at startup)

@lru_cache(maxsize=1)
def _get_df():
    raw = load_data()
    clean = clean_data(raw)
    return engineer_features(clean)


@lru_cache(maxsize=1)
def _get_countries():
    df = _get_df()
    return sorted(df["country"].unique().tolist())


# Pydantic schemas

class BatchForecastRequest(BaseModel):
    countries: list[str]
    horizon: int = 10


# Endpoints

@app.get("/health")
def health():
    return {"status": "ok", "service": "CO2 Emission Analysis API"}


@app.get("/summary")
def summary():
    """High-level dataset summary."""
    df = _get_df()
    return get_summary_stats(df)


@app.get("/countries")
def list_countries():
    """Return all country names available in the dataset."""
    return {"countries": _get_countries(), "count": len(_get_countries())}


@app.get("/country/{country}/history")
def country_history(country: str, since: int = Query(1950, ge=1750, le=2020)):
    """Return annual CO2 time series for a specific country."""
    df = _get_df()
    countries = _get_countries()
    if country not in countries:
        raise HTTPException(status_code=404, detail=f"Country '{country}' not found.")
    sub = df[(df["country"] == country) & (df["year"] >= since)][
        ["year", "co2_tons", "co2_yoy_pct", "co2_rolling_10y", "co2_per_capita"]
    ].dropna(subset=["co2_tons"])
    records = sub.to_dict(orient="records")
    return {
        "country": country,
        "since": since,
        "records": records,
        "count": len(records),
    }


@app.get("/descriptive")
def descriptive():
    """Descriptive analysis: what happened?"""
    df = _get_df()
    return descriptive_analysis(df)


@app.get("/diagnostic")
def diagnostic():
    """Diagnostic analysis: why did it happen?"""
    df = _get_df()
    return diagnostic_analysis(df)


@app.get("/forecast/{country}")
def forecast_country(
    country: str,
    horizon: int = Query(10, ge=1, le=30),
    use_cache: bool = Query(True),
):
    """
    Forecast CO2 emissions for a single country using ARIMA.
    Set use_cache=false to refit even if a saved forecast exists.
    """
    df = _get_df()
    if country not in _get_countries():
        raise HTTPException(status_code=404, detail=f"Country '{country}' not found.")

    if use_cache:
        cached = load_forecast(country)
        if cached:
            logger.info("Returning cached forecast for %s", country)
            return cached

    series = get_country_series(df, country)
    result = train_and_forecast(series, country, horizon=horizon, log_to_mlflow=True)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@app.post("/forecast/batch")
def forecast_batch(request: BatchForecastRequest):
    """Forecast CO2 for multiple countries. Results logged to MLflow."""
    df = _get_df()
    valid_countries = [c for c in request.countries if c in _get_countries()]
    missing = [c for c in request.countries if c not in _get_countries()]

    results = []
    for country in valid_countries:
        cached = load_forecast(country)
        if cached:
            results.append(cached)
        else:
            series = get_country_series(df, country)
            result = train_and_forecast(series, country, horizon=request.horizon, log_to_mlflow=True)
            results.append(result)

    return {
        "requested": len(request.countries),
        "processed": len(valid_countries),
        "missing_countries": missing,
        "results": results,
    }


@app.get("/predictive")
def predictive(
    n_countries: int = Query(10, ge=1, le=50),
    since_year: int = Query(1950, ge=1750, le=2020),
):
    """
    Predictive analysis: ARIMA forecasts for top-n emitters.
    Aggregates risk assessment and trajectory projections.
    """
    df = _get_df()
    top_countries = get_top_emitters(df, n=n_countries, since_year=since_year)

    forecast_results = []
    for country in top_countries:
        cached = load_forecast(country)
        if cached:
            forecast_results.append(cached)
        else:
            series = get_country_series(df, country)
            result = train_and_forecast(series, country, horizon=10, log_to_mlflow=True)
            forecast_results.append(result)

    return predictive_summary(forecast_results)


@app.get("/prescriptive")
def prescriptive(
    n_countries: int = Query(10, ge=1, le=50),
    reduction_target_pct: float = Query(45.0, ge=0, le=100),
):
    """
    Prescriptive analysis: what reductions are needed and what actions to take?
    Based on Paris Agreement 45% reduction target by 2030.
    """
    df = _get_df()
    top_countries = get_top_emitters(df, n=n_countries, since_year=1950)

    forecast_results = []
    for country in top_countries:
        cached = load_forecast(country)
        if cached:
            forecast_results.append(cached)
        else:
            series = get_country_series(df, country)
            result = train_and_forecast(series, country, horizon=10, log_to_mlflow=True)
            forecast_results.append(result)

    return prescriptive_analysis(df, forecast_results, reduction_target_pct)


@app.on_event("startup")
async def startup_event():
    logger.info("Loading dataset...")
    _get_df()  # warm cache
    logger.info("Dataset loaded. API ready.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
