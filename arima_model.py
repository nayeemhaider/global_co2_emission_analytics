import os
import json
import pickle
import warnings
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import mlflow
import mlflow.statsmodels

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
MLFLOW_EXPERIMENT = "CO2_ARIMA_Forecasting"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Default ARIMA orders for known series patterns
DEFAULT_ORDER = (1, 1, 1)
FORECAST_HORIZON = 20


def setup_mlflow():
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def _select_arima_order(series: pd.Series) -> tuple:
 
    try:
        import pmdarima as pm
        model = pm.auto_arima(
            series,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            d=None,
            seasonal=False,
            information_criterion="aic",
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )
        return model.order
    except Exception:
        return DEFAULT_ORDER


def _compute_risk_score(
    series: pd.Series,
    forecast_values: np.ndarray,
    trend_pct: float,
) -> dict:
    
    # Normalised level (log-scaled to dampen extremes)
    level = float(series.iloc[-1])
    level_score = min(np.log1p(level) / np.log1p(5e11) * 40, 40)

    # Trend component: positive trend = high risk
    trend_score = max(0, min(trend_pct / 5.0 * 30, 30))  # cap at 30

    # Forecast slope: rising forecast = extra penalty
    if len(forecast_values) >= 2:
        slope = (forecast_values[-1] - forecast_values[0]) / max(len(forecast_values), 1)
        slope_score = max(0, min(slope / max(level, 1e6) * 100 * 30, 30))
    else:
        slope_score = 0.0

    total = level_score + trend_score + slope_score
    risk_label = (
        "Critical" if total >= 70
        else "High" if total >= 50
        else "Medium" if total >= 30
        else "Low"
    )
    return {
        "risk_score": round(float(total), 2),
        "risk_label": risk_label,
        "level_component": round(float(level_score), 2),
        "trend_component": round(float(trend_score), 2),
        "forecast_component": round(float(slope_score), 2),
    }


def train_and_forecast(
    series: pd.Series,
    country: str,
    horizon: int = FORECAST_HORIZON,
    log_to_mlflow: bool = True,
) -> dict:
   
    series = series.dropna()
    if len(series) < 10:
        logger.warning("Series for %s too short (%d points). Skipping.", country, len(series))
        return {"country": country, "error": "insufficient data"}

    order = _select_arima_order(series)

    try:
        model = ARIMA(series, order=order)
        fitted = model.fit()

        # In-sample metrics
        residuals = fitted.resid
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        aic = float(fitted.aic)
        bic = float(fitted.bic)

        # Forecast
        fc = fitted.get_forecast(steps=horizon)
        forecast_mean = fc.predicted_mean.values
        conf_int = fc.conf_int()
        last_year = int(series.index[-1])
        forecast_years = list(range(last_year + 1, last_year + horizon + 1))

        # Trend: average YoY % change over last 10 obs
        recent = series.iloc[-10:]
        trend_pct = float(recent.pct_change().mean() * 100)

        risk = _compute_risk_score(series, forecast_mean, trend_pct)

        result = {
            "country": country,
            "order": list(order),
            "aic": aic,
            "bic": bic,
            "mae": mae,
            "rmse": rmse,
            "trend_pct_10y_avg": round(trend_pct, 4),
            "forecast_years": forecast_years,
            "forecast_values": [round(float(v), 2) for v in forecast_mean],
            "conf_int_lower": [round(float(v), 2) for v in conf_int.iloc[:, 0]],
            "conf_int_upper": [round(float(v), 2) for v in conf_int.iloc[:, 1]],
            "historical_years": [int(y) for y in series.index],
            "historical_values": [float(v) for v in series.values],
            **risk,
        }

        if log_to_mlflow:
            setup_mlflow()
            with mlflow.start_run(run_name=f"ARIMA_{country}"):
                mlflow.log_param("country", country)
                mlflow.log_param("arima_order", str(order))
                mlflow.log_param("forecast_horizon", horizon)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("aic", aic)
                mlflow.log_metric("bic", bic)
                mlflow.log_metric("risk_score", risk["risk_score"])
                mlflow.log_metric("trend_pct_10y_avg", trend_pct)

                # Save model artifact
                model_path = MODEL_DIR / f"arima_{country.replace(' ', '_')}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(fitted, f)
                mlflow.log_artifact(str(model_path))

                # Log forecast as JSON artifact (utf-8 explicit)
                fc_path = MODEL_DIR / f"forecast_{country.replace(' ', '_')}.json"
                with open(fc_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=True, indent=2)
                mlflow.log_artifact(str(fc_path))

        return result

    except Exception as exc:
        logger.error("ARIMA fitting failed for %s: %s", country, str(exc))
        return {"country": country, "error": str(exc)}


def load_forecast(country: str) -> Optional[dict]:
    
    fc_path = MODEL_DIR / f"forecast_{country.replace(' ', '_')}.json"
    if fc_path.exists():
        with open(fc_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def batch_forecast(countries: list, series_map: dict, horizon: int = FORECAST_HORIZON) -> list:
    
    results = []
    for country in countries:
        logger.info("Forecasting: %s", country)
        if country in series_map:
            result = train_and_forecast(series_map[country], country, horizon)
            results.append(result)
    return results


if __name__ == "__main__":
    from data_processing import load_data, clean_data, engineer_features, get_country_series, get_top_emitters

    df = engineer_features(clean_data(load_data()))
    top_countries = get_top_emitters(df, n=5, since_year=1950)

    print("Running ARIMA forecasts for top 5 emitters...")
    for country in top_countries:
        series = get_country_series(df, country)
        result = train_and_forecast(series, country, horizon=10)
        if "error" not in result:
            print(
                f"  {result['country']}: ARIMA{tuple(result['order'])} | "
                f"RMSE={result['rmse']:.2e} | Risk={result['risk_label']} ({result['risk_score']})"
            )
        else:
            print(f"  {result['country']}: ERROR - {result['error']}")
