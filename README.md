# Global CO2 Emission Analysis — End-to-End ML Project

Full data science portfolio project: 270 years of CO2 data (220 countries, 1750–2020),
four-layer analysis pipeline, ARIMA time-series forecasting, FastAPI serving, and MLflow tracking.

## Project Architecture

```
co2_project/
├── data/                         # Raw dataset (CO2_emission_by_countries.csv)
├── data_processing.py            # ETL: load, clean, feature engineering
├── analysis.py                   # 4-layer analysis (descriptive/diagnostic/predictive/prescriptive)
├── arima_model.py                # ARIMA forecasting + MLflow experiment tracking
├── api/
│   └── main.py                   # FastAPI REST API (8 endpoints)
├── streamlit_app/
│   └── app.py                    # Interactive Streamlit dashboard
├── models/                       # Saved ARIMA models + forecast JSON artifacts
├── mlruns/                       # MLflow SQLite database + artifacts
├── tests/
│   └── test_project.py           # 29 unit + integration tests
├── Dockerfile                    # Container for API + Streamlit
├── docker-compose.yml            # API + Streamlit + MLflow stack
├── requirements.txt
```

## Dataset

| Attribute    | Value                              |
|-------------|-------------------------------------|
| Rows         | 59,620                             |
| Countries    | 220                                |
| Year range   | 1750–2020                          |
| Key columns  | country, year, co2_tons, population, area, density |


## API Endpoints

| Method | Endpoint                       | Description                              |
|--------|--------------------------------|------------------------------------------|
| GET    | `/health`                      | Service health check                     |
| GET    | `/summary`                     | Dataset-level statistics                 |
| GET    | `/countries`                   | List all 220 countries                   |
| GET    | `/country/{country}/history`   | Annual CO2 time series for a country     |
| GET    | `/descriptive`                 | Full descriptive analysis                |
| GET    | `/diagnostic`                  | Full diagnostic analysis                 |
| GET    | `/forecast/{country}`          | ARIMA forecast for one country           |
| POST   | `/forecast/batch`              | Batch ARIMA forecasts                    |
| GET    | `/predictive`                  | Aggregated predictive + risk assessment  |
| GET    | `/prescriptive`                | Policy recommendations                   |

Interactive docs at: `http://localhost:8000/docs`

## Local Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Run preprocessing
python data_processing.py

# 3. Run tests (29 tests)
pytest tests/test_project.py -v

# 4. Run Arima model
python arima_model.py

# 5. Start FastAPI
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Start the frontend UI
co2_emission_analytics.html

# 7. Start Streamlit dashboard
streamlit run streamlit_app/app.py

# 8. View MLflow experiments
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

## Docker Deployment

```bash
# Full stack: API + Streamlit + MLflow
docker-compose up --build

# Services:
#   FastAPI:    http://localhost:8000
#   Streamlit:  http://localhost:8501
#   MLflow UI:  http://localhost:5000
```

## Streamlit Dashboard

Five interactive tabs:
- **Descriptive** — trend charts, top emitters, per-capita leaders, decade heatmaps
- **Diagnostic** — share shifts, period growth rates, volatility rankings
- **Predictive** — ARIMA forecast charts per country, risk scatter plot, aggregate projection
- **Prescriptive** — gap-to-target bar charts, country-level policy action cards
- **Country Deep-Dive** — full individual country analysis with forecast + risk breakdown

## MLflow Tracking

Every ARIMA run logs:
- **Parameters:** country, ARIMA order, forecast horizon
- **Metrics:** MAE, RMSE, AIC, BIC, risk score, 10Y trend
- **Artifacts:** pickled ARIMA model, forecast JSON
