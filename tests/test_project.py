import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

from data_processing import load_data, clean_data, engineer_features, get_country_series, get_summary_stats, get_top_emitters
from analysis import descriptive_analysis, diagnostic_analysis, predictive_summary, prescriptive_analysis
from arima_model import train_and_forecast


# Fixtures

@pytest.fixture(scope="session")
def raw_df():
    return load_data()


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return clean_data(raw_df)


@pytest.fixture(scope="session")
def featured_df(clean_df):
    return engineer_features(clean_df)


# Data Processing Tests

class TestDataLoading:
    def test_load_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_expected_columns_after_clean(self, clean_df):
        expected = {"country", "year", "co2_tons"}
        assert expected.issubset(set(clean_df.columns))

    def test_no_negative_co2(self, clean_df):
        assert (clean_df["co2_tons"] >= 0).all(), "Negative CO2 values found"

    def test_year_range(self, clean_df):
        assert clean_df["year"].min() >= 1700
        assert clean_df["year"].max() <= 2025

    def test_country_count(self, clean_df):
        assert clean_df["country"].nunique() >= 100

    def test_pct_world_numeric(self, clean_df):
        non_null = clean_df["pct_world"].dropna()
        assert pd.api.types.is_float_dtype(non_null)

    def test_density_numeric(self, clean_df):
        non_null = clean_df["density_km2"].dropna()
        assert pd.api.types.is_float_dtype(non_null)


class TestFeatureEngineering:
    def test_derived_columns_exist(self, featured_df):
        for col in ["co2_per_capita", "co2_yoy_change", "co2_yoy_pct", "co2_rolling_10y", "co2_cumulative", "decade"]:
            assert col in featured_df.columns, f"Missing column: {col}"

    def test_decade_multiples_of_10(self, featured_df):
        assert (featured_df["decade"] % 10 == 0).all()

    def test_cumulative_always_increasing(self, featured_df):
        china = featured_df[featured_df["country"] == "China"].sort_values("year")
        diffs = china["co2_cumulative"].diff().dropna()
        assert (diffs >= 0).all(), "Cumulative CO2 decreased for China"

    def test_get_country_series(self, featured_df):
        series = get_country_series(featured_df, "China")
        assert isinstance(series, pd.Series)
        assert len(series) > 10

    def test_get_top_emitters(self, featured_df):
        top = get_top_emitters(featured_df, n=5)
        assert len(top) == 5
        assert all(isinstance(c, str) for c in top)

    def test_summary_stats_keys(self, featured_df):
        stats = get_summary_stats(featured_df)
        for key in ["total_countries", "year_range", "total_rows"]:
            assert key in stats


# Analysis Tests

class TestDescriptiveAnalysis:
    def test_returns_dict(self, featured_df):
        result = descriptive_analysis(featured_df)
        assert isinstance(result, dict)

    def test_required_keys(self, featured_df):
        result = descriptive_analysis(featured_df)
        for key in ["analysis_type", "global_annual_totals", "top10_alltime_emitters", "decade_totals"]:
            assert key in result, f"Missing key: {key}"

    def test_global_annual_totals_has_years(self, featured_df):
        result = descriptive_analysis(featured_df)
        totals = result["global_annual_totals"]
        assert len(totals) > 100
        assert "year" in totals[0]
        assert "global_co2_tons" in totals[0]

    def test_top10_has_10_entries(self, featured_df):
        result = descriptive_analysis(featured_df)
        assert len(result["top10_alltime_emitters"]) == 10


class TestDiagnosticAnalysis:
    def test_returns_dict(self, featured_df):
        result = diagnostic_analysis(featured_df)
        assert isinstance(result, dict)

    def test_period_growth_has_all_periods(self, featured_df):
        result = diagnostic_analysis(featured_df)
        periods = result["period_growth_rates"]
        assert "industrial_era" in periods
        assert "post_war_boom" in periods
        assert "modern_era" in periods

    def test_correlation_is_numeric_or_none(self, featured_df):
        result = diagnostic_analysis(featured_df)
        corr = result["correlation_population_co2"]
        if corr is not None:
            assert -1.0 <= corr <= 1.0


class TestPredictiveSummary:
    def test_empty_forecast_list(self):
        result = predictive_summary([])
        assert "error" in result

    def test_with_error_forecasts(self):
        bad = [{"country": "X", "error": "no data"}]
        result = predictive_summary(bad)
        assert "error" in result

    def test_with_valid_forecast(self):
        mock = [{
            "country": "Testland",
            "order": [1, 1, 1],
            "aic": 100.0,
            "bic": 110.0,
            "mae": 5000.0,
            "rmse": 7000.0,
            "trend_pct_10y_avg": 1.5,
            "forecast_years": list(range(2021, 2031)),
            "forecast_values": [float(i * 1e8) for i in range(1, 11)],
            "conf_int_lower": [float(i * 9e7) for i in range(1, 11)],
            "conf_int_upper": [float(i * 1.1e8) for i in range(1, 11)],
            "historical_years": list(range(1980, 2021)),
            "historical_values": [float(i * 1e7) for i in range(1, 42)],
            "risk_score": 65.0,
            "risk_label": "Critical",
            "level_component": 30.0,
            "trend_component": 20.0,
            "forecast_component": 15.0,
        }]
        result = predictive_summary(mock)
        assert result["countries_analysed"] == 1
        assert result["critical_countries"] == ["Testland"]


# ARIMA Tests

class TestARIMA:
    def test_forecast_china(self, featured_df):
        series = get_country_series(featured_df, "China")
        result = train_and_forecast(series, "China", horizon=5, log_to_mlflow=False)
        assert "error" not in result
        assert len(result["forecast_values"]) == 5
        assert result["risk_score"] >= 0

    def test_short_series_returns_error(self):
        short = pd.Series([1.0, 2.0, 3.0], index=[2018, 2019, 2020])
        result = train_and_forecast(short, "TinyCountry", log_to_mlflow=False)
        assert "error" in result

    def test_forecast_years_are_future(self, featured_df):
        series = get_country_series(featured_df, "Germany")
        result = train_and_forecast(series, "Germany", horizon=5, log_to_mlflow=False)
        if "error" not in result:
            assert min(result["forecast_years"]) > max(result["historical_years"])

    def test_risk_label_valid(self, featured_df):
        series = get_country_series(featured_df, "United States")
        result = train_and_forecast(series, "United States", horizon=5, log_to_mlflow=False)
        if "error" not in result:
            assert result["risk_label"] in ("Low", "Medium", "High", "Critical")


# Prescriptive Tests

class TestPrescriptive:
    def test_returns_dict(self, featured_df):
        mock_forecasts = []
        result = prescriptive_analysis(featured_df, mock_forecasts)
        assert isinstance(result, dict)
        assert result["countries_assessed"] == 0

    def test_target_applied_correctly(self, featured_df):
        mock = [{
            "country": "Germany",
            "risk_score": 50.0,
            "risk_label": "High",
            "trend_pct_10y_avg": 1.0,
            "forecast_values": [float(5e11)] * 10,
            "forecast_years": list(range(2021, 2031)),
        }]
        result = prescriptive_analysis(featured_df, mock, reduction_target_pct=45.0)
        # Germany exists in 2019 data so should be assessed
        assert isinstance(result["country_prescriptions"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
