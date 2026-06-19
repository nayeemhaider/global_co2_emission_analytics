import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# 1. DESCRIPTIVE ANALYSIS

def descriptive_analysis(df: pd.DataFrame) -> dict:
    
    latest_year = int(df["year"].max())
    earliest_year = int(df["year"].min())

    # Global annual totals
    global_annual = (
        df.groupby("year")["co2_tons"].sum()
        .reset_index()
        .rename(columns={"co2_tons": "global_co2_tons"})
    )
    global_annual = global_annual.to_dict(orient="records")

    # Top 10 total emitters (all time)
    top10_alltime = (
        df.groupby("country")["co2_tons"].sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top10_alltime.columns = ["country", "total_co2_tons"]
    top10_alltime = top10_alltime.to_dict(orient="records")

    # Top 10 emitters in the latest year
    latest_df = df[df["year"] == latest_year]
    top10_latest = (
        latest_df.nlargest(10, "co2_tons")[["country", "co2_tons"]]
        .reset_index(drop=True)
        .to_dict(orient="records")
    )

    # Decade aggregation
    decade_totals = (
        df.groupby("decade")["co2_tons"].sum()
        .reset_index()
        .to_dict(orient="records")
    )

    # Per-capita leaders (latest year, population known)
    if "co2_per_capita" in df.columns:
        per_capita = (
            latest_df.dropna(subset=["co2_per_capita"])
            .nlargest(10, "co2_per_capita")[["country", "co2_per_capita"]]
            .to_dict(orient="records")
        )
    else:
        per_capita = []

    # Summary statistics
    co2_stats = df["co2_tons"].describe().to_dict()
    co2_stats = {k: float(v) for k, v in co2_stats.items()}

    return {
        "analysis_type": "descriptive",
        "year_range": [earliest_year, latest_year],
        "total_countries": int(df["country"].nunique()),
        "global_annual_totals": global_annual,
        "top10_alltime_emitters": top10_alltime,
        "top10_latest_year_emitters": top10_latest,
        "decade_totals": decade_totals,
        "top10_per_capita_latest": per_capita,
        "co2_distribution_stats": co2_stats,
    }


# 2. DIAGNOSTIC ANALYSIS

def diagnostic_analysis(df: pd.DataFrame) -> dict:
    
    # Growth rates by country over key periods
    periods = {
        "industrial_era": (1850, 1950),
        "post_war_boom": (1950, 1980),
        "modern_era": (1990, 2020),
    }

    period_growth = {}
    for label, (y0, y1) in periods.items():
        sub = df[df["year"].between(y0, y1)]
        start_vals = sub[sub["year"] == sub["year"].min()].set_index("country")["co2_tons"]
        end_vals = sub[sub["year"] == sub["year"].max()].set_index("country")["co2_tons"]
        common = start_vals.index.intersection(end_vals.index)
        growth = ((end_vals[common] - start_vals[common]) / (start_vals[common] + 1) * 100)
        top_growers = growth.nlargest(5).reset_index()
        top_growers.columns = ["country", "growth_pct"]
        period_growth[label] = {
            "period": [y0, y1],
            "top_5_growers": top_growers.to_dict(orient="records"),
        }

    # Global acceleration: identify decades where emission growth accelerated most
    global_by_decade = df.groupby("decade")["co2_tons"].sum()
    decade_growth = global_by_decade.pct_change() * 100
    fastest_growth_decade = int(decade_growth.idxmax()) if not decade_growth.empty else None
    biggest_drop_decade = int(decade_growth.idxmin()) if not decade_growth.empty else None

    # Emission share shift: top 5 countries' share in 1950 vs 2020
    def share_in_year(yr):
        sub = df[df["year"] == yr]
        total = sub["co2_tons"].sum()
        if total == 0:
            return {}
        top5 = sub.nlargest(5, "co2_tons")[["country", "co2_tons"]].copy()
        top5["share_pct"] = (top5["co2_tons"] / total * 100).round(2)
        return top5[["country", "share_pct"]].to_dict(orient="records")

    share_1950 = share_in_year(1950)
    share_2020 = share_in_year(2020)

    # Correlation: population vs CO2 (latest year)
    latest = df[df["year"] == df["year"].max()].dropna(subset=["population_2022", "co2_tons"])
    if len(latest) > 5:
        corr_pop_co2 = float(latest[["population_2022", "co2_tons"]].corr().iloc[0, 1])
    else:
        corr_pop_co2 = None

    # Volatility: std dev of YoY pct change per country (last 30 years)
    recent = df[df["year"] >= 1990]
    volatility = (
        recent.groupby("country")["co2_yoy_pct"]
        .std()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "analysis_type": "diagnostic",
        "period_growth_rates": period_growth,
        "fastest_growth_decade": fastest_growth_decade,
        "biggest_drop_decade": biggest_drop_decade,
        "emission_share_1950": share_1950,
        "emission_share_2020": share_2020,
        "correlation_population_co2": corr_pop_co2,
        "most_volatile_countries_1990_2020": volatility,
    }


# 3. PREDICTIVE ANALYSIS

def predictive_summary(forecast_results: list) -> dict:
 
    valid = [r for r in forecast_results if "error" not in r]

    if not valid:
        return {"analysis_type": "predictive", "error": "no valid forecasts"}

    # Country-level risk table
    risk_table = [
        {
            "country": r["country"],
            "risk_score": r.get("risk_score"),
            "risk_label": r.get("risk_label"),
            "trend_pct_10y_avg": r.get("trend_pct_10y_avg"),
            "forecast_2030": r["forecast_values"][9] if len(r.get("forecast_values", [])) >= 10 else None,
        }
        for r in valid
    ]
    risk_table.sort(key=lambda x: (x["risk_score"] or 0), reverse=True)

    # Aggregate forecast: sum of forecasted emissions across all valid countries
    all_years = valid[0]["forecast_years"] if valid else []
    agg_forecast = []
    for i, yr in enumerate(all_years):
        total = sum(
            r["forecast_values"][i]
            for r in valid
            if i < len(r.get("forecast_values", []))
        )
        agg_forecast.append({"year": yr, "aggregate_co2_tons": round(total, 2)})

    critical_countries = [r["country"] for r in risk_table if r["risk_label"] == "Critical"]
    high_risk_countries = [r["country"] for r in risk_table if r["risk_label"] == "High"]

    return {
        "analysis_type": "predictive",
        "countries_analysed": len(valid),
        "forecast_horizon_years": len(all_years),
        "country_risk_table": risk_table,
        "critical_countries": critical_countries,
        "high_risk_countries": high_risk_countries,
        "aggregate_forecast": agg_forecast,
        "model_type": "ARIMA",
    }


# 4. PRESCRIPTIVE ANALYSIS

def prescriptive_analysis(
    df: pd.DataFrame,
    forecast_results: list,
    reduction_target_pct: float = 45.0,
) -> dict:
   
    valid = [r for r in forecast_results if "error" not in r]

    # Baseline: 2019 emissions per country
    baseline_year = 2019
    baseline = df[df["year"] == baseline_year].set_index("country")["co2_tons"].to_dict()

    # Target = baseline * (1 - reduction_target_pct/100)
    prescriptions = []
    for r in valid:
        country = r["country"]
        bl = baseline.get(country)
        if bl is None or bl == 0:
            continue

        target = bl * (1 - reduction_target_pct / 100)
        # Forecasted value in 2030 (index 9 for 10-year horizon from 2020)
        forecast_2030 = r["forecast_values"][9] if len(r.get("forecast_values", [])) >= 10 else None
        if forecast_2030 is None:
            continue

        gap = forecast_2030 - target  # positive = over target
        gap_pct = (gap / bl * 100) if bl > 0 else 0

        # Derive policy levers based on risk & gap
        risk_label = r.get("risk_label", "Unknown")
        trend = r.get("trend_pct_10y_avg", 0) or 0

        if gap > 0:
            urgency = "Immediate"
            actions = _policy_actions(risk_label, gap_pct, trend)
        else:
            urgency = "On Track"
            actions = ["Maintain current reduction trajectory", "Share best practices internationally"]

        prescriptions.append({
            "country": country,
            "baseline_2019_tons": round(float(bl), 2),
            "target_2030_tons": round(float(target), 2),
            "forecast_2030_tons": round(float(forecast_2030), 2),
            "gap_tons": round(float(gap), 2),
            "gap_pct_of_baseline": round(float(gap_pct), 2),
            "urgency": urgency,
            "recommended_actions": actions,
            "risk_label": risk_label,
        })

    # Sort by gap (worst first)
    prescriptions.sort(key=lambda x: x["gap_tons"], reverse=True)

    # Global aggregate
    total_forecast_2030 = sum(p["forecast_2030_tons"] for p in prescriptions)
    total_target_2030 = sum(p["target_2030_tons"] for p in prescriptions)
    global_gap = total_forecast_2030 - total_target_2030

    return {
        "analysis_type": "prescriptive",
        "reduction_target_pct": reduction_target_pct,
        "baseline_year": baseline_year,
        "target_year": 2030,
        "countries_assessed": len(prescriptions),
        "countries_over_target": sum(1 for p in prescriptions if p["urgency"] == "Immediate"),
        "countries_on_track": sum(1 for p in prescriptions if p["urgency"] == "On Track"),
        "global_forecast_2030_tons": round(total_forecast_2030, 2),
        "global_target_2030_tons": round(total_target_2030, 2),
        "global_gap_tons": round(global_gap, 2),
        "country_prescriptions": prescriptions,
    }


def _policy_actions(risk_label: str, gap_pct: float, trend: float) -> list:
  
    base = []

    if gap_pct > 50:
        base.append("Emergency decarbonisation programme required")
    elif gap_pct > 20:
        base.append("Accelerate national carbon pricing mechanism")
    else:
        base.append("Strengthen existing emissions trading scheme")

    if trend > 2:
        base.append("Impose interim annual emission caps with penalties")
    elif trend > 0:
        base.append("Set binding 5-year emission reduction milestones")
    else:
        base.append("Sustain current reduction trend with legislation")

    if risk_label in ("Critical", "High"):
        base.extend([
            "Phase out coal power by 2030",
            "Mandate EV transition for road transport",
            "Launch large-scale reforestation and carbon capture programme",
        ])
    else:
        base.extend([
            "Invest in renewable energy capacity expansion",
            "Implement building energy efficiency standards",
        ])

    return base


if __name__ == "__main__":
    from data_processing import load_data, clean_data, engineer_features

    df = engineer_features(clean_data(load_data()))
    print("Descriptive analysis keys:", list(descriptive_analysis(df).keys()))
    print("Diagnostic analysis keys:", list(diagnostic_analysis(df).keys()))
    print("Analysis modules loaded successfully.")
