import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/CO2_emission_by_countries.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:

    df = pd.read_csv(path, encoding="latin-1")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    # Rename columns to safe snake_case (no special chars)
    df.columns = [
        "country", "code", "calling_code", "year",
        "co2_tons", "population_2022", "area_km2",
        "pct_world", "density_km2"
    ]

    # Parse pct_world: "0.40%" -> 0.40
    df["pct_world"] = (
        df["pct_world"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace("nan", np.nan)
        .astype(float)
    )

    # Parse density: "1,924/km2" -> 1924  (strip /km suffix, remove commas)
    df["density_km2"] = (
        df["density_km2"]
        .astype(str)
        .str.replace(r"/km.*", "", regex=True)   # strip /km² suffix
        .str.replace(",", "", regex=False)        # remove thousands separator
        .str.strip()
        .replace("nan", np.nan)
        .apply(lambda v: float(v) if v not in ("nan", "", "None") else np.nan)
    )

    # co2_tons already numeric; ensure float
    df["co2_tons"] = pd.to_numeric(df["co2_tons"], errors="coerce").fillna(0.0)

    # Sort for time series work
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Per-capita CO2 (tons / person) where population known
    df["co2_per_capita"] = np.where(
        df["population_2022"] > 0,
        df["co2_tons"] / df["population_2022"],
        np.nan,
    )

    # Year-over-year absolute change per country
    df["co2_yoy_change"] = df.groupby("country")["co2_tons"].diff()

    # Year-over-year percentage change
    df["co2_yoy_pct"] = df.groupby("country")["co2_tons"].pct_change() * 100

    # 10-year rolling average (centred)
    df["co2_rolling_10y"] = (
        df.groupby("country")["co2_tons"]
        .transform(lambda s: s.rolling(10, min_periods=3).mean())
    )

    # Cumulative CO2 per country
    df["co2_cumulative"] = df.groupby("country")["co2_tons"].cumsum()

    # Decade label
    df["decade"] = (df["year"] // 10) * 10

    return df


def get_country_series(df: pd.DataFrame, country: str) -> pd.Series:

    sub = df[df["country"] == country].set_index("year")["co2_tons"]
    return sub.sort_index()


def get_top_emitters(df: pd.DataFrame, n: int = 10, since_year: int = 1950) -> list:

    subset = df[df["year"] >= since_year]
    totals = subset.groupby("country")["co2_tons"].sum().sort_values(ascending=False)
    return totals.head(n).index.tolist()


def get_summary_stats(df: pd.DataFrame) -> dict:
    
    recent = df[df["year"] == df["year"].max()]
    return {
        "total_countries": int(df["country"].nunique()),
        "year_range": (int(df["year"].min()), int(df["year"].max())),
        "total_rows": int(len(df)),
        "global_co2_latest_year": int(df["year"].max()),
        "global_co2_latest_total_tons": float(
            recent["co2_tons"].sum()
        ),
        "highest_emitter_latest": str(
            recent.loc[recent["co2_tons"].idxmax(), "country"]
        ),
        "highest_emission_value": float(recent["co2_tons"].max()),
    }


if __name__ == "__main__":
    raw = load_data()
    clean = clean_data(raw)
    featured = engineer_features(clean)
    stats = get_summary_stats(featured)
    print("Dataset loaded and processed successfully.")
    for k, v in stats.items():
        print(f"  {k}: {v}")
