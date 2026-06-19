import sys
import os

# Make project root importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_processing import load_data, clean_data, engineer_features, get_country_series, get_top_emitters, get_summary_stats
from analysis import descriptive_analysis, diagnostic_analysis, predictive_summary, prescriptive_analysis
from arima_model import train_and_forecast, load_forecast

# Page config 
st.set_page_config(
    page_title="Global CO2 Emission Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a472a, #2d6a4f, #52b788);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: #1e2d2f;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #52b788;
        margin-bottom: 0.8rem;
    }
    .risk-critical { color: #ff4444; font-weight: bold; }
    .risk-high     { color: #ff8800; font-weight: bold; }
    .risk-medium   { color: #ffcc00; font-weight: bold; }
    .risk-low      { color: #44bb44; font-weight: bold; }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #52b788;
        border-bottom: 2px solid #52b788;
        padding-bottom: 0.3rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Data loading
@st.cache_data(show_spinner="Loading dataset...")
def get_data():
    raw = load_data()
    clean = clean_data(raw)
    return engineer_features(clean)


@st.cache_data(show_spinner="Running analysis...")
def get_descriptive(_df):
    return descriptive_analysis(_df)


@st.cache_data(show_spinner="Running diagnostic analysis...")
def get_diagnostic(_df):
    return diagnostic_analysis(_df)


@st.cache_data(show_spinner="Fitting ARIMA models...")
def get_forecasts(_df, n_countries, since_year):
    top = get_top_emitters(_df, n=n_countries, since_year=since_year)
    results = []
    for country in top:
        cached = load_forecast(country)
        if cached:
            results.append(cached)
        else:
            series = get_country_series(_df, country)
            r = train_and_forecast(series, country, horizon=10, log_to_mlflow=True)
            results.append(r)
    return results, top


# Colour helpers
PALETTE = px.colors.qualitative.Set2
RISK_COLOURS = {"Critical": "#ff4444", "High": "#ff8800", "Medium": "#ffcc00", "Low": "#44bb44"}


def risk_badge(label: str) -> str:
    colour = RISK_COLOURS.get(label, "#888")
    return f'<span style="background:{colour};color:#000;padding:2px 8px;border-radius:4px;font-size:0.85rem;">{label}</span>'


# Main header 
st.markdown('<p class="main-header">🌍 Global CO2 Emission Analysis</p>', unsafe_allow_html=True)
st.caption("End-to-end data science pipeline: Descriptive → Diagnostic → Predictive (ARIMA) → Prescriptive")

df = get_data()
stats = get_summary_stats(df)
countries_list = sorted(df["country"].unique().tolist())

# Sidebar 
with st.sidebar:
    st.image("https://img.icons8.com/color/96/co2.png", width=80)
    st.title("Controls")

    st.markdown("**Dataset Info**")
    st.info(
        f"Countries: {stats['total_countries']}  \n"
        f"Years: {stats['year_range'][0]}–{stats['year_range'][1]}  \n"
        f"Rows: {stats['total_rows']:,}"
    )

    selected_tab = st.radio(
        "Analysis Layer",
        ["📊 Descriptive", "🔍 Diagnostic", "🔮 Predictive", "💡 Prescriptive", "🔎 Country Deep-Dive"],
        index=0,
    )

    st.divider()
    n_top = st.slider("Top N countries for analysis", 5, 30, 10)
    since_year = st.slider("Data since year", 1900, 2000, 1950)

    st.divider()
    st.markdown("**Quick country lookup**")
    selected_country = st.selectbox("Country", countries_list, index=countries_list.index("China") if "China" in countries_list else 0)


# TAB 1: DESCRIPTIVE

if selected_tab == "📊 Descriptive":
    st.markdown('<p class="section-header">Descriptive Analysis</p>', unsafe_allow_html=True)

    desc = get_descriptive(df)

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    latest_year = desc["year_range"][1]
    latest_total = next(
        (r["global_co2_tons"] for r in desc["global_annual_totals"] if r["year"] == latest_year), 0
    )
    top_country = desc["top10_latest_year_emitters"][0]["country"] if desc["top10_latest_year_emitters"] else "N/A"
    top_val = desc["top10_latest_year_emitters"][0]["co2_tons"] if desc["top10_latest_year_emitters"] else 0

    col1.metric("Total Countries", stats["total_countries"])
    col2.metric("Latest Year", latest_year)
    col3.metric("Global CO2 Latest Year", f"{latest_total/1e9:.1f}B tons")
    col4.metric("Largest Emitter", top_country)

    st.divider()

    # Global CO2 trend
    st.markdown("#### Global Annual CO2 Emissions (1750–2020)")
    global_df = pd.DataFrame(desc["global_annual_totals"])
    fig = px.area(
        global_df, x="year", y="global_co2_tons",
        color_discrete_sequence=["#52b788"],
        labels={"global_co2_tons": "CO2 Emissions (Tons)", "year": "Year"},
    )
    fig.update_layout(
        template="plotly_dark", height=360,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Top 10 All-Time Emitters")
        top10_df = pd.DataFrame(desc["top10_alltime_emitters"])
        fig2 = px.bar(
            top10_df.sort_values("total_co2_tons"),
            x="total_co2_tons", y="country",
            orientation="h",
            color="total_co2_tons",
            color_continuous_scale="Greens",
            labels={"total_co2_tons": "Total CO2 (Tons)", "country": ""},
        )
        fig2.update_layout(template="plotly_dark", height=360, coloraxis_showscale=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("#### Emission by Decade (Global)")
        decade_df = pd.DataFrame(desc["decade_totals"])
        fig3 = px.bar(
            decade_df, x="decade", y="co2_tons",
            color="co2_tons",
            color_continuous_scale="YlOrRd",
            labels={"co2_tons": "CO2 (Tons)", "decade": "Decade"},
        )
        fig3.update_layout(template="plotly_dark", height=360, coloraxis_showscale=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    # Top emitters in latest year
    st.markdown("#### Top 10 Emitters — Latest Year")
    latest_df2 = pd.DataFrame(desc["top10_latest_year_emitters"])
    fig4 = px.pie(
        latest_df2, values="co2_tons", names="country",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4,
    )
    fig4.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig4, use_container_width=True)

    # Per capita
    if desc["top10_per_capita_latest"]:
        st.markdown("#### Highest Per-Capita Emitters (Latest Year)")
        pc_df = pd.DataFrame(desc["top10_per_capita_latest"])
        fig5 = px.bar(
            pc_df.sort_values("co2_per_capita"),
            x="co2_per_capita", y="country", orientation="h",
            color="co2_per_capita", color_continuous_scale="Oranges",
            labels={"co2_per_capita": "CO2 per Capita (Tons)", "country": ""},
        )
        fig5.update_layout(template="plotly_dark", height=360, coloraxis_showscale=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig5, use_container_width=True)


# TAB 2: DIAGNOSTIC

elif selected_tab == "🔍 Diagnostic":
    st.markdown('<p class="section-header">Diagnostic Analysis</p>', unsafe_allow_html=True)

    diag = get_diagnostic(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Fastest Growth Decade", str(diag["fastest_growth_decade"]) + "s")
    col2.metric("Biggest Drop Decade", str(diag["biggest_drop_decade"]) + "s")
    col3.metric("Pop-CO2 Correlation", f"{diag['correlation_population_co2']:.3f}" if diag["correlation_population_co2"] else "N/A")

    st.divider()

    # Emission share shift
    st.markdown("#### Emission Share Shift: 1950 vs 2020 (Top 5 Countries)")
    col_a, col_b = st.columns(2)

    share_1950 = diag.get("emission_share_1950", [])
    share_2020 = diag.get("emission_share_2020", [])

    if share_1950:
        with col_a:
            st.markdown("**1950**")
            df_1950 = pd.DataFrame(share_1950)
            fig = px.pie(df_1950, values="share_pct", names="country", hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

    if share_2020:
        with col_b:
            st.markdown("**2020**")
            df_2020 = pd.DataFrame(share_2020)
            fig = px.pie(df_2020, values="share_pct", names="country", hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Period growth
    st.markdown("#### Emission Growth by Historical Period")
    for period_key, period_data in diag["period_growth_rates"].items():
        y0, y1 = period_data["period"]
        st.markdown(f"**{period_key.replace('_', ' ').title()} ({y0}–{y1})**")
        growers_df = pd.DataFrame(period_data["top_5_growers"])
        if not growers_df.empty:
            fig = px.bar(
                growers_df, x="country", y="growth_pct",
                color="growth_pct", color_continuous_scale="RdYlGn_r",
                labels={"growth_pct": "Growth (%)", "country": "Country"},
            )
            fig.update_layout(template="plotly_dark", height=280, coloraxis_showscale=False,
                             margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Volatility
    st.markdown("#### Most Volatile Emitters (1990–2020)")
    vol_df = pd.DataFrame(diag["most_volatile_countries_1990_2020"])
    if not vol_df.empty:
        vol_df.columns = ["country", "co2_yoy_pct_std"]
        fig = px.bar(
            vol_df, x="country", y="co2_yoy_pct_std",
            color="co2_yoy_pct_std", color_continuous_scale="Reds",
            labels={"co2_yoy_pct_std": "Std Dev YoY Change (%)", "country": ""},
        )
        fig.update_layout(template="plotly_dark", height=320, coloraxis_showscale=False,
                         margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


# TAB 3: PREDICTIVE

elif selected_tab == "🔮 Predictive":
    st.markdown('<p class="section-header">Predictive Analysis</p>', unsafe_allow_html=True)
    st.info("ARIMA models are fitted per country. Results are cached after first run.")

    forecast_results, top_countries = get_forecasts(df, n_top, since_year)
    pred = predictive_summary(forecast_results)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Countries Analysed", pred.get("countries_analysed", 0))
    col2.metric("Critical Risk", len(pred.get("critical_countries", [])))
    col3.metric("High Risk", len(pred.get("high_risk_countries", [])))
    col4.metric("Forecast Horizon", f"{pred.get('forecast_horizon_years', 10)} years")

    st.divider()

    # Risk table
    st.markdown("#### Country Risk Assessment")
    risk_df = pd.DataFrame(pred.get("country_risk_table", []))
    if not risk_df.empty:
        risk_df = risk_df.fillna(0)
        fig = px.scatter(
            risk_df, x="trend_pct_10y_avg", y="risk_score",
            size="risk_score", color="risk_label",
            color_discrete_map=RISK_COLOURS,
            text="country",
            labels={"trend_pct_10y_avg": "10Y Avg Trend (%)", "risk_score": "Risk Score"},
            hover_data=["country", "risk_label", "risk_score"],
        )
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            risk_df[["country", "risk_label", "risk_score", "trend_pct_10y_avg", "forecast_2030"]].style.format({
                "risk_score": "{:.1f}",
                "trend_pct_10y_avg": "{:.2f}%",
                "forecast_2030": "{:,.0f}",
            }),
            use_container_width=True,
            height=300,
        )

    # Forecast charts for each country
    st.markdown("#### Individual Country Forecasts")
    valid_results = [r for r in forecast_results if "error" not in r]

    for i in range(0, len(valid_results), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j >= len(valid_results):
                break
            r = valid_results[i + j]
            with col:
                hist_years = r["historical_years"]
                hist_vals = r["historical_values"]
                fc_years = r["forecast_years"]
                fc_vals = r["forecast_values"]
                lower = r["conf_int_lower"]
                upper = r["conf_int_upper"]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist_years[-40:], y=hist_vals[-40:],
                    mode="lines", name="Historical",
                    line=dict(color="#52b788", width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=fc_years, y=fc_vals,
                    mode="lines+markers", name="Forecast",
                    line=dict(color="#ff8c00", width=2, dash="dash"),
                ))
                fig.add_trace(go.Scatter(
                    x=fc_years + fc_years[::-1],
                    y=upper + lower[::-1],
                    fill="toself", fillcolor="rgba(255,140,0,0.15)",
                    line=dict(color="rgba(255,140,0,0)"),
                    name="95% CI",
                ))
                risk_col = RISK_COLOURS.get(r.get("risk_label", "Low"), "#888")
                fig.add_annotation(
                    x=0.98, y=0.95, xref="paper", yref="paper",
                    text=f"Risk: {r.get('risk_label','?')} ({r.get('risk_score','?')})",
                    showarrow=False,
                    bgcolor=risk_col, font=dict(color="#000", size=11),
                    borderpad=4,
                )
                fig.update_layout(
                    title=f"{r['country']} — ARIMA{tuple(r['order'])}",
                    template="plotly_dark", height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", y=-0.15),
                )
                st.plotly_chart(fig, use_container_width=True)

    # Aggregate forecast
    agg = pred.get("aggregate_forecast", [])
    if agg:
        st.markdown("#### Aggregate Forecast (Sum of Analysed Countries)")
        agg_df = pd.DataFrame(agg)
        fig = px.bar(
            agg_df, x="year", y="aggregate_co2_tons",
            color="aggregate_co2_tons", color_continuous_scale="Reds",
            labels={"aggregate_co2_tons": "Total CO2 (Tons)", "year": "Year"},
        )
        fig.update_layout(template="plotly_dark", height=320, coloraxis_showscale=False,
                         margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


# TAB 4: PRESCRIPTIVE

elif selected_tab == "💡 Prescriptive":
    st.markdown('<p class="section-header">Prescriptive Analysis</p>', unsafe_allow_html=True)
    st.info("Based on the Paris Agreement target: 45% reduction in CO2 below 2019 levels by 2030.")

    reduction_target = st.slider("Reduction Target (%)", 20, 70, 45)

    forecast_results, _ = get_forecasts(df, n_top, since_year)
    presc = prescriptive_analysis(df, forecast_results, reduction_target_pct=reduction_target)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Countries Assessed", presc["countries_assessed"])
    col2.metric("Over Target", presc["countries_over_target"])
    col3.metric("On Track", presc["countries_on_track"])
    global_gap = presc.get("global_gap_tons", 0)
    col4.metric("Global Gap (tons)", f"{global_gap/1e9:.2f}B" if abs(global_gap) > 1e9 else f"{global_gap/1e6:.1f}M")

    st.divider()

    prescriptions = presc.get("country_prescriptions", [])

    # Gap chart
    if prescriptions:
        presc_df = pd.DataFrame([
            {
                "country": p["country"],
                "gap_pct": p["gap_pct_of_baseline"],
                "urgency": p["urgency"],
                "risk_label": p["risk_label"],
            }
            for p in prescriptions
        ])

        fig = px.bar(
            presc_df.sort_values("gap_pct", ascending=False),
            x="country", y="gap_pct",
            color="urgency",
            color_discrete_map={"Immediate": "#ff4444", "On Track": "#44bb44"},
            labels={"gap_pct": "Gap vs Target (% of 2019 Baseline)", "country": ""},
            title=f"Emission Gap to {reduction_target}% Reduction Target by 2030",
        )
        fig.add_hline(y=0, line_color="white", line_dash="dot")
        fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Country cards
        st.markdown("#### Country-Level Prescriptions")
        for p in prescriptions[:n_top]:
            with st.expander(
                f"{'🔴' if p['urgency']=='Immediate' else '🟢'} {p['country']} — "
                f"Risk: {p['risk_label']} | Gap: {p['gap_pct_of_baseline']:.1f}%"
            ):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("2019 Baseline", f"{p['baseline_2019_tons']/1e6:.1f}M tons")
                col_b.metric("2030 Target", f"{p['target_2030_tons']/1e6:.1f}M tons")
                col_c.metric("2030 Forecast", f"{p['forecast_2030_tons']/1e6:.1f}M tons")

                st.markdown(f"**Urgency:** {p['urgency']}  |  **Risk Level:** {p['risk_label']}")
                st.markdown("**Recommended Actions:**")
                for action in p["recommended_actions"]:
                    st.markdown(f"  - {action}")


# TAB 5: COUNTRY DEEP DIVE

elif selected_tab == "🔎 Country Deep-Dive":
    st.markdown(f'<p class="section-header">Country Deep-Dive: {selected_country}</p>', unsafe_allow_html=True)

    country_df = df[df["country"] == selected_country].sort_values("year")

    if country_df.empty:
        st.error(f"No data found for {selected_country}")
    else:
        # KPIs
        latest = country_df.iloc[-1]
        first = country_df.iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Year", int(latest["year"]))
        col2.metric("Latest CO2 (tons)", f"{latest['co2_tons']/1e6:.1f}M")
        col3.metric("Population (2022)", f"{int(latest['population_2022']):,}" if pd.notna(latest["population_2022"]) else "N/A")
        col4.metric("Data From", int(first["year"]))

        st.divider()

        # Historical chart
        st.markdown("#### Historical CO2 Emissions")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                           subplot_titles=["Annual CO2 Emissions (Tons)", "Year-over-Year % Change"])

        fig.add_trace(go.Scatter(
            x=country_df["year"], y=country_df["co2_tons"],
            mode="lines", name="CO2 Emissions",
            line=dict(color="#52b788", width=2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=country_df["year"], y=country_df["co2_rolling_10y"],
            mode="lines", name="10Y Rolling Avg",
            line=dict(color="#ffb703", width=1.5, dash="dash"),
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=country_df["year"], y=country_df["co2_yoy_pct"],
            name="YoY %",
            marker_color=country_df["co2_yoy_pct"].apply(
                lambda v: "#ff4444" if (pd.notna(v) and v > 0) else "#44bb44"
            ),
        ), row=2, col=1)

        fig.update_layout(template="plotly_dark", height=480, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # ARIMA forecast for this country
        st.markdown("#### ARIMA Forecast")
        with st.spinner(f"Fitting ARIMA for {selected_country}..."):
            cached = load_forecast(selected_country)
            if not cached:
                series = get_country_series(df, selected_country)
                result = train_and_forecast(series, selected_country, horizon=10, log_to_mlflow=True)
            else:
                result = cached

        if "error" in result:
            st.error(f"Forecast failed: {result['error']}")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("ARIMA Order", str(tuple(result["order"])))
            col2.metric("AIC", f"{result['aic']:.1f}")
            col3.metric("RMSE", f"{result['rmse']:.2e}")
            col4.metric(
                "Risk Label",
                result["risk_label"],
                delta=f"Score: {result['risk_score']}",
                delta_color="inverse",
            )

            fig2 = go.Figure()
            hist_years = result["historical_years"]
            hist_vals = result["historical_values"]
            fc_years = result["forecast_years"]
            fc_vals = result["forecast_values"]
            lower = result["conf_int_lower"]
            upper = result["conf_int_upper"]

            fig2.add_trace(go.Scatter(
                x=hist_years[-50:], y=hist_vals[-50:],
                mode="lines", name="Historical",
                line=dict(color="#52b788", width=2),
            ))
            fig2.add_trace(go.Scatter(
                x=fc_years, y=fc_vals,
                mode="lines+markers", name="Forecast",
                line=dict(color="#ff8c00", width=2, dash="dash"),
                marker=dict(size=6),
            ))
            fig2.add_trace(go.Scatter(
                x=fc_years + fc_years[::-1],
                y=upper + lower[::-1],
                fill="toself", fillcolor="rgba(255,140,0,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% Confidence Interval",
            ))
            fig2.update_layout(
                title=f"{selected_country} — 10-Year CO2 Forecast",
                template="plotly_dark", height=400,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Risk breakdown
            st.markdown("#### Risk Score Breakdown")
            risk_components = {
                "Emission Level": result.get("level_component", 0),
                "Trend": result.get("trend_component", 0),
                "Forecast Slope": result.get("forecast_component", 0),
            }
            fig3 = go.Figure(go.Bar(
                x=list(risk_components.values()),
                y=list(risk_components.keys()),
                orientation="h",
                marker_color=["#52b788", "#ffb703", "#ff4444"],
            ))
            fig3.update_layout(
                template="plotly_dark", height=220,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Score Component",
            )
            st.plotly_chart(fig3, use_container_width=True)

# Footer
st.divider()
st.caption("CO2 Emission Analysis")
