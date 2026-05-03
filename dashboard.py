"""
dashboard.py  (fixed)
=====================

Aviation Safety Intelligence Dashboard.

Key fixes over the original:
  1. All charts fall back to INCIDENTS (incidents_clean) when INCIDENT_ENRICHED
     columns are missing/null — so the dashboard works even if BTS or weather
     enrichment only partially joined.
  2. RQ1 bar always has data — uses weather_condition (always present in NTSB)
     when weather_category (Open-Meteo) is sparse.
  3. RQ1 scatter falls back to incident-count vs mean-severity per carrier tier
     when route_complexity_score is missing.
  4. RQ2 improvement bar dynamically picks the two EARLIEST and LATEST decades
     in the filtered data, not hardcoded "1980s"/"2010s" (which disappear when
     the year-range slider starts at 2008).
  5. RQ2 area chart adds a fallback when decade is missing.
  6. RQ3 map guards lat/lon range and samples correctly.
  7. is_imc cast robustly (PostgreSQL returns 't'/'f' strings in some drivers).
  8. has_fatalities cast robustly.
  9. apply_filters handles weather filter using weather_condition OR weather_category.
 10. EMPTY_FIG is a function (not a shared mutable object) to avoid Dash
     "figure was mutated" warnings.

Run: python dashboard.py  →  http://localhost:8050
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import Dash, Input, Output, callback, dcc, html
import dash_bootstrap_components as dbc

from config import neon_engine

warnings.filterwarnings("ignore", category=UserWarning)

TEMPLATE = "plotly_white"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data once at startup
# ─────────────────────────────────────────────────────────────────────────────
print("[dashboard] Connecting to PostgreSQL…")
ENGINE = neon_engine()


def _safe_read(sql: str, label: str = "") -> pd.DataFrame:
    try:
        df = pd.read_sql(sql, ENGINE)
        print(f"[dashboard]  ✓ {label or sql[:40]}: {len(df):,} rows")
        return df
    except Exception as exc:
        print(f"[dashboard]  ✗ {label or sql[:40]}: {exc.__class__.__name__} — {exc}")
        return pd.DataFrame()


INCIDENTS         = _safe_read("SELECT * FROM incidents_clean",        "incidents_clean")
CARRIER_STATS     = _safe_read("SELECT * FROM carrier_stats",           "carrier_stats")
INCIDENTS_WEATHER = _safe_read("SELECT * FROM incidents_weather",       "incidents_weather")
INCIDENT_ENRICHED = _safe_read("SELECT * FROM incident_enriched",       "incident_enriched")

# ── ETL cross-dataset analytical tables (built by etl_pipeline.py Step 2) ───
# These tables are pre-joined across all three member datasets:
#   rq1_carrier_risk    : per-carrier M1 (NTSB) + M2 (BTS) join
#   rq2_weather_carrier : weather_category x carrier_tier x year (M1+M2+M3)
#   rq3_state_risk      : state-level aggregation with weather + carrier mix
ETL_RQ1 = _safe_read("SELECT * FROM rq1_carrier_risk",    "rq1_carrier_risk")
ETL_RQ2 = _safe_read("SELECT * FROM rq2_weather_carrier", "rq2_weather_carrier")
ETL_RQ3 = _safe_read("SELECT * FROM rq3_state_risk",      "rq3_state_risk")

# ── robust bool cast ─────────────────────────────────────────────────────────
def _to_bool(series: pd.Series) -> pd.Series:
    """Handle PostgreSQL 't'/'f', Python True/False, 1/0, 'true'/'false'."""
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["true", "t", "1", "yes", "y"])


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "severity_score" in df.columns:
        df["severity_score"] = pd.to_numeric(df["severity_score"], errors="coerce")
    if "is_imc" in df.columns:
        df["is_imc"] = _to_bool(df["is_imc"])
    if "has_fatalities" in df.columns:
        df["has_fatalities"] = _to_bool(df["has_fatalities"])
    # Ensure decade column is clean string e.g. "1980s"
    if "decade" in df.columns:
        df["decade"] = df["decade"].astype(str).str.strip()
        df.loc[df["decade"].isin(["nan", "None", "<NA>", ""]), "decade"] = np.nan
    return df


INCIDENTS         = _prep(INCIDENTS)
INCIDENT_ENRICHED = _prep(INCIDENT_ENRICHED)
INCIDENTS_WEATHER = _prep(INCIDENTS_WEATHER)

# ── pick the best available source ───────────────────────────────────────────
# INCIDENT_ENRICHED is preferred; fall back to INCIDENTS for any column missing
def _best(preferred: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    if not preferred.empty:
        return preferred
    return fallback

MAIN = _best(INCIDENT_ENRICHED, INCIDENTS)

print(f"[dashboard] Main source: {'incident_enriched' if not INCIDENT_ENRICHED.empty else 'incidents_clean'} — {len(MAIN):,} rows")
print(f"[dashboard] Columns: {sorted(MAIN.columns.tolist())}")

# ── KPIs ─────────────────────────────────────────────────────────────────────
def _compute_kpis() -> dict:
    src = INCIDENTS if not INCIDENTS.empty else MAIN
    out = {"total_incidents": len(src), "fatal_rate_pct": 0.0,
           "weather_linked_pct": 0.0, "decade_improvement_pct": 0.0,
           "carriers_joined": 0, "states_analysed": 0}
    if src.empty:
        return out
    if "has_fatalities" in src.columns:
        out["fatal_rate_pct"] = src["has_fatalities"].fillna(False).astype(int).mean() * 100
    if "is_imc" in src.columns:
        out["weather_linked_pct"] = src["is_imc"].fillna(False).astype(int).mean() * 100
    if "decade" in src.columns and "has_fatalities" in src.columns:
        rates = (src.dropna(subset=["decade"])
                 .groupby("decade")["has_fatalities"]
                 .apply(lambda s: s.fillna(False).astype(int).mean()))
        decades = sorted(rates.index)
        if len(decades) >= 2:
            first, last = decades[0], decades[-1]
            if rates[first] > 0:
                out["decade_improvement_pct"] = (rates[first] - rates[last]) / rates[first] * 100
    # ETL cross-dataset stats
    if not ETL_RQ1.empty and "air_carrier" in ETL_RQ1.columns:
        out["carriers_joined"] = ETL_RQ1["air_carrier"].nunique()
    if not ETL_RQ3.empty and "state" in ETL_RQ3.columns:
        out["states_analysed"] = len(ETL_RQ3)
    return out

KPIS = _compute_kpis()
print(f"[dashboard] KPIs: {KPIS}")

# ── Dropdown options ──────────────────────────────────────────────────────────
def _opts(series: pd.Series, label_all: str = "All") -> list[dict]:
    vals = sorted([v for v in series.dropna().unique() if str(v).strip()])
    return [{"label": label_all, "value": "All"}] + [{"label": v, "value": v} for v in vals]

PHASE_OPTIONS   = _opts(MAIN.get("phase_simplified", pd.Series(dtype=str)))
WEATHER_OPTIONS = [{"label": "All", "value": "All"},
                   {"label": "VMC", "value": "VMC"},
                   {"label": "IMC", "value": "IMC"}]
CARRIER_TIER_OPTIONS = [
    {"label": "All",               "value": "All"},
    {"label": "Major",             "value": "major"},
    {"label": "Regional",          "value": "regional"},
    {"label": "General Aviation",  "value": "general_aviation"},
]

YEAR_MIN, YEAR_MAX = 1982, 2024
if not INCIDENTS.empty and "year" in INCIDENTS.columns and INCIDENTS["year"].notna().any():
    YEAR_MIN = int(max(1982, INCIDENTS["year"].dropna().min()))
    YEAR_MAX = int(min(2024, INCIDENTS["year"].dropna().max()))

TIER_COLORS = {
    "major":            "#2563EB",
    "regional":         "#DC2626",
    "general_aviation": "#6B7280",
}


def empty_fig(msg: str = "No data for selected filters") -> go.Figure:
    return go.Figure().update_layout(
        template=TEMPLATE,
        annotations=[dict(text=msg, xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=16, color="grey"))],
        margin=dict(t=40, b=20, l=20, r=20),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Layout helpers
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(title: str, value: str, color: str = "primary"):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted small text-uppercase fw-semibold"),
            html.H3(value, className="mb-0", style={"color": f"var(--bs-{color})"}),
        ]),
        className="shadow-sm h-100",
    )


improvement_color = "success" if KPIS["decade_improvement_pct"] > 0 else "warning"

header = dbc.Container([
    html.H2("Aviation Safety Intelligence Dashboard", className="mt-3 mb-1"),
    html.P("NTSB incidents joined with BTS on-time data and Open-Meteo weather history. "
           "Three research questions, one screen.", className="text-muted mb-3"),
    dbc.Row([
        dbc.Col(kpi_card("Total incidents",
                         f"{KPIS['total_incidents']:,}", "primary"), md=2),
        dbc.Col(kpi_card("Fatal accident rate",
                         f"{KPIS['fatal_rate_pct']:.1f}%", "danger"), md=2),
        dbc.Col(kpi_card("Weather-linked (IMC)",
                         f"{KPIS['weather_linked_pct']:.1f}%", "info"), md=2),
        dbc.Col(kpi_card("Decade improvement",
                         f"{KPIS['decade_improvement_pct']:+.1f}%",
                         improvement_color), md=2),
        dbc.Col(kpi_card("Carriers cross-joined",
                         f"{KPIS['carriers_joined']}",
                         "success"), md=2),
        dbc.Col(kpi_card("States analysed",
                         f"{KPIS['states_analysed']}",
                         "warning"), md=2),
    ], className="g-3 mb-3"),
], fluid=True)

filter_row = dbc.Container([
    dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Year range", className="fw-semibold"),
                dcc.RangeSlider(
                    id="filter-year", min=YEAR_MIN, max=YEAR_MAX, step=1,
                    value=[YEAR_MIN, YEAR_MAX],
                    marks={y: str(y) for y in range(YEAR_MIN, YEAR_MAX + 1, 5)},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], md=5),
            dbc.Col([
                html.Label("Carrier tier", className="fw-semibold"),
                dcc.Dropdown(id="filter-tier", options=CARRIER_TIER_OPTIONS,
                             value="All", clearable=False),
            ], md=2),
            dbc.Col([
                html.Label("Phase of flight", className="fw-semibold"),
                dcc.Dropdown(id="filter-phase", options=PHASE_OPTIONS,
                             value="All", clearable=False),
            ], md=2),
            dbc.Col([
                html.Label("Weather condition", className="fw-semibold"),
                dcc.Dropdown(id="filter-weather", options=WEATHER_OPTIONS,
                             value="All", clearable=False),
            ], md=3),
        ], className="g-3"),
    ]), className="shadow-sm mb-3"),
], fluid=True)


def _graph(id_: str, height: int = 380) -> dcc.Graph:
    return dcc.Loading(dcc.Graph(id=id_, style={"height": f"{height}px"},
                                 config={"displayModeBar": False}))


tab1 = dbc.Container([
    dbc.Row([
        dbc.Col(_graph("rq1-bar", 400), md=7),
        dbc.Col(_graph("rq1-line", 400), md=5),
    ], className="g-3"),
    dbc.Row([dbc.Col(_graph("rq1-scatter", 380), md=12)], className="g-3"),
    dbc.Row([dbc.Col(dbc.Alert(id="rq1-insight", color="info"), md=12)]),
], fluid=True, className="pt-3")

tab2 = dbc.Container([
    dbc.Row([
        dbc.Col(_graph("rq2-lines", 400), md=8),
        dbc.Col(_graph("rq2-bar-improvement", 400), md=4),
    ], className="g-3"),
    dbc.Row([dbc.Col(_graph("rq2-area", 360), md=12)], className="g-3"),
    dbc.Row([dbc.Col(dbc.Alert(id="rq2-insight", color="info"), md=12)]),
], fluid=True, className="pt-3")

tab3 = dbc.Container([
    dbc.Row([dbc.Col(_graph("rq3-map", 500), md=12)], className="g-3"),
    dbc.Row([
        dbc.Col(_graph("rq3-states", 400), md=6),
        dbc.Col(_graph("rq3-mountain", 400), md=6),
    ], className="g-3"),
    dbc.Row([dbc.Col(dbc.Alert(id="rq3-insight", color="info"), md=12)]),
], fluid=True, className="pt-3")

# ── Tab 4: Cross-dataset ETL research questions ───────────────────────────────
# Populated from rq1_carrier_risk, rq2_weather_carrier, rq3_state_risk
# which are pre-joined analytical tables built by etl_pipeline.py.
# The global filter bar does NOT apply here — these charts use the ETL tables
# directly and expose their own inline controls.
_etl_note = dbc.Alert(
    [
        html.Strong("ℹ️ ETL Cross-Dataset Tab — "),
        "Charts below use pre-joined tables built by ",
        html.Code("etl_pipeline.py"),
        " (rq1_carrier_risk, rq2_weather_carrier, rq3_state_risk). "
        "The sidebar filter bar above does not apply here. "
        "Run ",
        html.Code("python etl_pipeline.py --skip-members"),
        " to rebuild these tables if they appear empty.",
    ],
    color="secondary", className="mb-3",
)

_etl_rq1_controls = dbc.Row([
    dbc.Col([
        html.Label("Carrier tier", className="fw-semibold small"),
        dcc.Checklist(
            id="etl-tier-filter",
            options=[{"label": " Major",    "value": "major"},
                     {"label": " Regional", "value": "regional"}],
            value=["major", "regional"],
            inline=True,
            inputStyle={"marginRight": "4px"},
            labelStyle={"marginRight": "14px"},
        ),
    ], md=4),
    dbc.Col([
        html.Label("X-axis metric (BTS M2)", className="fw-semibold small"),
        dcc.Dropdown(
            id="etl-xmetric",
            options=[
                {"label": "Cancellation rate",           "value": "cancel_rate"},
                {"label": "Mean arrival delay (min)",    "value": "mean_arr_delay"},
                {"label": "Mean departure delay (min)",  "value": "mean_dep_delay"},
                {"label": "Diversion rate",              "value": "divert_rate"},
                {"label": "Route complexity (# routes)", "value": "route_complexity_score"},
            ],
            value="cancel_rate", clearable=False,
        ),
    ], md=4),
    dbc.Col([
        html.Label("Choropleth metric (state map)", className="fw-semibold small"),
        dcc.Dropdown(
            id="etl-map-metric",
            options=[
                {"label": "Fatal rate (%)",         "value": "fatal_rate"},
                {"label": "Mean severity score",    "value": "mean_severity"},
                {"label": "GA proportion (%)",      "value": "ga_pct"},
                {"label": "IMC proportion (%)",     "value": "imc_pct"},
                {"label": "Mean wind speed (km/h)", "value": "mean_windspeed"},
            ],
            value="fatal_rate", clearable=False,
        ),
    ], md=4),
], className="g-2 mb-3")

_etl_rq2_year_slider = dbc.Row([
    dbc.Col([
        html.Label("Year range (RQ2 heatmap + IMC trend)", className="fw-semibold small"),
        dcc.RangeSlider(
            id="etl-year-range",
            min=YEAR_MIN, max=YEAR_MAX, step=1,
            value=[max(YEAR_MIN, YEAR_MAX - 15), YEAR_MAX],
            marks={y: str(y) for y in range(YEAR_MIN, YEAR_MAX + 1, 5)},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ], md=12),
], className="mb-3")

tab4 = dbc.Container([
    _etl_note,

    # Controls row
    dbc.Card(dbc.CardBody([_etl_rq1_controls, _etl_rq2_year_slider]),
             className="shadow-sm mb-3"),

    # RQ1 — Carrier performance vs accident risk
    html.H5("RQ1 — Does BTS operational performance predict NTSB accident risk?",
            className="fw-semibold text-primary mt-2"),
    html.P("Join: M1 (NTSB incidents_clean) ↔ M2 (BTS carrier_stats) on carrier name.",
           className="text-muted small mb-2"),
    dbc.Row([
        dbc.Col(_graph("etl-rq1-scatter", 400), md=7),
        dbc.Col(_graph("etl-rq1-bar",     400), md=5),
    ], className="g-3"),
    dbc.Row([dbc.Col(_graph("etl-rq1-bubble", 360), md=12)], className="g-3 mb-3"),

    # RQ2 — Weather × carrier quality
    html.H5("RQ2 — How does weather interact with carrier quality to drive incident severity?",
            className="fw-semibold text-primary mt-2"),
    html.P("Join: M1 + M3 (Open-Meteo weather per incident) grouped by M2 carrier tier.",
           className="text-muted small mb-2"),
    dbc.Row([
        dbc.Col(_graph("etl-rq2-heatmap", 400), md=6),
        dbc.Col(_graph("etl-rq2-windbox", 400), md=6),
    ], className="g-3"),
    dbc.Row([dbc.Col(_graph("etl-rq2-imc", 340), md=12)], className="g-3 mb-3"),

    # RQ3 — State safety drivers
    html.H5("RQ3 — Which states are worst, and is weather or carrier mix the driver?",
            className="fw-semibold text-primary mt-2"),
    html.P("Join: M1 state fatal rates + M3 weather at incident time + M1/M2 carrier mix.",
           className="text-muted small mb-2"),
    dbc.Row([
        dbc.Col(_graph("etl-rq3-choropleth", 440), md=7),
        dbc.Col(_graph("etl-rq3-scatter",    440), md=5),
    ], className="g-3"),
    dbc.Row([dbc.Col(_graph("etl-rq3-bar", 360), md=12)], className="g-3"),
    dbc.Row([dbc.Col(dbc.Alert(id="etl-insight", color="primary"), md=12)]),
], fluid=True, className="pt-3")

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
           title="Aviation Safety Intelligence")
server = app.server

app.layout = dbc.Container([
    header,
    filter_row,
    dbc.Container([
        dbc.Tabs([
            dbc.Tab(tab1, label="RQ1: Weather & carrier risk",    tab_id="tab-rq1"),
            dbc.Tab(tab2, label="RQ2: Severity trends over time", tab_id="tab-rq2"),
            dbc.Tab(tab3, label="RQ3: Geographic hotspots",       tab_id="tab-rq3"),
            dbc.Tab(tab4, label="🔗 Cross-Dataset (ETL)",         tab_id="tab-etl"),
        ], active_tab="tab-rq1"),
    ], fluid=True),
    html.Footer(
        html.Small("Data: NTSB · BTS · Open-Meteo · OurAirports", className="text-muted"),
        className="text-center my-3",
    ),
], fluid=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Shared filter helper
# ─────────────────────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame, year_range, tier: str,
                  phase: str, weather: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    if "year" in out.columns:
        yr = pd.to_numeric(out["year"], errors="coerce")
        out = out[(yr >= year_range[0]) & (yr <= year_range[1])]

    if tier != "All" and "broad_carrier_tier" in out.columns:
        out = out[out["broad_carrier_tier"] == tier]

    if phase != "All" and "phase_simplified" in out.columns:
        out = out[out["phase_simplified"] == phase]

    if weather != "All":
        # Try weather_condition first (always present), then weather_category
        if "weather_condition" in out.columns:
            out = out[out["weather_condition"].str.upper().fillna("") == weather.upper()]
        elif "weather_category" in out.columns:
            out = out[out["weather_category"].str.upper().fillna("") == weather.upper()]

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4a. RQ1 callback — Weather & carrier risk
# ─────────────────────────────────────────────────────────────────────────────
@callback(
    [Output("rq1-bar",     "figure"),
     Output("rq1-line",    "figure"),
     Output("rq1-scatter", "figure"),
     Output("rq1-insight", "children")],
    [Input("filter-year",    "value"),
     Input("filter-tier",    "value"),
     Input("filter-phase",   "value"),
     Input("filter-weather", "value")],
)
def update_rq1(year_range, tier, phase, weather):
    sub = apply_filters(MAIN, year_range, tier, phase, weather)
    if sub.empty:
        ef = empty_fig()
        return ef, ef, ef, "No data for selected filters."

    sub = sub.copy()
    sub["fatal_int"]  = _to_bool(sub["has_fatalities"]).astype(int) if "has_fatalities" in sub.columns else 0
    sub["is_imc_int"] = _to_bool(sub["is_imc"]).astype(int) if "is_imc" in sub.columns else 0

    # ── (a) Bar: mean severity by weather × carrier tier ─────────────────────
    # Use weather_category if populated, else fall back to weather_condition
    weather_col = None
    for col in ("weather_category", "weather_condition"):
        if col in sub.columns and sub[col].notna().sum() > 10:
            weather_col = col
            break

    if weather_col and "broad_carrier_tier" in sub.columns:
        agg = (
            sub.dropna(subset=[weather_col, "broad_carrier_tier", "severity_score"])
            .groupby([weather_col, "broad_carrier_tier"])
            .agg(mean_severity=("severity_score", "mean"),
                 incident_count=("severity_score", "count"))
            .reset_index()
        )
        agg = agg[agg["incident_count"] >= 5]  # filter noise
        if not agg.empty:
            bar = px.bar(
                agg, x=weather_col, y="mean_severity",
                color="broad_carrier_tier", barmode="group",
                color_discrete_map=TIER_COLORS,
                title="Mean incident severity by weather condition & carrier tier",
                template=TEMPLATE,
                labels={weather_col: "Weather condition",
                        "mean_severity": "Mean severity (0=none, 3=fatal)",
                        "broad_carrier_tier": "Carrier tier"},
            )
            bar.update_yaxes(range=[0, 3.2])
        else:
            bar = empty_fig("Not enough data per weather × tier cell")
    else:
        # Ultimate fallback: severity distribution by carrier tier only
        if "broad_carrier_tier" in sub.columns and "severity_score" in sub.columns:
            agg2 = (sub.dropna(subset=["broad_carrier_tier", "severity_score"])
                    .groupby("broad_carrier_tier")["severity_score"]
                    .mean().reset_index(name="mean_severity"))
            bar = px.bar(agg2, x="broad_carrier_tier", y="mean_severity",
                         color="broad_carrier_tier", color_discrete_map=TIER_COLORS,
                         title="Mean severity by carrier tier (weather data unavailable)",
                         template=TEMPLATE)
        else:
            bar = empty_fig("No weather or carrier tier data available")

    # ── (b) Line: IMC rate over time per carrier tier ─────────────────────────
    line_src = sub.dropna(subset=["year", "broad_carrier_tier"]).copy() if "broad_carrier_tier" in sub.columns else pd.DataFrame()
    if not line_src.empty:
        line_data = (
            line_src.groupby(["year", "broad_carrier_tier"])["is_imc_int"]
            .mean().reset_index(name="imc_rate")
        )
        # Only keep tiers with enough years
        tier_counts = line_data.groupby("broad_carrier_tier")["year"].nunique()
        valid_tiers = tier_counts[tier_counts >= 3].index
        line_data = line_data[line_data["broad_carrier_tier"].isin(valid_tiers)]

        if not line_data.empty:
            line = px.line(
                line_data, x="year", y="imc_rate", color="broad_carrier_tier",
                color_discrete_map=TIER_COLORS,
                title="IMC incident rate over time by carrier tier",
                template=TEMPLATE,
                labels={"imc_rate": "IMC rate (proportion)",
                        "year": "Year", "broad_carrier_tier": "Carrier tier"},
            )
            line.update_yaxes(tickformat=".0%")
            line.add_annotation(
                text="Higher = more incidents in instrument conditions",
                xref="paper", yref="paper", x=0.01, y=1.05,
                showarrow=False, font=dict(size=10, color="grey"),
            )
        else:
            line = empty_fig("Insufficient yearly data per carrier tier")
    else:
        line = empty_fig("No carrier tier data in this filter range")

    # ── (c) Scatter: incident count vs mean severity per carrier tier ─────────
    # Primary: route_complexity_score if available
    # Fallback: incident count vs mean severity per air_carrier
    if ("route_complexity_score" in sub.columns
            and sub["route_complexity_score"].notna().sum() > 20
            and "air_carrier" in sub.columns):
        scatter_src = (
            sub.dropna(subset=["route_complexity_score", "air_carrier"])
            .groupby("air_carrier")
            .agg(route_complexity_score=("route_complexity_score", "max"),
                 mean_severity=("severity_score", "mean"),
                 incident_count=("severity_score", "count"),
                 broad_carrier_tier=("broad_carrier_tier", "first"))
            .reset_index()
        )
        scatter_src = scatter_src[scatter_src["incident_count"] >= 5]
        scatter = px.scatter(
            scatter_src, x="route_complexity_score", y="mean_severity",
            size="incident_count", color="broad_carrier_tier",
            color_discrete_map=TIER_COLORS, hover_name="air_carrier",
            size_max=40,
            title="Route complexity vs mean incident severity",
            template=TEMPLATE,
            labels={"route_complexity_score": "Route complexity (distinct routes)",
                    "mean_severity": "Mean severity (0–3)",
                    "broad_carrier_tier": "Carrier tier"},
        )
    else:
        # Fallback: phase × carrier tier bubble chart (always works)
        if "phase_simplified" in sub.columns and "broad_carrier_tier" in sub.columns:
            scatter_src = (
                sub.dropna(subset=["phase_simplified", "broad_carrier_tier", "severity_score"])
                .groupby(["phase_simplified", "broad_carrier_tier"])
                .agg(mean_severity=("severity_score", "mean"),
                     fatal_rate=("fatal_int", "mean"),
                     incident_count=("severity_score", "count"))
                .reset_index()
            )
            scatter_src = scatter_src[scatter_src["incident_count"] >= 5]
            scatter = px.scatter(
                scatter_src, x="incident_count", y="mean_severity",
                size="fatal_rate", color="broad_carrier_tier",
                hover_name="phase_simplified",
                color_discrete_map=TIER_COLORS,
                size_max=40,
                title="Incident volume vs severity by phase & carrier tier",
                template=TEMPLATE,
                labels={"incident_count": "Incident count",
                        "mean_severity": "Mean severity (0–3)",
                        "broad_carrier_tier": "Carrier tier",
                        "fatal_rate": "Fatal rate"},
            )
        else:
            scatter = empty_fig("Insufficient data for scatter chart")

    # ── (d) Insight ──────────────────────────────────────────────────────────
    insight = "Insufficient data to compare regional vs major carrier IMC severity."
    if "is_imc" in sub.columns and "broad_carrier_tier" in sub.columns:
        imc_sub = sub[sub["is_imc"] == True].copy()
        if not imc_sub.empty:
            means = imc_sub.groupby("broad_carrier_tier")["severity_score"].mean()
            if {"regional", "major"}.issubset(means.index) and means["major"] > 0:
                pct = (means["regional"] - means["major"]) / means["major"] * 100
                insight = (
                    f"🔍 In IMC conditions, regional carriers show "
                    f"{abs(pct):.1f}% {'higher' if pct > 0 else 'lower'} mean severity "
                    f"than major airlines "
                    f"(regional: {means['regional']:.2f} vs major: {means['major']:.2f} on a 0–3 scale)."
                )
            elif "general_aviation" in means.index and len(means) >= 1:
                top = means.idxmax()
                insight = (
                    f"🔍 In IMC conditions, '{top}' operations have the highest mean severity "
                    f"score ({means[top]:.2f} / 3.0). "
                    f"Carrier tier filter may be limiting comparison — try selecting 'All'."
                )

    return bar, line, scatter, insight


# ─────────────────────────────────────────────────────────────────────────────
# 4b. RQ2 callback — Severity trends over time
# ─────────────────────────────────────────────────────────────────────────────
@callback(
    [Output("rq2-lines",           "figure"),
     Output("rq2-bar-improvement", "figure"),
     Output("rq2-area",            "figure"),
     Output("rq2-insight",         "children")],
    [Input("filter-year",    "value"),
     Input("filter-tier",    "value"),
     Input("filter-phase",   "value"),
     Input("filter-weather", "value")],
)
def update_rq2(year_range, tier, phase, weather):
    sub = apply_filters(MAIN, year_range, tier, phase, weather)
    if sub.empty:
        ef = empty_fig()
        return ef, ef, ef, "No data for selected filters."

    sub = sub.copy()
    sub["fatal_int"] = (_to_bool(sub["has_fatalities"]).astype(int)
                        if "has_fatalities" in sub.columns else 0)

    # ── (a) Multi-line: fatal rate per phase by year ──────────────────────────
    if "phase_simplified" in sub.columns and "year" in sub.columns:
        line_src = sub.dropna(subset=["year", "phase_simplified"]).copy()
        # Group and smooth
        line_data = (
            line_src.groupby(["year", "phase_simplified"])["fatal_int"]
            .mean().mul(100).reset_index(name="fatal_rate_pct")
        )
        # Only show phases with enough data points
        phase_years = line_data.groupby("phase_simplified")["year"].nunique()
        valid_phases = phase_years[phase_years >= 3].index
        line_data = line_data[line_data["phase_simplified"].isin(valid_phases)]

        if not line_data.empty:
            lines = px.line(
                line_data, x="year", y="fatal_rate_pct",
                color="phase_simplified",
                title="Fatal-rate (%) by phase of flight over time",
                template=TEMPLATE,
                labels={"fatal_rate_pct": "Fatal rate (%)",
                        "year": "Year", "phase_simplified": "Phase"},
            )
            lines.update_traces(line=dict(width=2))
        else:
            lines = empty_fig("Not enough yearly data per phase for this filter")
    else:
        # Fallback: overall fatal rate by year
        yr_data = (sub.dropna(subset=["year"])
                   .groupby("year")["fatal_int"]
                   .mean().mul(100).reset_index(name="fatal_rate_pct"))
        if not yr_data.empty:
            lines = px.line(yr_data, x="year", y="fatal_rate_pct",
                            title="Overall fatal rate (%) by year",
                            template=TEMPLATE)
        else:
            lines = empty_fig("No yearly data available")

    # ── (b) Bar: % improvement — dynamically chosen decades ──────────────────
    bar = empty_fig("Insufficient decade data for improvement comparison")
    if "decade" in sub.columns:
        valid_decades = sorted(sub.dropna(subset=["decade"])["decade"].unique())
        # Need at least 2 decades to compare
        if len(valid_decades) >= 2:
            first_dec = valid_decades[0]
            last_dec  = valid_decades[-1]
            d_first   = sub[sub["decade"] == first_dec]
            d_last    = sub[sub["decade"] == last_dec]

            phase_col = "phase_simplified" if "phase_simplified" in sub.columns else "broad_phase_of_flight"
            improvements = []
            if phase_col in sub.columns:
                for ph in sub[phase_col].dropna().unique():
                    r_first = d_first[d_first[phase_col] == ph]["fatal_int"].mean()
                    r_last  = d_last[d_last[phase_col]  == ph]["fatal_int"].mean()
                    if pd.notna(r_first) and pd.notna(r_last) and r_first > 0:
                        improvements.append({
                            "phase": ph,
                            "improvement_pct": (r_first - r_last) / r_first * 100,
                        })

            if improvements:
                imp_df = pd.DataFrame(improvements).sort_values("improvement_pct", ascending=True)
                colors = ["#16A34A" if v > 0 else "#DC2626" for v in imp_df["improvement_pct"]]
                bar = go.Figure(go.Bar(
                    x=imp_df["improvement_pct"],
                    y=imp_df["phase"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.1f}%" for v in imp_df["improvement_pct"]],
                    textposition="outside",
                ))
                bar.update_layout(
                    title=f"Fatal-rate improvement: {first_dec} → {last_dec}",
                    template=TEMPLATE,
                    xaxis_title="% improvement (green = safer)",
                    yaxis_title="Flight phase",
                    margin=dict(l=120),
                )
                bar.add_vline(x=0, line_color="grey", line_width=1)

    # ── (c) Stacked area: incident composition by decade × weather ────────────
    area = empty_fig("No decade data in this filter range")
    if "decade" in sub.columns and "weather_condition" in sub.columns:
        area_src = (
            sub.dropna(subset=["decade", "weather_condition"])
            .groupby(["decade", "weather_condition"])
            .size().reset_index(name="incidents")
        )
        area_src = area_src[area_src["incidents"] >= 5]
        if not area_src.empty:
            area = px.area(
                area_src, x="decade", y="incidents", color="weather_condition",
                title="Incident composition by decade × weather condition",
                template=TEMPLATE,
                labels={"incidents": "Incident count", "decade": "Decade",
                        "weather_condition": "Weather"},
            )

    # ── (d) Insight ──────────────────────────────────────────────────────────
    insight = "Insufficient data to compute decade-on-decade improvement for this filter."
    if "decade" in sub.columns:
        valid_decades = sorted(sub.dropna(subset=["decade"])["decade"].unique())
        if len(valid_decades) >= 2:
            first_dec, last_dec = valid_decades[0], valid_decades[-1]
            d_first = sub[sub["decade"] == first_dec]
            d_last  = sub[sub["decade"] == last_dec]
            phase_col = "phase_simplified" if "phase_simplified" in sub.columns else "broad_phase_of_flight"
            records = []
            if phase_col in sub.columns:
                for ph in sub[phase_col].dropna().unique():
                    r1 = d_first[d_first[phase_col] == ph]["fatal_int"].mean()
                    r2 = d_last[d_last[phase_col]   == ph]["fatal_int"].mean()
                    if pd.notna(r1) and pd.notna(r2) and r1 > 0:
                        records.append((ph, r1, r2, (r1 - r2) / r1 * 100))
            if records:
                best = max(records, key=lambda r: r[3])
                insight = (
                    f"📈 The {best[0]} phase showed the greatest improvement, "
                    f"with fatal rate falling from {best[1]*100:.1f}% ({first_dec}) "
                    f"to {best[2]*100:.1f}% ({last_dec}) "
                    f"— a {best[3]:+.1f}% relative reduction."
                )

    return lines, bar, area, insight


# ─────────────────────────────────────────────────────────────────────────────
# 4c. RQ3 callback — Geographic hotspots
# ─────────────────────────────────────────────────────────────────────────────
@callback(
    [Output("rq3-map",      "figure"),
     Output("rq3-states",   "figure"),
     Output("rq3-mountain", "figure"),
     Output("rq3-insight",  "children")],
    [Input("filter-year",    "value"),
     Input("filter-tier",    "value"),
     Input("filter-phase",   "value"),
     Input("filter-weather", "value")],
)
def update_rq3(year_range, tier, phase, weather):
    # For geographic charts (map/mountain), INCIDENTS_WEATHER is the best source 
    # as it contains the backfilled coordinates from OurAirports.
    # For the state-level bar chart, INCIDENTS provides full coverage.
    
    geo_src = INCIDENTS_WEATHER if not INCIDENTS_WEATHER.empty else MAIN
    state_src = INCIDENTS if not INCIDENTS.empty else MAIN
    
    sub_geo = apply_filters(geo_src, year_range, tier, phase, weather)
    sub_st  = apply_filters(state_src, year_range, tier, phase, weather)
    
    if sub_geo.empty and sub_st.empty:
        ef = empty_fig()
        return ef, ef, ef, "No data for selected filters."

    # Robust boolean casting for stats
    sub_st = sub_st.copy()
    sub_st["is_imc_int"] = (_to_bool(sub_st["is_imc"]).astype(int)
                           if "is_imc" in sub_st.columns else 0)

    # ── (a) Map ───────────────────────────────────────────────────────────────
    m = empty_fig("No geocoded incidents in this filter range")
    if "latitude" in sub_geo.columns and "longitude" in sub_geo.columns:
        geo = sub_geo.dropna(subset=["latitude", "longitude"]).copy()
        geo["latitude"]  = pd.to_numeric(geo["latitude"],  errors="coerce")
        geo["longitude"] = pd.to_numeric(geo["longitude"], errors="coerce")
        # Clip to continental US + AK + HI + territories
        geo = geo.dropna(subset=["latitude", "longitude"])
        geo = geo[(geo["latitude"].between(15, 72)) &
                  (geo["longitude"].between(-180, -60))]

        if not geo.empty:
            plot_geo = geo.sample(min(len(geo), 10_000), random_state=42)
            plot_geo["sev_size"] = (pd.to_numeric(
                plot_geo["severity_score"], errors="coerce").fillna(0) + 1)

            color_col = ("phase_simplified" if "phase_simplified" in plot_geo.columns
                         else "weather_condition")
            m = px.scatter_mapbox(
                plot_geo, lat="latitude", lon="longitude",
                color=color_col,
                size="sev_size", size_max=10, opacity=0.55,
                zoom=2.8, center={"lat": 39.5, "lon": -98.35},
                hover_name="airport_name" if "airport_name" in plot_geo.columns else None,
                hover_data={col: True for col in
                            ["state", "year", "severity_score", "weather_condition"]
                            if col in plot_geo.columns},
                title=f"US incident map — {len(geo):,} geocoded incidents "
                      f"(sample of {len(plot_geo):,} shown)",
                template=TEMPLATE,
            )
            m.update_layout(
                mapbox_style="open-street-map",
                margin=dict(l=0, r=0, t=40, b=0),
            )

    # ── (b) Top-15 states bar ─────────────────────────────────────────────────
    states_fig = empty_fig("No state data available")
    if "state" in sub_st.columns:
        state_src_df = sub_st.dropna(subset=["state"]).copy()
        state_agg = (
            state_src_df.groupby("state")
            .agg(incidents=("event_id", "count") if "event_id" in state_src_df.columns
                           else ("severity_score", "count"),
                 imc_rate=("is_imc_int", "mean"))
            .reset_index()
            .sort_values("incidents", ascending=False)
            .head(15)
        )
        if not state_agg.empty:
            states_fig = px.bar(
                state_agg.sort_values("incidents", ascending=True),
                x="incidents", y="state", orientation="h",
                color="imc_rate", color_continuous_scale="Reds",
                title="Top 15 states by incident count  (colour = IMC rate)",
                template=TEMPLATE,
                labels={"incidents": "Incident count", "state": "State",
                        "imc_rate": "IMC rate"},
            )
            states_fig.update_coloraxes(colorbar_title="IMC rate")

    # ── (c) Mountain-state severity scatter ───────────────────────────────────
    mtn_fig = empty_fig("No state/severity data for terrain chart")
    mountain_set = {"CO", "WA", "AK", "MT", "WY", "ID", "UT", "NV", "OR", "CA"}
    if "state" in sub_geo.columns and "severity_score" in sub_geo.columns:
        # Mountain chart is better with sub_geo because it uses latitude
        count_col = "event_id" if "event_id" in sub_geo.columns else "severity_score"
        state_means = (
            sub_geo.dropna(subset=["state", "severity_score"])
            .groupby("state")
            .agg(mean_lat=("latitude", "mean") if "latitude" in sub_geo.columns
                           else ("severity_score", "count"),
                 mean_sev=("severity_score", "mean"),
                 incidents=(count_col, "count"),
                 imc_rate=("is_imc_int", "mean") if "is_imc_int" in sub_geo.columns else ("latitude", "count"))
            .reset_index()
        )
        if "is_imc_int" not in sub_geo.columns:
             # robust casting if missing
             sub_geo["is_imc_int"] = (_to_bool(sub_geo["is_imc"]).astype(int) if "is_imc" in sub_geo.columns else 0)
             state_means["imc_rate"] = sub_geo.groupby("state")["is_imc_int"].mean().values
        if "mean_lat" not in state_means.columns or state_means["mean_lat"].isna().all():
            # fallback: use incident count on x-axis instead of latitude
            state_means["mean_lat"] = state_means["incidents"]
            x_label = "Incident count"
        else:
            x_label = "Mean latitude (terrain proxy)"

        state_means["mountainous"] = state_means["state"].isin(mountain_set)
        state_means = state_means[state_means["incidents"] >= 10]

        if not state_means.empty:
            mtn_fig = px.scatter(
                state_means,
                x="mean_lat", y="mean_sev",
                size="incidents",
                color="mountainous",
                color_discrete_map={True: "#F59E0B", False: "#6B7280"},
                hover_name="state",
                hover_data={"imc_rate": ":.1%", "incidents": ":,",
                            "mean_sev": ":.2f"},
                title="State terrain proxy vs mean incident severity",
                template=TEMPLATE,
                labels={"mean_lat": x_label,
                        "mean_sev": "Mean severity score (0–3)",
                        "mountainous": "Mountainous state",
                        "incidents": "Incident count",
                        "imc_rate": "IMC rate"},
                size_max=45,
            )
            mtn_fig.add_annotation(
                text="🟡 Mountainous states highlighted",
                xref="paper", yref="paper", x=0.01, y=1.05,
                showarrow=False, font=dict(size=10, color="grey"),
            )

    # ── (d) Insight ──────────────────────────────────────────────────────────
    insight = "Insufficient state-level data for the geographic insight."
    if "state" in sub_st.columns:
        st_counts = sub_st["state"].dropna().value_counts()
        if not st_counts.empty:
            top_state  = st_counts.idxmax()
            top_count  = int(st_counts.iloc[0])
            avg_count  = st_counts.mean()
            ak_count   = int(st_counts.get("AK", 0))
            ak_imc_pct = 0.0
            if "is_imc" in sub_st.columns:
                ak_rows = sub_st[sub_st["state"] == "AK"]["is_imc"]
                ak_imc_pct = _to_bool(ak_rows).mean() * 100

            if ak_count > 0 and avg_count > 0:
                ratio = ak_count / avg_count
                insight = (
                    f"🗺️ Alaska records {ratio:.1f}× more incidents than the average US state "
                    f"in this filter ({ak_count:,} vs avg {avg_count:.0f}), "
                    f"with {ak_imc_pct:.1f}% occurring in IMC conditions. "
                    f"The most incident-prone state overall is {top_state} ({top_count:,} incidents)."
                )
            else:
                insight = (
                    f"🗺️ Most incident-prone state: {top_state} ({top_count:,} incidents). "
                    f"Expand the year range or remove filters to see Alaska IMC statistics."
                )

    return m, states_fig, mtn_fig, insight


# ─────────────────────────────────────────────────────────────────────────────
# 4d. ETL cross-dataset callback — tab 4
# ─────────────────────────────────────────────────────────────────────────────
WEATHER_ORDER_ETL = ["Clear", "Cloudy", "Rain", "Fog", "Snow", "Thunderstorm", "Other", "Unknown"]
WEATHER_COLORS_ETL = {
    "Clear": "#16A34A", "Cloudy": "#60A5FA", "Fog": "#D1D5DB",
    "Rain": "#2563EB", "Snow": "#7C3AED", "Thunderstorm": "#DC2626",
    "Other": "#9CA3AF", "Unknown": "#E5E7EB",
}
TIER_XLABELS = {
    "cancel_rate":           "Cancellation Rate (BTS M2)",
    "mean_arr_delay":        "Mean Arrival Delay — min (BTS M2)",
    "mean_dep_delay":        "Mean Departure Delay — min (BTS M2)",
    "divert_rate":           "Diversion Rate (BTS M2)",
    "route_complexity_score":"Route Complexity — distinct routes (BTS M2)",
}
CHORO_SCALE_COLS = {"fatal_rate", "ga_pct", "imc_pct", "major_carrier_pct"}


@callback(
    [Output("etl-rq1-scatter",    "figure"),
     Output("etl-rq1-bar",        "figure"),
     Output("etl-rq1-bubble",     "figure"),
     Output("etl-rq2-heatmap",    "figure"),
     Output("etl-rq2-windbox",    "figure"),
     Output("etl-rq2-imc",        "figure"),
     Output("etl-rq3-choropleth", "figure"),
     Output("etl-rq3-scatter",    "figure"),
     Output("etl-rq3-bar",        "figure"),
     Output("etl-insight",        "children")],
    [Input("etl-tier-filter", "value"),
     Input("etl-xmetric",     "value"),
     Input("etl-year-range",  "value"),
     Input("etl-map-metric",  "value")],
)
def update_etl_tab(tiers, xmet, yr_range, map_metric):
    ef = empty_fig

    # ── helpers ──────────────────────────────────────────────────────────────
    def _empty(msg="No data — run etl_pipeline.py first"):
        return empty_fig(msg)

    no_etl = ETL_RQ1.empty and ETL_RQ2.empty and ETL_RQ3.empty

    # ═════════════════════════════════════════════════════════════════════════
    # RQ1 charts — carrier performance vs accident risk
    # Source: rq1_carrier_risk (M1 × M2 join, one row per carrier)
    # ═════════════════════════════════════════════════════════════════════════
    if ETL_RQ1.empty or not tiers:
        scat = _empty(); bar_rq1 = _empty(); bub = _empty()
    else:
        d1 = ETL_RQ1[ETL_RQ1["broad_carrier_tier"].isin(tiers)].copy() if "broad_carrier_tier" in ETL_RQ1.columns else ETL_RQ1.copy()
        xl = TIER_XLABELS.get(xmet, xmet)

        # Scatter A: BTS metric (x) vs NTSB fatal rate (y), bubble = incident count
        try:
            scat = px.scatter(
                d1, x=xmet, y="fatal_rate",
                color="broad_carrier_tier", color_discrete_map=TIER_COLORS,
                hover_name="air_carrier",
                hover_data={xmet: ":.4f", "fatal_rate": ":.3f",
                            "ntsb_incidents": True, "mean_severity": ":.2f"},
                size="ntsb_incidents", size_max=32,
                trendline="ols",
                labels={xmet: xl, "fatal_rate": "NTSB Fatal Rate (M1)",
                        "broad_carrier_tier": "Tier"},
                title=f"RQ1a — {xl} vs NTSB fatal rate by carrier",
                template=TEMPLATE,
            )
            scat.update_layout(legend=dict(orientation="h", y=-0.22), height=400)
        except Exception:
            scat = _empty("RQ1a — trendline requires statsmodels: pip install statsmodels")

        # Bar B: mean NTSB severity per BTS severity bucket (delay quartile)
        # Group carriers into delay quartiles (low/medium/high/very-high)
        try:
            if "mean_arr_delay" in d1.columns and "mean_severity" in d1.columns:
                d1q = d1.dropna(subset=["mean_arr_delay", "mean_severity"]).copy()
                d1q["delay_quartile"] = pd.qcut(
                    d1q["mean_arr_delay"], q=4,
                    labels=["Low delay", "Med-low", "Med-high", "High delay"],
                    duplicates="drop",
                )
                agg_bar = (d1q.groupby(["delay_quartile", "broad_carrier_tier"])
                           .agg(mean_severity=("mean_severity", "mean"),
                                n=("mean_severity", "count"))
                           .reset_index())
                bar_rq1 = px.bar(
                    agg_bar, x="delay_quartile", y="mean_severity",
                    color="broad_carrier_tier", barmode="group",
                    color_discrete_map=TIER_COLORS,
                    labels={"delay_quartile": "BTS Delay Quartile (M2)",
                            "mean_severity": "Mean NTSB Severity (M1)",
                            "broad_carrier_tier": "Tier"},
                    title="RQ1b — BTS delay quartile vs NTSB mean severity",
                    template=TEMPLATE,
                )
                bar_rq1.update_layout(legend=dict(orientation="h", y=-0.22), height=400)
            else:
                bar_rq1 = _empty("RQ1b — delay/severity data missing")
        except Exception as exc:
            bar_rq1 = _empty(f"RQ1b — {exc}")

        # Bubble C: route complexity (x) vs incident count (y), bubble = delay
        try:
            bub = px.scatter(
                d1, x="route_complexity_score", y="ntsb_incidents",
                size="mean_arr_delay", color="broad_carrier_tier",
                color_discrete_map=TIER_COLORS,
                hover_name="air_carrier",
                hover_data={"ntsb_incidents": True, "mean_arr_delay": ":.1f",
                            "route_complexity_score": True},
                size_max=38,
                labels={"route_complexity_score": "Route Complexity (BTS M2)",
                        "ntsb_incidents": "NTSB Incident Count (M1)",
                        "mean_arr_delay": "Mean Arr Delay (M2)"},
                title="RQ1c — Route complexity (M2) vs NTSB incidents (M1), bubble = BTS delay",
                template=TEMPLATE,
            )
            bub.update_layout(legend=dict(orientation="h", y=-0.18), height=360)
        except Exception:
            bub = _empty("RQ1c — insufficient data")

    # ═════════════════════════════════════════════════════════════════════════
    # RQ2 charts — weather × carrier quality
    # Source: rq2_weather_carrier (M1 + M2 + M3 grouped)
    # ═════════════════════════════════════════════════════════════════════════
    if ETL_RQ2.empty:
        hm = _empty(); wbox = _empty(); imc_ln = _empty()
    else:
        y0, y1 = int(yr_range[0]), int(yr_range[1])
        d2 = ETL_RQ2[ETL_RQ2["year"].between(y0, y1)].copy() if "year" in ETL_RQ2.columns else ETL_RQ2.copy()
        d2_comm = d2[d2["broad_carrier_tier"].isin(["major", "regional"])].copy() if "broad_carrier_tier" in d2.columns else d2.copy()

        # Heatmap A: weather_category × carrier_tier → mean_severity
        try:
            pivot = (d2_comm.groupby(["weather_category", "broad_carrier_tier"])["mean_severity"]
                     .mean().unstack(fill_value=float("nan")))
            pivot = pivot.reindex([w for w in WEATHER_ORDER_ETL if w in pivot.index])
            hm = px.imshow(
                pivot, color_continuous_scale="Reds", zmin=0, zmax=3,
                aspect="auto", text_auto=".2f",
                labels={"x": "Carrier Tier (M2)", "y": "Weather (M3)", "color": "Mean Severity"},
                title=f"RQ2a — Mean NTSB severity (M1) by weather (M3) × carrier tier (M2) | {y0}–{y1}",
            )
            hm.update_layout(template=TEMPLATE, height=400)
        except Exception:
            hm = _empty("RQ2a — pivot failed (check rq2_weather_carrier)")

        # Box B: windspeed (M3) by NTSB severity tier — from INCIDENT_ENRICHED
        try:
            wx_src = INCIDENT_ENRICHED if not INCIDENT_ENRICHED.empty else INCIDENTS_WEATHER
            if not wx_src.empty and "windspeed_10m" in wx_src.columns and "severity_score" in wx_src.columns:
                wx = wx_src[wx_src["windspeed_10m"].notna() &
                            wx_src["severity_score"].notna()].copy()
                if "year" in wx.columns:
                    wx = wx[wx["year"].between(y0, y1)]
                wx["sev_label"] = pd.to_numeric(wx["severity_score"], errors="coerce").map(
                    {0: "None/Incident", 1: "Minor", 2: "Serious", 3: "Fatal"})
                wbox = px.box(
                    wx, x="sev_label", y="windspeed_10m",
                    color="sev_label",
                    color_discrete_sequence=["#16A34A", "#F59E0B", "#EA580C", "#DC2626"],
                    points=False,
                    category_orders={"sev_label": ["None/Incident", "Minor", "Serious", "Fatal"]},
                    labels={"sev_label": "NTSB Severity (M1)",
                            "windspeed_10m": "Wind Speed — km/h (Open-Meteo M3)"},
                    title=f"RQ2b — Measured wind speed (M3) by NTSB severity tier (M1) | {y0}–{y1}",
                    template=TEMPLATE,
                )
                wbox.update_layout(showlegend=False, height=400)
            else:
                wbox = _empty("RQ2b — No wind speed data (M3 enrichment required)")
        except Exception:
            wbox = _empty("RQ2b — wind/severity data unavailable")

        # IMC trend C: proportion of incidents in IMC per carrier tier per year
        try:
            imc_d = (d2.groupby(["year", "broad_carrier_tier"])
                     .agg(imc_rate=("imc_rate", "mean"))
                     .reset_index())
            imc_d = imc_d[imc_d["broad_carrier_tier"].isin(
                ["major", "regional", "general_aviation"])]
            imc_ln = px.line(
                imc_d, x="year", y="imc_rate", color="broad_carrier_tier",
                color_discrete_map=TIER_COLORS, markers=False,
                labels={"imc_rate": "Proportion in IMC (M1+M3)",
                        "year": "Year", "broad_carrier_tier": "Carrier Tier"},
                title=f"RQ2c — IMC proportion by carrier tier (M1+M3) | {y0}–{y1}",
                template=TEMPLATE,
            )
            imc_ln.update_layout(
                legend=dict(orientation="h", y=-0.18), height=340,
                yaxis_tickformat=".0%",
            )
        except Exception:
            imc_ln = _empty("RQ2c — IMC trend unavailable")

    # ═════════════════════════════════════════════════════════════════════════
    # RQ3 charts — state safety drivers
    # Source: rq3_state_risk (M1 + M2 + M3 state aggregation)
    # ═════════════════════════════════════════════════════════════════════════
    if ETL_RQ3.empty:
        choro = _empty(); sc3 = _empty(); bar3 = _empty()
    else:
        mlabels = {
            "fatal_rate": "Fatal Rate", "mean_severity": "Mean Severity",
            "ga_pct": "GA Proportion", "imc_pct": "IMC Proportion",
            "mean_windspeed": "Mean Wind Speed (km/h)",
        }
        mlabel = mlabels.get(map_metric, map_metric)
        d3 = ETL_RQ3[ETL_RQ3[map_metric].notna()].copy() if map_metric in ETL_RQ3.columns else ETL_RQ3.copy()

        # Choropleth A
        try:
            scale = 100 if map_metric in CHORO_SCALE_COLS else 1
            d3["z_val"] = d3[map_metric] * scale
            z_label = f"{mlabel} (%)" if scale == 100 else mlabel

            def _col(col, mul=1):
                return (d3[col] * mul).fillna(0).round(1) if col in d3.columns else pd.Series([0] * len(d3))

            import numpy as np_inner
            cdata = np_inner.column_stack([
                _col("total_incidents").values,
                _col("fatal_rate", mul=100).values,
                _col("ga_pct",     mul=100).values,
                d3.get("dominant_weather", pd.Series(["N/A"] * len(d3))).fillna("N/A").values,
            ])
            choro = go.Figure(go.Choropleth(
                locations=d3["state"], z=d3["z_val"],
                locationmode="USA-states", colorscale="Reds",
                colorbar_title=z_label,
                customdata=cdata,
                hovertemplate=(
                    "<b>%{location}</b><br>"
                    f"<b>{mlabel}:</b> %{{z:.2f}}<br>"
                    "<b>Incidents (M1):</b> %{customdata[0]:,}<br>"
                    "<b>Fatal rate (M1):</b> %{customdata[1]}%<br>"
                    "<b>GA proportion (M1):</b> %{customdata[2]}%<br>"
                    "<b>Dominant weather (M3):</b> %{customdata[3]}<br>"
                    "<extra></extra>"
                ),
            ))
            choro.update_layout(
                geo=dict(scope="usa", showlakes=True, lakecolor="white"),
                title=f"RQ3a — {mlabel} by US State (M1+M3) — hover for full profile",
                template=TEMPLATE, height=440,
                margin=dict(l=0, r=0, t=45, b=0),
            )
        except Exception as exc:
            choro = _empty(f"RQ3a — choropleth error: {exc}")

        # Scatter B: mean windspeed (M3) vs fatal rate (M1), colour = GA%
        try:
            ds = ETL_RQ3.dropna(subset=["mean_windspeed", "fatal_rate"]).copy() if                  "mean_windspeed" in ETL_RQ3.columns and "fatal_rate" in ETL_RQ3.columns                  else pd.DataFrame()
            if not ds.empty:
                ds["fatal_pct"]  = ds["fatal_rate"] * 100
                ds["ga_pct_pct"] = ds.get("ga_pct", pd.Series(0, index=ds.index)) * 100
                sc3 = px.scatter(
                    ds, x="mean_windspeed", y="fatal_pct",
                    size="total_incidents", color="ga_pct_pct",
                    color_continuous_scale="RdYlGn_r",
                    hover_name="state",
                    hover_data={"mean_windspeed": ":.1f", "fatal_pct": ":.1f",
                                "ga_pct_pct": ":.0f", "total_incidents": True},
                    labels={"mean_windspeed": "Mean wind at incident — km/h (M3)",
                            "fatal_pct": "Fatal rate % (M1)",
                            "ga_pct_pct": "GA %"},
                    title="RQ3b — Wind (M3) vs fatal rate (M1), colour = GA% (M1)",
                    template=TEMPLATE,
                )
                sc3.update_layout(coloraxis_colorbar=dict(title="GA %"), height=440)
            else:
                sc3 = _empty("RQ3b — wind/fatal rate data missing in rq3_state_risk")
        except Exception:
            sc3 = _empty("RQ3b — scatter unavailable")

        # Stacked bar C: top-15 worst states by fatal rate, decomposed by weather
        try:
            worst15 = (ETL_RQ3.dropna(subset=["fatal_rate"])
                       .sort_values("fatal_rate", ascending=False)
                       .head(15)["state"].tolist()) if "fatal_rate" in ETL_RQ3.columns else []
            # Use INCIDENT_ENRICHED for per-incident weather breakdown
            enr = INCIDENT_ENRICHED if not INCIDENT_ENRICHED.empty else INCIDENTS_WEATHER
            if worst15 and not enr.empty and "weather_category" in enr.columns and "state" in enr.columns:
                sw = (enr[enr["state"].isin(worst15)]
                      .groupby(["state", "weather_category"])
                      .size().reset_index(name="n"))
                sw["prop"] = sw["n"] / sw.groupby("state")["n"].transform("sum") * 100
                sw["state"] = pd.Categorical(sw["state"], categories=worst15, ordered=True)
                sw = sw.sort_values("state")
                bar3 = px.bar(
                    sw, x="state", y="prop", color="weather_category",
                    color_discrete_map=WEATHER_COLORS_ETL,
                    category_orders={"state": worst15,
                                     "weather_category": WEATHER_ORDER_ETL},
                    labels={"prop": "% of state incidents (M1+M3)",
                            "state": "State — worst → best fatal rate",
                            "weather_category": "Weather (M3)"},
                    title="RQ3c — Weather decomposition for top-15 highest-fatal-rate states (M1+M3)",
                    template=TEMPLATE,
                )
                bar3.update_layout(
                    yaxis_ticksuffix="%", height=360,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
            elif worst15 and "weather_condition" in enr.columns:
                # Fallback: use NTSB weather_condition if M3 weather_category unavailable
                sw = (enr[enr["state"].isin(worst15)]
                      .groupby(["state", "weather_condition"])
                      .size().reset_index(name="n"))
                sw["prop"] = sw["n"] / sw.groupby("state")["n"].transform("sum") * 100
                sw["state"] = pd.Categorical(sw["state"], categories=worst15, ordered=True)
                bar3 = px.bar(
                    sw.sort_values("state"), x="state", y="prop",
                    color="weather_condition",
                    labels={"prop": "% of state incidents", "state": "State",
                            "weather_condition": "Weather (NTSB M1)"},
                    title="RQ3c — Weather breakdown (using NTSB condition — M3 weather unavailable)",
                    template=TEMPLATE,
                )
                bar3.update_layout(yaxis_ticksuffix="%", height=360,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            else:
                bar3 = _empty("RQ3c — run etl_pipeline.py to populate rq3_state_risk")
        except Exception as exc:
            bar3 = _empty(f"RQ3c — {exc}")

    # ── Insight text ──────────────────────────────────────────────────────────
    insight_parts = []
    if not ETL_RQ1.empty and "cancel_rate" in ETL_RQ1.columns and "fatal_rate" in ETL_RQ1.columns:
        corr_val = ETL_RQ1[["cancel_rate", "fatal_rate"]].dropna().corr().iloc[0, 1]
        direction = "positive" if corr_val > 0 else "negative"
        insight_parts.append(
            f"🔗 RQ1: Pearson correlation between cancellation rate (M2) and NTSB fatal rate (M1) "
            f"across {len(ETL_RQ1)} carriers is {corr_val:+.2f} ({direction} — "
            f"{'worse BTS performance → more fatal accidents' if corr_val > 0 else 'no clear BTS–NTSB link'})."
        )
    if not ETL_RQ3.empty and "fatal_rate" in ETL_RQ3.columns and "ga_pct" in ETL_RQ3.columns:
        worst = ETL_RQ3.nlargest(1, "fatal_rate")
        if not worst.empty:
            st = worst.iloc[0]
            insight_parts.append(
                f"🗺️ RQ3: Highest fatal rate state is {st.get('state', 'N/A')} "
                f"({st['fatal_rate']*100:.1f}%), "
                f"GA proportion {st.get('ga_pct', 0)*100:.0f}%, "
                f"dominant weather: {st.get('dominant_weather', 'N/A')}."
            )
    if not insight_parts:
        insight_parts = ["Run etl_pipeline.py to populate the cross-dataset analytical tables."]

    insight = " | ".join(insight_parts)
    return scat, bar_rq1, bub, hm, wbox, imc_ln, choro, sc3, bar3, insight


# ─────────────────────────────────────────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[dashboard] Starting on http://localhost:8050")
    app.run(debug=True, port=8050)
