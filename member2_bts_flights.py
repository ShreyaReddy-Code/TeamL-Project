"""
member2_bts_flights.py
======================

Member 2 — Bureau of Transportation Statistics (BTS) Reporting Carrier
On-Time Performance.

Pipeline:

    [1] Download monthly CSV-zips from BTS Transtats   (4 years × 4 quarters)
    [2] Concatenate, sample 10 % of rows, write data/bts_flights.csv
    [3] Insert 10 % sample into MongoDB collection "flights"
    [4] Pull back into pandas, clean, derive carrier_tier / route / etc.
    [5] Produce 4 plotly visualisations
    [6] Persist carrier_stats and flights_sample to Postgres

Run standalone: `python member2_bts_flights.py`
"""

from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import pymongo
from tqdm import tqdm

from config import (
    MAJOR_US_CARRIERS,
    REGIONAL_KEYWORDS,
    mongo_db,
    neon_engine,
)

# -----------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
COMBINED_CSV = DATA_DIR / "bts_flights.csv"

YEARS = [2018, 2019, 2022, 2023]
MONTHS = [1, 4, 7, 10]   # quarterly sampling

BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

KEEP_COLS = [
    "FlightDate", "Reporting_Airline", "Origin", "Dest",
    "DepDelay", "ArrDelay", "Cancelled", "Diverted",
    "Distance", "AirTime",
]

MONGO_COLLECTION = "flights"

# Fraction of the full combined dataset to retain.
# Set to 0.10 to keep exactly 10 % (reduces MongoDB Atlas storage
# and speeds up every downstream step proportionally).
SAMPLE_FRACTION = 0.10
RANDOM_SEED = 42


# =================================================================
# SECTION 1 — DATA ACQUISITION
# =================================================================
def _download_one(year: int, month: int) -> Optional[pd.DataFrame]:
    """Download one BTS monthly file and return its rows as a DataFrame."""
    url = BTS_URL.format(year=year, month=month)
    try:
        resp = requests.get(url, stream=True, timeout=120,
                            headers={"User-Agent": "ProjectA-Aviation/1.0"})
        resp.raise_for_status()
    except Exception as exc:
        print(f"[member2]   FAIL {year}-{month:02d}: {exc}")
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_member = next(
                (n for n in zf.namelist() if n.lower().endswith(".csv")), None
            )
            if csv_member is None:
                print(f"[member2]   {year}-{month:02d}: no CSV in zip")
                return None
            with zf.open(csv_member) as fh:
                df = pd.read_csv(fh, low_memory=False)
    except Exception as exc:
        print(f"[member2]   {year}-{month:02d} parse error: {exc}")
        return None

    # Some monthly files use slightly different column names; align them.
    rename = {
        "FL_DATE": "FlightDate",
        "OP_UNIQUE_CARRIER": "Reporting_Airline",
        "ORIGIN": "Origin",
        "DEST": "Dest",
        "DEP_DELAY": "DepDelay",
        "ARR_DELAY": "ArrDelay",
        "CANCELLED": "Cancelled",
        "DIVERTED": "Diverted",
        "DISTANCE": "Distance",
        "AIR_TIME": "AirTime",
    }
    df = df.rename(columns=rename)
    cols_present = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols_present].copy()
    print(f"[member2]   OK   {year}-{month:02d}: {len(df):,} rows")
    return df


def acquire_bts_data() -> pd.DataFrame:
    """
    Download all (year, month) combos, concatenate, take a 10 % random
    sample, and persist that sample to disk.

    Sampling at this stage means every downstream step — MongoDB ingest,
    Postgres writes, and all visualisations — operates on the reduced set.
    The sample is stratified implicitly because rows come from multiple
    years and months, preserving temporal diversity.

    If all downloads fail raise with a clear message.
    """
    print("\n[member2] === SECTION 1 — BTS DATA ACQUISITION ===")
    print(f"[member2] Years: {YEARS}, months: {MONTHS} (quarterly sampling)")
    print(f"[member2] Will retain {SAMPLE_FRACTION:.0%} of combined rows")

    frames: list[pd.DataFrame] = []
    for year in YEARS:
        for month in MONTHS:
            df = _download_one(year, month)
            if df is not None and not df.empty:
                frames.append(df)

    if not frames:
        raise RuntimeError(
            "All BTS downloads failed. Check network access to "
            "transtats.bts.gov, then rerun."
        )

    combined = pd.concat(frames, ignore_index=True)
    full_size = len(combined)
    print(f"[member2] Total rows before sampling: {full_size:,}")

    # ── 10 % random sample (reproducible via RANDOM_SEED) ──────────
    combined = combined.sample(
        frac=SAMPLE_FRACTION,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)
    print(
        f"[member2] After {SAMPLE_FRACTION:.0%} sample: "
        f"{len(combined):,} rows  (dropped {full_size - len(combined):,})"
    )
    # ───────────────────────────────────────────────────────────────

    combined.to_csv(COMBINED_CSV, index=False)
    print(f"[member2] Wrote sampled CSV -> {COMBINED_CSV}")
    return combined


# =================================================================
# SECTION 2 — MONGO INGESTION
# =================================================================
def insert_to_mongo(df: pd.DataFrame) -> int:
    print(f"\n[member2] === SECTION 2 — MONGO INGEST ({len(df):,} rows) ===")
    db = mongo_db()
    coll = db[MONGO_COLLECTION]
    coll.drop()

    BATCH = 5000
    total = 0
    # Convert to list-of-dicts in chunks to avoid loading entire frame as
    # one giant Python list when the DataFrame is huge.
    try:
        for start in tqdm(range(0, len(df), BATCH), desc="mongo insert"):
            chunk = df.iloc[start:start + BATCH]
            records = chunk.to_dict("records")
            coll.insert_many(records, ordered=False)
            total += len(records)
    except pymongo.errors.OperationFailure as exc:
        # Code 8000 is the Atlas space quota error.
        if exc.code == 8000:
            print(f"\n[member2] WARNING: MongoDB space quota reached ({exc.details.get('errmsg')})")
            print(f"[member2] Proceeding with the {total:,} rows already inserted.")
        else:
            raise

    try:
        coll.create_index("FlightDate")
        coll.create_index("Reporting_Airline")
        coll.create_index("Origin")
        coll.create_index("Dest")
    except pymongo.errors.OperationFailure:
        print("[member2] WARNING: Could not create all indexes due to space quota.")

    count = coll.count_documents({})
    print(f"[member2] Inserted {count:,} flight documents")
    return count


# =================================================================
# SECTION 3 — PREPROCESSING & CLEANING
# =================================================================
# A small inline mapping from BTS 2-letter carrier codes → friendly name
# so we can apply the same MAJOR_US_CARRIERS list as member 1.
CARRIER_CODE_TO_NAME = {
    "AA": "AMERICAN AIRLINES",
    "DL": "DELTA AIR LINES",
    "UA": "UNITED AIRLINES",
    "WN": "SOUTHWEST AIRLINES",
    "AS": "ALASKA AIRLINES",
    "B6": "JETBLUE AIRWAYS",
    "NK": "SPIRIT AIRLINES",
    "F9": "FRONTIER AIRLINES",
    "HA": "HAWAIIAN AIRLINES",
    "G4": "ALLEGIANT AIR",
    # Common regionals — name derived so REGIONAL_KEYWORDS catches them.
    "OO": "SKYWEST EXPRESS",
    "MQ": "ENVOY AIR REGIONAL",
    "YX": "REPUBLIC REGIONAL",
    "OH": "PSA REGIONAL",
    "9E": "ENDEAVOR REGIONAL",
    "EV": "EXPRESSJET REGIONAL",
    "YV": "MESA REGIONAL",
    "ZW": "AIR WISCONSIN REGIONAL",
}


def _carrier_tier_from_code(code: str) -> str:
    name = CARRIER_CODE_TO_NAME.get(code, code or "UNKNOWN")
    up = name.upper()
    if up in MAJOR_US_CARRIERS:
        return "major"
    if any(kw in up for kw in REGIONAL_KEYWORDS):
        return "regional"
    return "general_aviation"


def load_and_clean() -> pd.DataFrame:
    print("\n[member2] === SECTION 3 — LOAD FROM MONGO & CLEAN ===")
    coll = mongo_db()[MONGO_COLLECTION]
    cursor = coll.find({}, {"_id": 0})
    df = pd.DataFrame(list(cursor))
    print(f"[member2] Loaded shape from Mongo: {df.shape}")

    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    for col in ("DepDelay", "ArrDelay", "Cancelled", "Diverted",
                "Distance", "AirTime"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clip extreme delays — almost certainly data-entry errors.
    df["DepDelay"] = df["DepDelay"].clip(lower=-60, upper=600)
    df["ArrDelay"] = df["ArrDelay"].clip(lower=-60, upper=600)

    df["is_cancelled"] = df["Cancelled"].fillna(0).astype(int).astype(bool)
    df["is_diverted"] = df["Diverted"].fillna(0).astype(int).astype(bool)

    df["route"] = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
    df["carrier_tier"] = df["Reporting_Airline"].apply(_carrier_tier_from_code)

    # Per-carrier route complexity: distinct routes operated.
    complexity = (
        df.groupby("Reporting_Airline")["route"].nunique()
        .rename("route_complexity_score").reset_index()
    )
    df = df.merge(complexity, on="Reporting_Airline", how="left")

    df["month"] = df["FlightDate"].dt.month

    print(f"[member2] After cleaning shape: {df.shape}")
    print("[member2] Numeric describe():")
    print(df[["DepDelay", "ArrDelay", "Distance", "AirTime"]].describe().to_string())
    return df


# =================================================================
# SECTION 4 — ANALYSIS & VISUALISATIONS
# =================================================================
TEMPLATE = "plotly_white"
TIER_COLORS = {
    "major": "#1f77b4",
    "regional": "#d62728",
    "general_aviation": "#7f7f7f",
}


def _save_fig(fig, n: int) -> Path:
    out = DATA_DIR / f"member2_viz{n}.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"[member2] Saved visualisation: {out}")
    return out


def viz1_arrdelay_box(df: pd.DataFrame) -> None:
    print("\n[member2] VIZ 1 — Arr-delay distribution by carrier tier")
    fig = px.box(
        df, x="carrier_tier", y="ArrDelay", color="carrier_tier",
        color_discrete_map=TIER_COLORS, points=False,
        category_orders={"carrier_tier": ["major", "regional", "general_aviation"]},
    )
    fig.update_yaxes(range=[-30, 120])
    fig.update_layout(
        title="Arrival delay distribution by carrier tier",
        template=TEMPLATE, xaxis_title="Carrier tier",
        yaxis_title="Arrival delay (minutes)",
        showlegend=False,
    )
    medians = df.groupby("carrier_tier")["ArrDelay"].median()
    if len(medians) > 1:
        worst = medians.idxmax()
        fig.add_annotation(
            x=worst, y=medians.max(), text=f"Highest median delay: {worst}",
            showarrow=True, arrowhead=2, ay=-40,
        )
    fig.show()
    _save_fig(fig, 1)


def viz2_cancellation_rate(df: pd.DataFrame) -> None:
    print("\n[member2] VIZ 2 — Cancellation rate by carrier")
    top_carriers = (
        df["Reporting_Airline"].value_counts().head(20).index.tolist()
    )
    sub = df[df["Reporting_Airline"].isin(top_carriers)].copy()

    agg = sub.groupby("Reporting_Airline").agg(
        total_flights=("FlightDate", "count"),
        cancelled=("is_cancelled", "sum"),
    ).reset_index()
    agg["cancellation_rate"] = agg["cancelled"] / agg["total_flights"] * 100
    agg["carrier_tier"] = agg["Reporting_Airline"].apply(_carrier_tier_from_code)
    agg = agg.sort_values("cancellation_rate", ascending=False)

    fig = px.bar(
        agg, x="Reporting_Airline", y="cancellation_rate",
        color="carrier_tier", color_discrete_map=TIER_COLORS,
        text=agg["cancellation_rate"].round(2),
    )
    fig.update_layout(
        title="Cancellation rate by carrier (2018–2023)",
        template=TEMPLATE,
        xaxis_title="Carrier", yaxis_title="Cancellation rate (%)",
    )
    if not agg.empty:
        worst = agg.iloc[0]
        fig.add_annotation(
            x=worst["Reporting_Airline"], y=worst["cancellation_rate"],
            text=f"Highest: {worst['Reporting_Airline']} ({worst['cancellation_rate']:.2f}%)",
            showarrow=True, arrowhead=2, ay=-40,
        )
    fig.show()
    _save_fig(fig, 2)


def viz3_complexity_vs_delay(df: pd.DataFrame) -> None:
    print("\n[member2] VIZ 3 — Route complexity vs delay")
    agg = df.groupby("Reporting_Airline").agg(
        route_complexity_score=("route_complexity_score", "max"),
        mean_arr_delay=("ArrDelay", "mean"),
        total_flights=("FlightDate", "count"),
    ).reset_index()
    agg["carrier_tier"] = agg["Reporting_Airline"].apply(_carrier_tier_from_code)

    fig = px.scatter(
        agg, x="route_complexity_score", y="mean_arr_delay",
        size="total_flights", color="carrier_tier",
        color_discrete_map=TIER_COLORS,
        hover_name="Reporting_Airline", size_max=50,
    )
    # Trendline using numpy polyfit (no statsmodels dependency).
    if len(agg) >= 2:
        x = agg["route_complexity_score"].to_numpy(dtype=float)
        y = agg["mean_arr_delay"].to_numpy(dtype=float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() >= 2:
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            xs = np.linspace(x[mask].min(), x[mask].max(), 50)
            ys = slope * xs + intercept
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name="Trend",
                line=dict(color="black", dash="dash"),
            ))
            fig.add_annotation(
                x=xs[-1], y=ys[-1],
                text=f"Slope: {slope:.3f} min / route",
                showarrow=False, xanchor="right", yanchor="bottom",
                font=dict(color="black"),
            )
    fig.update_layout(
        title="Route complexity vs. average arrival delay by carrier",
        template=TEMPLATE,
        xaxis_title="Distinct routes operated (complexity score)",
        yaxis_title="Mean arrival delay (minutes)",
    )
    fig.show()
    _save_fig(fig, 3)


def viz4_monthly_heatmap(df: pd.DataFrame) -> None:
    print("\n[member2] VIZ 4 — Monthly performance heatmap")
    top_carriers = df["Reporting_Airline"].value_counts().head(15).index
    sub = df[df["Reporting_Airline"].isin(top_carriers)]
    pivot = (
        sub.pivot_table(
            index="Reporting_Airline", columns="month",
            values="ArrDelay", aggfunc="mean",
        )
        .reindex(columns=range(1, 13))
    )
    fig = px.imshow(
        pivot, color_continuous_scale="RdBu_r", aspect="auto",
        labels=dict(x="Month", y="Carrier", color="Mean arr delay"),
        x=[str(m) for m in pivot.columns], y=pivot.index,
    )
    fig.update_layout(
        title="Mean arrival delay heatmap: carrier × month",
        template=TEMPLATE,
    )
    if not pivot.empty:
        worst_month = int(pivot.mean().idxmax())
        fig.add_annotation(
            text=f"Industry worst month: {worst_month}",
            xref="paper", yref="paper", x=0.99, y=1.08,
            showarrow=False, xanchor="right",
        )
    fig.show()
    _save_fig(fig, 4)


# =================================================================
# SECTION 5 — STORE TO POSTGRESQL
# =================================================================
def write_to_postgres(df: pd.DataFrame) -> None:
    print("\n[member2] === SECTION 5 — WRITE TO POSTGRES ===")
    eng = neon_engine()

    carrier_stats = df.groupby("Reporting_Airline").agg(
        total_flights=("FlightDate", "count"),
        mean_dep_delay=("DepDelay", "mean"),
        mean_arr_delay=("ArrDelay", "mean"),
        cancel_rate=("is_cancelled", "mean"),
        divert_rate=("is_diverted", "mean"),
        route_complexity_score=("route", "nunique"),
    ).reset_index()
    carrier_stats["carrier_tier"] = (
        carrier_stats["Reporting_Airline"].apply(_carrier_tier_from_code)
    )
    # Map codes (AA) to full names (AMERICAN AIRLINES) for joining with M1
    carrier_stats["Reporting_Airline"] = carrier_stats["Reporting_Airline"].map(CARRIER_CODE_TO_NAME).fillna(carrier_stats["Reporting_Airline"])

    # Drop with CASCADE to handle dependent views (e.g. incident_enriched)
    with eng.begin() as conn:
        conn.execute(text("DROP VIEW IF EXISTS incident_enriched"))
        conn.execute(text("DROP TABLE IF EXISTS carrier_stats"))

    carrier_stats.to_sql(
        "carrier_stats", eng, if_exists="replace", index=False
    )
    print(f"[member2] Wrote {len(carrier_stats)} rows to carrier_stats")

    # The full df is already the 10 % sample, so write it all to Postgres.
    # Cap at 500 000 rows as a safety net for very large SAMPLE_FRACTION values.
    sample = df if len(df) <= 500_000 else df.sample(n=500_000, random_state=RANDOM_SEED)
    sample = sample.copy()
    # Make sure FlightDate is a real timestamp (not Mongo ObjectId garbage).
    sample["FlightDate"] = pd.to_datetime(sample["FlightDate"])
    sample.to_sql(
        "flights_sample", eng, if_exists="replace", index=False,
        chunksize=10_000, method="multi",
    )
    print(f"[member2] Wrote {len(sample):,} rows to flights_sample")


# =================================================================
# Orchestration
# =================================================================
def run() -> None:
    t0 = time.time()
    raw = acquire_bts_data()
    insert_to_mongo(raw)
    df = load_and_clean()
    viz1_arrdelay_box(df)
    viz2_cancellation_rate(df)
    viz3_complexity_vs_delay(df)
    viz4_monthly_heatmap(df)
    write_to_postgres(df)
    print(f"\n[member2] DONE in {time.time() - t0:,.1f}s")


if __name__ == "__main__":
    run()
