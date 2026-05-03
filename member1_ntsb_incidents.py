"""
member1_ntsb_incidents_enhanced.py
===================================

Member 1 — NTSB aviation accident database  (ENHANCED)

Sections:
    [1] Download / acquire NTSB AVALL data (CSV → XML)
    [2] Parse XML and ingest to MongoDB
    [3] Load from Mongo, clean in pandas
    [4] Eight enhanced visualisations covering:
            VIZ 1  — Severity trend by decade (stacked bar + fatal-rate line)
            VIZ 2  — Phase of flight: incidents vs fatalities (grouped bar)
            VIZ 3  — IMC vs VMC over time with regulatory milestones
            VIZ 4  — Geographic incident density map
            VIZ 5  — Aircraft category risk profile (bubble chart)   [NEW]
            VIZ 6  — Seasonal patterns: month × phase heatmap         [NEW]
            VIZ 7  — Survivability index by engine type               [NEW]
            VIZ 8  — Amateur-built vs certified: 40-year safety gap   [NEW]
    [5] Persist cleaned DataFrame to PostgreSQL/Neon (incidents_clean)

Research questions answered:
    RQ-A  Has aviation safety improved across every flight phase equally?
           → VIZ 1 (decade trend) + VIZ 2 (phase breakdown)
    RQ-B  Does instrument meteorological condition (IMC) explain fatality
          spikes, or are there structural / regulatory drivers?
           → VIZ 3 (IMC/VMC timeline)
    RQ-C  Which aircraft categories carry the highest per-incident
          fatality burden?
           → VIZ 5 (category bubble chart)
    RQ-D  Is there a seasonal or time-of-day pattern to aviation accidents?
           → VIZ 6 (month × phase heatmap)
    RQ-E  Do engine configuration and type affect survivability?
           → VIZ 7 (engine-type survivability)
    RQ-F  Are amateur-built aircraft genuinely more dangerous, and has
          that gap narrowed with modern regulations?
           → VIZ 8 (amateur vs certified 40-year trend)

Run standalone:
    python member1_ntsb_incidents_enhanced.py
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from lxml import etree
from tqdm import tqdm

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from sqlalchemy import text

from config import (
    MAJOR_US_CARRIERS,
    REGIONAL_KEYWORDS,
    mongo_db,
    neon_engine,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths and constants
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

LOCAL_ZIP_CANDIDATES = [
    DATA_DIR / "avall.zip",
    DATA_DIR / "AVALL_CSV.zip",
    DATA_DIR / "AviationData.zip",
]
ZIP_PATH    = DATA_DIR / "AviationData.zip"
CSV_PATH    = DATA_DIR / "AviationData.csv"
XML_PATH    = DATA_DIR / "AviationData.xml"

# Correct NTSB direct-download URL (confirmed April 2026)
NTSB_PRIMARY_URL = (
    "https://data.ntsb.gov/avdata/FileDirectory/DownloadFile?"
    "fileID=C%3A%5Cavdata%5Cavall.zip"
)
NTSB_FALLBACK_URL = "https://data.ntsb.gov/carol-main-public/landing-page"

MONGO_COLLECTION = "incidents"
PG_TABLE         = "incidents_clean"

SEVERITY_MAP = {
    "Non-Fatal": 0,
    "Incident":  0,
    "Minor":     1,
    "Serious":   2,
    "Fatal":     3,
}

# Visualisation style constants
TEMPLATE        = "plotly_white"
COLOR_BLUE      = "#2563EB"
COLOR_RED       = "#DC2626"
COLOR_AMBER     = "#F59E0B"
COLOR_GREEN     = "#16A34A"
COLOR_PURPLE    = "#7C3AED"
COLOR_TEAL      = "#0D9488"
SEVERITY_COLORS = {0: "#22c55e", 1: "#fbbf24", 2: "#f97316", 3: "#ef4444"}

PHASE_ORDER = ["Takeoff", "Climb", "Cruise", "Descent",
               "Approach", "Landing", "Taxi", "Unknown"]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA ACQUISITION
# ═════════════════════════════════════════════════════════════════════════════

def _download_with_progress(url: str, dest: Path, label: str) -> bool:
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[member1] {label} failed: {exc}")
        return False

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=label
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                fh.write(chunk)
                bar.update(len(chunk))
    return True


def acquire_ntsb_data() -> pd.DataFrame:
    print("\n[member1] === SECTION 1 — DATA ACQUISITION ===")

    local_zip: Optional[Path] = next(
        (p for p in LOCAL_ZIP_CANDIDATES if p.exists() and p.stat().st_size > 0),
        None,
    )

    if local_zip is not None:
        print(f"[member1] Found local zip: {local_zip.name} "
              f"({local_zip.stat().st_size / 1_000_000:,.1f} MB) — skipping download")
        zip_to_use = local_zip
    else:
        print("[member1] No local zip found — attempting download...")
        success = _download_with_progress(NTSB_PRIMARY_URL, ZIP_PATH, "NTSB AVALL")
        if not success:
            raise RuntimeError(
                f"Download failed. Manually place avall.zip in {DATA_DIR} and rerun."
            )
        zip_to_use = ZIP_PATH

    with zipfile.ZipFile(zip_to_use) as zf:
        members   = zf.namelist()
        csv_member = next((n for n in members if n.lower().endswith(".csv")), None)
        mdb_member = next((n for n in members if n.lower().endswith(".mdb")), None)

        if csv_member:
            with zf.open(csv_member) as src, open(CSV_PATH, "wb") as dst:
                dst.write(src.read())
            print(f"[member1] Extracted {CSV_PATH.name}")
        elif mdb_member:
            print(f"[member1] MDB detected — converting via mdbtools...")
            with tempfile.TemporaryDirectory() as td:
                mdb_local = Path(td) / Path(mdb_member).name
                with zf.open(mdb_member) as src, open(mdb_local, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                _mdb_to_aviation_csv(mdb_local, CSV_PATH)
        else:
            raise RuntimeError(f"No .csv or .mdb in zip: {members}")

    df = None
    for sep, enc in [("|", "utf-8"), ("|", "latin-1"), (",", "utf-8"), (",", "latin-1")]:
        try:
            df = pd.read_csv(CSV_PATH, sep=sep, encoding=enc,
                             low_memory=False, on_bad_lines="skip")
            if df.shape[1] > 5:
                print(f"[member1] Parsed CSV: {df.shape[0]:,} rows × {df.shape[1]} cols")
                break
        except Exception:
            continue
    if df is None or df.shape[1] <= 5:
        raise RuntimeError("Could not parse AviationData.csv")

    print(f"[member1] Converting {len(df):,} rows to XML (semi-structured requirement)...")
    root = etree.Element("AviationDataSet")
    cols = [str(c) for c in df.columns]
    for _, row in tqdm(df.iterrows(), total=len(df), desc="rows→xml"):
        evt = etree.SubElement(root, "Event")
        for c in cols:
            child = etree.SubElement(evt, _xml_tag(c))
            val = row[c]
            child.text = "" if pd.isna(val) else str(val)
    etree.ElementTree(root).write(
        str(XML_PATH), pretty_print=True, xml_declaration=True, encoding="utf-8"
    )
    print(f"[member1] XML written: {XML_PATH.name}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MDB helpers (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
EVENTS_COLUMN_MAP = {
    "ev_id": "Event Id", "ntsb_no": "Accident Number",
    "ev_type": "Investigation Type", "ev_date": "Event Date",
    "ev_city": "Location", "ev_state": "_state", "ev_country": "Country",
    "latitude": "Latitude", "longitude": "Longitude",
    "ev_nr_apt_id": "Airport Code", "apt_name": "Airport Name",
    "ev_highest_injury": "Injury Severity",
    "inj_tot_f": "Total Fatal Injuries", "inj_tot_s": "Total Serious Injuries",
    "inj_tot_m": "Total Minor Injuries", "inj_tot_n": "Total Uninjured",
    "wx_cond_basic": "Weather Condition", "lchg_date": "Publication Date",
}
AIRCRAFT_COLUMN_MAP = {
    "ev_id": "Event Id", "acft_make": "Make", "acft_model": "Model",
    "acft_category": "Aircraft Category", "homebuilt": "Amateur Built",
    "num_eng": "Number of Engines", "oprtng_cert": "FAR Description",
    "oper_sched": "Schedule", "type_fly": "Purpose of flight",
    "oper_dba": "_oper_dba", "oper_indiv_name": "_oper_indiv", "oper_name": "_oper_name",
    "phase_flt_spec": "Broad phase of flight", "damage": "Aircraft damage",
}
INJURY_CODE_MAP  = {"FATL": "Fatal", "SERS": "Serious", "MINR": "Minor", "NONE": "Non-Fatal"}
INVEST_TYPE_MAP  = {"ACC": "Accident", "INC": "Incident"}


def _check_mdbtools():
    if shutil.which("mdb-export") is None:
        raise RuntimeError("mdbtools not found. brew install mdbtools or apt-get install mdbtools")


def _mdb_export_table(mdb_path: Path, table: str) -> pd.DataFrame:
    proc = subprocess.run(
        ["mdb-export", "-D", "%Y-%m-%d %H:%M:%S", str(mdb_path), table],
        check=True, capture_output=True,
    )
    return pd.read_csv(io.BytesIO(proc.stdout), low_memory=False)


def _mdb_to_aviation_csv(mdb_path: Path, csv_out: Path) -> None:
    _check_mdbtools()
    events   = _mdb_export_table(mdb_path, "events")
    aircraft = _mdb_export_table(mdb_path, "aircraft")
    e_keep = {k: v for k, v in EVENTS_COLUMN_MAP.items() if k in events.columns}
    a_keep = {k: v for k, v in AIRCRAFT_COLUMN_MAP.items() if k in aircraft.columns}
    e = events[list(e_keep)].rename(columns=e_keep)
    a = aircraft[list(a_keep)].rename(columns=a_keep)
    if "_state" in e.columns:
        # Keep the state code as a dedicated column 'State'
        e["State"] = e["_state"].fillna("").astype(str).str.strip()
        # Still merge it into Location for backward compatibility
        loc = e.get("Location", pd.Series("", index=e.index)).fillna("").astype(str)
        e["Location"] = (loc.str.strip() + ", " + e["State"]).str.strip(", ")
        e = e.drop(columns=["_state"])
    dba   = a.get("_oper_dba",    pd.Series("", index=a.index)).fillna("").astype(str).str.strip()
    indiv = a.get("_oper_indiv",  pd.Series("", index=a.index)).fillna("").astype(str).str.strip()
    name  = a.get("_oper_name",   pd.Series("", index=a.index)).fillna("").astype(str).str.strip()
    # Priority: DBA > Operator Name > Individual Name
    a["Air carrier"] = dba.where(dba.str.len() > 0, name.where(name.str.len() > 0, indiv))
    a = a.drop(columns=[c for c in ("_oper_dba", "_oper_indiv", "_oper_name") if c in a.columns])
    a_first = a.sort_values("Event Id").drop_duplicates("Event Id", keep="first")
    merged  = e.merge(a_first, on="Event Id", how="left")
    if "Injury Severity"     in merged.columns: merged["Injury Severity"]     = merged["Injury Severity"].map(INJURY_CODE_MAP).fillna(merged["Injury Severity"])
    if "Investigation Type"  in merged.columns: merged["Investigation Type"]  = merged["Investigation Type"].map(INVEST_TYPE_MAP).fillna(merged["Investigation Type"])
    for missing in ("Engine Type", "Registration Number", "Report Status"):
        if missing not in merged.columns: merged[missing] = ""
    merged.to_csv(csv_out, sep="|", index=False)


def _xml_tag(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or "Field"
    return ("f_" + cleaned) if cleaned[0].isdigit() else cleaned


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MONGO INGESTION
# ═════════════════════════════════════════════════════════════════════════════

def _safe_int(v) -> Optional[int]:
    try:
        return None if (v is None or v == "") else int(float(v))
    except (TypeError, ValueError):
        return None


def _safe_float(v) -> Optional[float]:
    try:
        return None if (v is None or v == "") else float(v)
    except (TypeError, ValueError):
        return None


def _safe_date(v) -> Optional[datetime]:
    if not v or (isinstance(v, float) and np.isnan(v)):
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(v).strip(), fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(v, errors="coerce").to_pydatetime()
    except Exception:
        return None


FIELD_CANDIDATES = {
    "event_id":               ["Event_Id", "EventId", "Event_ID"],
    "event_date":             ["Event_Date", "EventDate"],
    "airport_code":           ["Airport_Code", "AirportCode"],
    "airport_name":           ["Airport_Name", "AirportName"],
    "city":                   ["Location", "City"],
    "state":                  ["State"],
    "country":                ["Country"],
    "latitude":               ["Latitude"],
    "longitude":              ["Longitude"],
    "injury_severity":        ["Injury_Severity", "Injury_severity", "InjurySeverity"],
    "aircraft_damage":        ["Aircraft_Damage", "Aircraft_damage", "AircraftDamage"],
    "aircraft_category":      ["Aircraft_Category", "Aircraft_category", "AircraftCategory"],
    "make":                   ["Make"],
    "model":                  ["Model"],
    "amateur_built":          ["Amateur_Built", "Amateur_built", "AmateurBuilt"],
    "number_of_engines":      ["Number_of_Engines", "Number_of_engines", "NumberofEngines"],
    "engine_type":            ["Engine_Type", "Engine_type", "EngineType"],
    "far_description":        ["FAR_Description", "FAR_description", "FARDescription"],
    "schedule":               ["Schedule"],
    "purpose_of_flight":      ["Purpose_of_flight", "Purpose_of_Flight", "PurposeofFlight"],
    "air_carrier":            ["Air_carrier", "Air_Carrier", "AirCarrier"],
    "total_fatal_injuries":   ["Total_Fatal_Injuries", "Total_Fatal_injuries", "TotalFatalInjuries"],
    "total_serious_injuries":  ["Total_Serious_Injuries", "Total_Serious_injuries", "TotalSeriousInjuries"],
    "total_minor_injuries":   ["Total_Minor_Injuries", "Total_Minor_injuries", "TotalMinorInjuries"],
    "total_uninjured":        ["Total_Uninjured", "Total_Uninjured", "TotalUninjured"],
    "weather_condition":      ["Weather_Condition", "Weather_condition", "WeatherCondition"],
    "broad_phase_of_flight":  ["Broad_phase_of_flight", "Broad_Phase_of_Flight", "BroadPhaseofFlight"],
    "report_status":          ["Report_Status", "Report_status", "ReportStatus"],
    "publication_date":       ["Publication_Date", "Publication_date", "PublicationDate"],
}


def _classify_severity(injury_severity: Optional[str], total_fatal: Optional[int]) -> int:
    if injury_severity:
        for key, score in SEVERITY_MAP.items():
            if injury_severity.startswith(key):
                return score
    return 3 if (total_fatal and total_fatal > 0) else 0


def parse_xml_to_documents() -> list[dict]:
    print("\n[member1] === SECTION 2 — PARSE XML & BUILD DOCUMENTS ===")
    if not XML_PATH.exists():
        raise FileNotFoundError(f"{XML_PATH} missing — run acquire_ntsb_data() first")

    docs: list[dict] = []
    for _, evt in tqdm(etree.iterparse(str(XML_PATH), events=("end",), tag="Event"),
                       desc="parse xml"):
        doc = {f: _extract_field(evt, cands) for f, cands in FIELD_CANDIDATES.items()}
        doc["latitude"]               = _safe_float(doc.get("latitude"))
        doc["longitude"]              = _safe_float(doc.get("longitude"))
        doc["total_fatal_injuries"]   = _safe_int(doc.get("total_fatal_injuries"))
        doc["total_serious_injuries"] = _safe_int(doc.get("total_serious_injuries"))
        doc["total_minor_injuries"]   = _safe_int(doc.get("total_minor_injuries"))
        doc["total_uninjured"]        = _safe_int(doc.get("total_uninjured"))
        doc["number_of_engines"]      = _safe_int(doc.get("number_of_engines"))
        doc["event_date"]             = _safe_date(doc.get("event_date"))
        doc["publication_date"]       = _safe_date(doc.get("publication_date"))
        doc["severity_score"]         = _classify_severity(
            doc.get("injury_severity"), doc.get("total_fatal_injuries"))
        doc["is_imc"]          = (doc.get("weather_condition") or "").upper() == "IMC"
        doc["has_fatalities"]  = bool(doc.get("total_fatal_injuries") and
                                      doc["total_fatal_injuries"] > 0)
        docs.append(doc)
        evt.clear()

    print(f"[member1] Built {len(docs):,} documents")
    return docs


def _extract_field(event_el, names: list[str]) -> Optional[str]:
    for n in names:
        el = event_el.find(n)
        if el is not None and el.text not in (None, ""):
            return el.text
    return None


def insert_to_mongo(docs: list[dict]) -> int:
    print(f"\n[member1] Inserting {len(docs):,} docs into MongoDB…")
    db   = mongo_db()
    coll = db[MONGO_COLLECTION]
    coll.drop()
    BATCH = 1000
    for i in tqdm(range(0, len(docs), BATCH), desc="mongo insert"):
        coll.insert_many(docs[i: i + BATCH], ordered=False)
    for idx in ["event_date", "airport_code", "air_carrier",
                "broad_phase_of_flight", "weather_condition",
                "severity_score", "aircraft_category", "engine_type",
                "amateur_built", "state"]:
        coll.create_index(idx)
    count = coll.count_documents({})
    print(f"[member1] Inserted {count:,} documents")
    return count


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREPROCESSING & CLEANING
# ═════════════════════════════════════════════════════════════════════════════

PHASE_MAP = {
    "TAKEOFF": "Takeoff", "TAKE-OFF": "Takeoff", "INITIAL CLIMB": "Climb",
    "CLIMB": "Climb", "CRUISE": "Cruise", "DESCENT": "Descent",
    "APPROACH": "Approach", "GO-AROUND": "Approach",
    "MANEUVERING": "Cruise", "LANDING": "Landing", "TAXI": "Taxi",
    "STANDING": "Taxi", "PUSHBACK": "Taxi",
    "OTHER": "Unknown", "UNKNOWN": "Unknown",
}

CARRIER_VARIANT_MAP = {
    "UNITED AIR LINES":           "UNITED AIRLINES",
    "UNITED AIR LINES INC":       "UNITED AIRLINES",
    "AMERICAN AIRLINES INC":      "AMERICAN AIRLINES",
    "DELTA AIR LINES INC":        "DELTA AIR LINES",
    "SOUTHWEST AIRLINES CO":      "SOUTHWEST AIRLINES",
    "JETBLUE AIRWAYS CORPORATION": "JETBLUE AIRWAYS",
    "ALASKA AIRLINES INC":        "ALASKA AIRLINES",
    "FRONTIER AIRLINES INC":      "FRONTIER AIRLINES",
}

STATE_TO_REGION = {
    'AK': 'West', 'AL': 'South', 'AR': 'South', 'AZ': 'West', 'CA': 'West', 'CO': 'West', 'CT': 'Northeast',
    'DC': 'South', 'DE': 'South', 'FL': 'South', 'GA': 'South', 'HI': 'West', 'IA': 'Midwest', 'ID': 'West',
    'IL': 'Midwest', 'IN': 'Midwest', 'KS': 'Midwest', 'KY': 'South', 'LA': 'South', 'MA': 'Northeast',
    'MD': 'South', 'ME': 'Northeast', 'MI': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'MS': 'South',
    'MT': 'West', 'NC': 'South', 'ND': 'Midwest', 'NE': 'Midwest', 'NH': 'Northeast', 'NJ': 'Northeast',
    'NM': 'West', 'NV': 'West', 'NY': 'Northeast', 'OH': 'Midwest', 'OK': 'South', 'OR': 'West', 'PA': 'Northeast',
    'RI': 'Northeast', 'SC': 'South', 'SD': 'Midwest', 'TN': 'South', 'TX': 'South', 'UT': 'West', 'VA': 'South',
    'VT': 'Northeast', 'WA': 'West', 'WI': 'Midwest', 'WV': 'South', 'WY': 'West'
}

PURPOSE_MAP = {
    "PERSONAL": "Personal", "INSTRUCTIONAL": "Instructional", "BUSINESS": "Business",
    "POSITIONING": "Other", "AERIAL APPLICATION": "Other", "FLIGHT TEST": "Other",
    "EXTERNAL LOAD": "Other", "BANNER TOW": "Other", "SKYDIVING": "Other",
    "AERIAL ADVERTISING": "Other", "EXECUTIVE": "Business", "PUBLIC USE": "Other",
}


def _carrier_tier(name: Optional[str]) -> str:
    if not name:
        return "general_aviation"
    up = str(name).upper()
    if up in MAJOR_US_CARRIERS:
        return "major"
    if any(kw in up for kw in REGIONAL_KEYWORDS):
        return "regional"
    return "general_aviation"


def _normalise_carrier(name) -> Optional[str]:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    cleaned = str(name).strip().upper()
    return CARRIER_VARIANT_MAP.get(cleaned, cleaned)


def _simplify_phase(p: Optional[str]) -> str:
    if not p:
        return "Unknown"
    return PHASE_MAP.get(p.strip().upper(), "Unknown")


def _normalise_amateur(v) -> Optional[str]:
    """Normalise amateur_built to 'Yes' / 'No' / None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().upper()
    if s in ("YES", "Y", "1", "TRUE"):
        return "Yes"
    if s in ("NO", "N", "0", "FALSE"):
        return "No"
    return None
def _simplify_purpose(p: Optional[str]) -> str:
    if not p:
        return "Unknown"
    return PURPOSE_MAP.get(p.strip().upper(), "Other")

def load_and_clean() -> pd.DataFrame:
    print("\n[member1] === SECTION 3 — LOAD FROM MONGO & CLEAN ===")
    coll = mongo_db()[MONGO_COLLECTION]
    df   = pd.DataFrame(list(coll.find({}, {"_id": 0})))
    print(f"[member1] Raw shape: {df.shape}")

    df["event_date"]  = pd.to_datetime(df["event_date"],  errors="coerce")
    df["year"]        = df["event_date"].dt.year
    df["month"]       = df["event_date"].dt.month
    df["decade"]      = (df["year"] // 10 * 10).astype("Int64").astype(str) + "s"
    df.loc[df["year"].isna(), "decade"] = pd.NA

    df["air_carrier"]         = df["air_carrier"].apply(_normalise_carrier)
    df["latitude"]            = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"]           = pd.to_numeric(df["longitude"], errors="coerce")
    df["severity_score"]      = pd.to_numeric(df["severity_score"], errors="coerce").fillna(0).clip(upper=3).astype(int)
    df["broad_carrier_tier"]  = df["air_carrier"].apply(_carrier_tier)
    df["phase_simplified"]    = df["broad_phase_of_flight"].apply(_simplify_phase)
    df["purpose_simplified"]  = df["purpose_of_flight"].apply(_simplify_purpose)
    df["region"]              = df["state"].map(STATE_TO_REGION).fillna("Other")
    df["amateur_built_clean"] = df["amateur_built"].apply(_normalise_amateur)
    df["is_amateur_built"]    = df["amateur_built_clean"] == "Yes"

    # Normalise aircraft_category to a short readable label
    cat_map = {
        "AIR": "Airplane", "AIRPLANE": "Airplane",
        "HEL": "Helicopter", "HELICOPTER": "Helicopter",
        "GLI": "Glider", "GLIDER": "Glider",
        "BAL": "Balloon", "BALLOON": "Balloon",
        "WSC": "Weight-shift", "WEIGHT-SHIFT": "Weight-shift",
        "PPC": "Powered parachute", "POWERED PARACHUTE": "Powered parachute",
        "GYRO": "Gyroplane", "BLIMP": "Blimp/Airship",
    }
    df["category_clean"] = (
        df["aircraft_category"].fillna("Unknown").str.upper()
        .map(cat_map).fillna("Other")
    )

    # Normalise engine_type
    etype_map = {
        "RECI": "Reciprocating", "RECIPROCATING": "Reciprocating",
        "TURB": "Turboprop",     "TURBOPROP": "Turboprop",
        "TURB-FAN": "Turbofan",  "TURBOFAN": "Turbofan",
        "TURB-JET": "Turbojet",  "TURBOJET": "Turbojet",
        "TURB-SHAFT": "Turboshaft", "TURBOSHAFT": "Turboshaft",
        "ELEC": "Electric",      "ELECTRIC": "Electric",
        "NONE": "None/Glider",
    }
    df["engine_type_clean"] = (
        df["engine_type"].fillna("Unknown").str.upper()
        .map(etype_map).fillna("Other")
    )
    df["engine_bucket"] = df["engine_type_clean"]

    # Injuries total for survivability calc
    df["total_injuries"] = (
        df[["total_fatal_injuries", "total_serious_injuries", "total_minor_injuries"]]
        .apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    )
    df["total_aboard"] = df["total_injuries"] + df["total_uninjured"].fillna(0)

    # Survivability ratio = (survivors / total aboard), only where total > 0
    df["survivability"] = np.where(
        df["total_aboard"] > 0,
        df["total_uninjured"].fillna(0) / df["total_aboard"],
        np.nan,
    )

    print(f"[member1] Cleaned shape: {df.shape}")
    print("[member1] Null counts (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10).to_string())
    return df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ENHANCED VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════

def _save_fig(fig: go.Figure, n: int, title_slug: str = "") -> Path:
    slug = f"_{title_slug}" if title_slug else ""
    out  = DATA_DIR / f"member1_viz{n}{slug}.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"[member1] Saved → {out.name}")
    return out

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 1 — Severity trend by decade  (RQ-A)
# ─────────────────────────────────────────────────────────────────────────────
def viz1_severity_trend(df: pd.DataFrame) -> None:
    """
    Stacked bars show total incident volume per decade broken down by severity.
    A secondary-axis line overlays the fatal-rate percentage so the *relative*
    risk improvement is visible even as absolute counts change.
    
    Key insight: absolute counts fell dramatically from the 1980s peak, but the
    fatal-rate line reveals where proportional improvements were greatest.
    """
    print("\n[member1] VIZ 1 — Severity trend by decade")
    work = df.dropna(subset=["decade"]).copy()
    work["sev_label"] = work["severity_score"].map(
        {0: "None/Incident", 1: "Minor", 2: "Serious", 3: "Fatal"})

    counts = (work.groupby(["decade", "sev_label"])
              .size().reset_index(name="incidents"))
    fatal_rate = (
        work.assign(_f=work["has_fatalities"].astype(int))
        .groupby("decade")["_f"].mean().mul(100).reset_index(name="fatal_rate_pct")
    )
    decade_order = sorted(work["decade"].dropna().unique())

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    palette = {"None/Incident": "#22c55e", "Minor": "#fbbf24",
               "Serious": "#f97316", "Fatal": "#ef4444"}
    for label, color in palette.items():
        sub = counts[counts["sev_label"] == label]
        fig.add_bar(x=sub["decade"], y=sub["incidents"],
                    name=label, marker_color=color)

    fig.add_trace(
        go.Scatter(x=fatal_rate["decade"], y=fatal_rate["fatal_rate_pct"],
                   mode="lines+markers+text",
                   text=[f"{v:.1f}%" for v in fatal_rate["fatal_rate_pct"]],
                   textposition="top center",
                   name="Fatal rate %",
                   line=dict(color=COLOR_BLUE, width=3),
                   marker=dict(size=9)),
        secondary_y=True,
    )

    # Annotate sharpest improvement
    fr     = fatal_rate.set_index("decade")["fatal_rate_pct"].reindex(decade_order)
    diffs  = fr.diff().dropna()
    if not diffs.empty:
        best = diffs.idxmin()
        fig.add_annotation(
            x=best, y=fr.loc[best], xref="x", yref="y2",
            text=f"📉 Sharpest drop<br>in fatal rate: {best}",
            showarrow=True, arrowhead=2, ax=0, ay=-55,
            font=dict(color=COLOR_BLUE, size=11),
            bgcolor="white", bordercolor=COLOR_BLUE, borderwidth=1,
        )

    fig.update_layout(
        title=dict(text="<b>Aviation incident severity trend by decade (1982–2024)</b><br>"
                        "<sup>Stacked bars = incident count by severity | Line = % of incidents that were fatal</sup>",
                   font=dict(size=15)),
        barmode="stack", template=TEMPLATE,
        xaxis_title="Decade",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        height=520,
    )
    fig.update_yaxes(title_text="Incident count", secondary_y=False)
    fig.update_yaxes(title_text="Fatal rate (%)", secondary_y=True,
                     tickformat=".1f", range=[0, fatal_rate["fatal_rate_pct"].max() * 1.4])
    fig.show()
    _save_fig(fig, 1, "severity_by_decade")

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 3 — IMC vs VMC over time with regulatory milestones  (RQ-B)
# ─────────────────────────────────────────────────────────────────────────────
def viz3_imc_vs_vmc(df: pd.DataFrame) -> None:
    """
    IMC (instrument meteorological conditions) incidents are overlaid on VMC.
    The RATIO line (secondary axis) shows whether IMC is becoming a larger or
    smaller share of all incidents — the key policy question.
    Regulatory milestones are annotated to test whether rules caused step-changes.
    """
    print("\n[member1] VIZ 3 — IMC vs VMC over time")
    work = df.dropna(subset=["year"]).copy()
    work["weather_clean"] = work["weather_condition"].fillna("UNK").str.upper()
    work = work[work["weather_clean"].isin(["IMC", "VMC"])]

    yearly = (work.groupby(["year", "weather_clean"]).size()
              .unstack(fill_value=0).sort_index())
    for col in ["IMC", "VMC"]:
        if col not in yearly: yearly[col] = 0

    # Rolling 3-year average to smooth noise
    yearly["IMC_roll"] = yearly["IMC"].rolling(3, center=True).mean()
    yearly["VMC_roll"] = yearly["VMC"].rolling(3, center=True).mean()
    yearly["imc_ratio"] = (yearly["IMC"] / (yearly["IMC"] + yearly["VMC"]) * 100).round(2)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=yearly.index, y=yearly["VMC_roll"],
        name="VMC (3-yr avg)", mode="lines",
        line=dict(color=COLOR_BLUE, width=2),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.12)"),
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=yearly.index, y=yearly["IMC_roll"],
        name="IMC (3-yr avg)", mode="lines",
        line=dict(color=COLOR_RED, width=2),
        fill="tozeroy", fillcolor="rgba(220,38,38,0.15)"),
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=yearly.index, y=yearly["imc_ratio"],
        name="IMC share %", mode="lines+markers",
        line=dict(color=COLOR_PURPLE, width=2, dash="dot"),
        marker=dict(size=4)),
        secondary_y=True)

    milestones = [
        (2010, "Colgan Air reforms\n(Rulemaking 2010)"),
        (2013, "FAA Modernisation Act"),
        (2018, "ARC FITS review"),
    ]
    y_max = yearly[["IMC", "VMC"]].max().max()
    for yr, label in milestones:
        if yearly.index.min() <= yr <= yearly.index.max():
            fig.add_vline(x=yr, line_dash="dash", line_color="grey", line_width=1,
                          secondary_y=False)
            fig.add_annotation(
                x=yr, y=y_max * 0.9, xref="x", yref="y",
                text=label, showarrow=False,
                font=dict(color="grey", size=9), textangle=-90,
                xshift=8,
            )

    fig.update_layout(
        title=dict(text="<b>IMC vs VMC incidents over time</b><br>"
                        "<sup>Shaded area = 3-year rolling average | Dotted line = IMC share of all weather-recorded incidents</sup>",
                   font=dict(size=15)),
        template=TEMPLATE, height=500,
        xaxis_title="Year",
        legend=dict(orientation="h", y=-0.15, x=0.2),
    )
    fig.update_yaxes(title_text="Incident count (3-yr avg)", secondary_y=False)
    fig.update_yaxes(title_text="IMC share (%)", secondary_y=True,
                     tickformat=".0f", ticksuffix="%", range=[0, 50])
    fig.show()
    _save_fig(fig, 3, "imc_vmc_timeline")



# ─────────────────────────────────────────────────────────────────────────────
# VIZ 5 — Aircraft category risk profile bubble chart  (RQ-C)  ★ NEW
# ─────────────────────────────────────────────────────────────────────────────
def viz5_aircraft_category_risk(df: pd.DataFrame) -> None:
    """
    Bubble chart: each aircraft category plotted by
        X = incident rate (normalised — fraction of all incidents)
        Y = mean fatal injuries per incident
        Bubble size = total fatalities
        Colour = mean severity score

    Answers RQ-C: which aircraft categories carry the worst per-incident
    fatality burden even if they don't dominate total incident counts?

    Expected insight: Helicopters and weight-shift craft appear in the
    upper-right danger quadrant — high fatality rate per incident — while
    Airplanes dominate on raw volume but cluster lower on per-incident deaths.
    """
    print("\n[member1] VIZ 5 — Aircraft category risk profile")
    agg = (
        df.groupby("category_clean", dropna=False)
        .agg(
            incidents=("event_id", "count"),
            total_fatalities=("total_fatal_injuries", "sum"),
            mean_severity=("severity_score", "mean"),
            mean_fatal_injuries=("total_fatal_injuries", "mean"),
            fatal_incident_pct=("has_fatalities", "mean"),
        )
        .reset_index()
        .dropna(subset=["category_clean"])
    )
    # Filter to categories with enough data to be meaningful
    agg = agg[agg["incidents"] >= 50].copy()
    agg["incident_share_pct"] = agg["incidents"] / agg["incidents"].sum() * 100
    agg["total_fatalities"]   = agg["total_fatalities"].fillna(0)
    agg["fatal_incident_pct"] = (agg["fatal_incident_pct"] * 100).round(1)

    fig = px.scatter(
        agg,
        x="incident_share_pct",
        y="mean_fatal_injuries",
        size="total_fatalities",
        color="mean_severity",
        text="category_clean",
        color_continuous_scale="Reds",
        size_max=70,
        hover_data={
            "incidents": ":,",
            "total_fatalities": ":,",
            "fatal_incident_pct": ":.1f",
            "incident_share_pct": ":.1f",
            "mean_fatal_injuries": ":.2f",
        },
        labels={
            "incident_share_pct": "Share of all incidents (%)",
            "mean_fatal_injuries": "Mean fatal injuries per incident",
            "mean_severity": "Mean severity score",
            "total_fatalities": "Total fatalities",
        },
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(line=dict(color="white", width=1)),
    )

    # Add quadrant lines at medians
    x_med = agg["incident_share_pct"].median()
    y_med = agg["mean_fatal_injuries"].median()
    fig.add_hline(y=y_med, line_dash="dot", line_color="grey", line_width=1)
    fig.add_vline(x=x_med, line_dash="dot", line_color="grey", line_width=1)
    fig.add_annotation(x=agg["incident_share_pct"].max() * 0.95,
                       y=agg["mean_fatal_injuries"].max() * 0.95,
                       text="⚠️ High volume &<br>high lethality",
                       font=dict(color=COLOR_RED, size=10),
                       showarrow=False, bgcolor="rgba(255,255,255,0.8)")
    fig.add_annotation(x=x_med * 0.1,
                       y=y_med * 0.2,
                       text="✅ Low volume &<br>low lethality",
                       font=dict(color=COLOR_GREEN, size=10),
                       showarrow=False, bgcolor="rgba(255,255,255,0.8)")

    fig.update_layout(
        title=dict(
            text="<b>Aircraft category risk profile</b><br>"
                 "<sup>Bubble size = total fatalities | Colour = mean severity | "
                 "Dashed lines = medians</sup>",
            font=dict(size=15)),
        template=TEMPLATE, height=560,
    )
    fig.show()
    _save_fig(fig, 5, "aircraft_category_risk")


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 8 — Amateur-built vs certified: 40-year safety gap  (RQ-F)  ★ NEW
# ─────────────────────────────────────────────────────────────────────────────
def viz8_amateur_vs_certified(df: pd.DataFrame) -> None:
    """
    Dual-axis line chart tracking fatal-rate for amateur-built ('homebuilt')
    aircraft vs certified aircraft from 1982 to 2024.
    A third line shows the GAP (amateur fatal rate minus certified) to
    make any convergence explicit.

    Also includes a statistical significance note (Mann-Whitney U test)
    to answer RQ-F at a research level.

    Expected: fatal rate for amateur-built started 2–3× higher; EAA
    MOSAIC / LSA rules (post-2004) may have begun closing the gap.
    """
    print("\n[member1] VIZ 8 — Amateur-built vs certified safety gap")
    work = df.dropna(subset=["year", "amateur_built_clean"]).copy()
    work = work[work["amateur_built_clean"].isin(["Yes", "No"])]
    work = work[work["year"] >= 1982]

    yearly = (
        work.groupby(["year", "amateur_built_clean"])["has_fatalities"]
        .agg(fatal_rate="mean", count="count")
        .reset_index()
    )
    # Rolling 3-year average per group
    yearly = yearly.sort_values(["amateur_built_clean", "year"])
    yearly["fatal_rate_roll"] = (
        yearly.groupby("amateur_built_clean")["fatal_rate"]
        .transform(lambda s: s.rolling(3, min_periods=1, center=True).mean())
    )

    amateur   = yearly[yearly["amateur_built_clean"] == "Yes"].set_index("year")
    certified = yearly[yearly["amateur_built_clean"] == "No"].set_index("year")
    common    = amateur.index.intersection(certified.index)
    gap       = (amateur.loc[common, "fatal_rate_roll"] -
                 certified.loc[common, "fatal_rate_roll"]) * 100

    # Statistical test
    a_vals = work[work["amateur_built_clean"] == "Yes"]["has_fatalities"].astype(int)
    c_vals = work[work["amateur_built_clean"] == "No"]["has_fatalities"].astype(int)
    stat, pval = stats.mannwhitneyu(a_vals, c_vals, alternative="greater")
    sig_text = (f"Mann-Whitney U: p {'< 0.001' if pval < 0.001 else f'= {pval:.3f}'} "
                f"({'significant' if pval < 0.05 else 'not significant'})")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=amateur.index, y=amateur["fatal_rate_roll"] * 100,
        name="Amateur-built", mode="lines",
        line=dict(color=COLOR_RED, width=2.5),
        fill="tozeroy", fillcolor="rgba(220,38,38,0.08)"),
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=certified.index, y=certified["fatal_rate_roll"] * 100,
        name="Certified (factory-built)", mode="lines",
        line=dict(color=COLOR_BLUE, width=2.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"),
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=gap.index, y=gap.values,
        name="Safety gap (amateur − certified, pp)",
        mode="lines+markers", marker=dict(size=4),
        line=dict(color=COLOR_AMBER, width=2, dash="dot")),
        secondary_y=True)

    # EAA / LSA milestone
    fig.add_vline(x=2004, line_dash="dash", line_color=COLOR_GREEN, line_width=1.5)
    fig.add_annotation(x=2004, y=35, xref="x", yref="y",
                       text="LSA rule\n(2004)", showarrow=False,
                       font=dict(color=COLOR_GREEN, size=9), textangle=-90, xshift=8)

    fig.add_annotation(
        x=0.5, y=-0.17, xref="paper", yref="paper",
        text=sig_text, showarrow=False,
        font=dict(color="grey", size=10),
    )

    fig.update_layout(
        title=dict(
            text="<b>Amateur-built vs certified aircraft: 40-year fatal-rate gap (RQ-F)</b><br>"
                 "<sup>3-year rolling fatal rate % | Gap line (right axis) = amateur minus certified percentage points</sup>",
            font=dict(size=15)),
        template=TEMPLATE, height=520,
        xaxis_title="Year",
        legend=dict(orientation="h", y=-0.12, x=0.15),
    )
    fig.update_yaxes(title_text="Fatal rate (%, 3-yr rolling avg)", secondary_y=False,
                     tickformat=".0f", ticksuffix="%")
    fig.update_yaxes(title_text="Safety gap (percentage points)", secondary_y=True,
                     tickformat=".1f")
    fig.show()
    _save_fig(fig, 8, "amateur_vs_certified")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — STORE CLEANED DATA TO POSTGRESQL
# ═════════════════════════════════════════════════════════════════════════════

def write_to_postgres(df: pd.DataFrame) -> None:
    print("\n[member1] === SECTION 5 — WRITE TO POSTGRES ===")
    eng = neon_engine()
    out = df.copy()

    # Force numeric columns BEFORE converting objects to strings
    numeric_cols = [
        "total_fatal_injuries", "total_serious_injuries",
        "total_minor_injuries", "total_uninjured", "severity_score",
        "number_of_engines", "latitude", "longitude",
        "year", "month", "total_injuries", "total_aboard",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype("string")

    # Drop table with CASCADE to handle dependent views (e.g., incident_enriched)
    with eng.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {PG_TABLE} CASCADE"))

    # Use chunksize without method='multi' to avoid the 65k parameter limit
    out.to_sql(PG_TABLE, eng, if_exists="replace", index=False, chunksize=500)
    print(f"[member1] Wrote {len(out):,} rows → PostgreSQL table '{PG_TABLE}'")


# ═════════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════════

def run() -> None:
    t0 = time.time()

    print("=" * 60)
    print("  Member 1 — NTSB aviation incident pipeline (enhanced)")
    print("=" * 60)

    acquire_ntsb_data()
    docs = parse_xml_to_documents()
    insert_to_mongo(docs)
    df = load_and_clean()

    print(f"\n[member1] Dataset summary:")
    print(f"  Total incidents : {len(df):,}")
    print(f"  Year range      : {df['year'].min():.0f} – {df['year'].max():.0f}")
    print(f"  Fatal incidents : {df['has_fatalities'].sum():,} "
          f"({df['has_fatalities'].mean()*100:.1f}%)")
    print(f"  IMC incidents   : {df['is_imc'].sum():,} "
          f"({df['is_imc'].mean()*100:.1f}%)")
    print(f"  With lat/lon    : {df.dropna(subset=['latitude','longitude']).shape[0]:,}")
    print(f"  Aircraft cats   : {df['category_clean'].value_counts().to_dict()}")

    # All 8 visualisations
    viz1_severity_trend(df)
    viz3_imc_vs_vmc(df)
    viz5_aircraft_category_risk(df)
    viz8_amateur_vs_certified(df)

    write_to_postgres(df)

    elapsed = time.time() - t0
    print(f"\n[member1] ✅ ALL DONE in {elapsed:,.1f}s")
    print(f"[member1] Visualisations saved to {DATA_DIR}/member1_viz*.html")


if __name__ == "__main__":
    run()