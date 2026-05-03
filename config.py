"""
config.py
=========

Central configuration / connection helpers for the aviation safety
analytics project. Every other module should import from here so we
have ONE place that knows about credentials, database URIs, and
external endpoints.

Usage:
    from config import mongo_client, neon_engine, OPENMETEO_BASE, AIRPORTS_CSV

Why this exists:
- Keeps credentials out of business logic.
- Makes it trivial to swap connection strategies (e.g. add pooling).
- Standardises error messaging when DBs are unreachable.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------
# Environment variables — read once at import time.
# -----------------------------------------------------------------
MONGO_URI: Optional[str] = os.getenv("MONGO_URI")
MONGO_DB: str = os.getenv("MONGO_DB", "aviation")
NEON_URI: Optional[str] = os.getenv("NEON_URI")

OPENMETEO_BASE: str = os.getenv(
    "OPENMETEO_BASE", "https://archive-api.open-meteo.com/v1/archive"
)
AIRPORTS_CSV: str = os.getenv(
    "AIRPORTS_CSV", "https://ourairports.com/data/airports.csv"
)

# -----------------------------------------------------------------
# Hard-coded constants shared across modules
# -----------------------------------------------------------------
# Top US carriers by passenger volume (used to tag carrier_tier).
MAJOR_US_CARRIERS = {
    "AMERICAN AIRLINES",
    "DELTA AIR LINES",
    "UNITED AIRLINES",
    "SOUTHWEST AIRLINES",
    "ALASKA AIRLINES",
    "JETBLUE AIRWAYS",
    "SPIRIT AIRLINES",
    "FRONTIER AIRLINES",
    "HAWAIIAN AIRLINES",
    "ALLEGIANT AIR",
}

REGIONAL_KEYWORDS = ("REGIONAL", "EXPRESS", "COMMUTER", "AIR TAXI")


# -----------------------------------------------------------------
# Connection helpers
# -----------------------------------------------------------------
def mongo_client():
    """
    Return a connected pymongo.MongoClient.
    Raises a friendly RuntimeError if MONGO_URI is missing or unreachable.
    """
    if not MONGO_URI:
        raise RuntimeError(
            "MONGO_URI not set. Copy .env.example to .env and fill in your "
            "MongoDB Atlas connection string."
        )
    try:
        import certifi
        from pymongo import MongoClient
        # tlsCAFile=certifi.where() avoids `[SSL: TLSV1_ALERT_INTERNAL_ERROR]`
        # on Python 3.12+ on macOS where the system trust store is not
        # picked up by OpenSSL automatically.
        # serverSelectionTimeoutMS keeps the failure fast and obvious instead
        # of hanging on a broken cluster URL.
        client = MongoClient(
            MONGO_URI,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=8000,
        )
        client.admin.command("ping")
        return client
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to MongoDB — check MONGO_URI in .env. "
            f"Underlying error: {exc}"
        ) from exc


def mongo_db():
    """Convenience wrapper that returns the configured database handle."""
    return mongo_client()[MONGO_DB]


def neon_engine():
    """
    Return a SQLAlchemy engine pointing at the Neon Postgres instance.
    Validates the connection eagerly so failures surface here, not in
    business code.
    """
    if not NEON_URI:
        raise RuntimeError(
            "NEON_URI not set. Copy .env.example to .env and fill in your "
            "Neon Postgres connection string (must include ?sslmode=require)."
        )
    if "sslmode=require" not in NEON_URI:
        # psycopg2 on some platforms silently fails the TLS upgrade;
        # warn loudly rather than mysteriously dying mid-pipeline.
        print(
            "[config] WARNING: NEON_URI is missing ?sslmode=require — "
            "psycopg2 may silently fail. Append it to your connection string.",
            file=sys.stderr,
        )
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(NEON_URI, pool_pre_ping=True, future=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to PostgreSQL/Neon — check NEON_URI in .env. "
            f"Underlying error: {exc}"
        ) from exc


if __name__ == "__main__":
    # A quick smoke test — `python config.py` to verify env wiring.
    print("[config] Verifying environment configuration...")
    print(f"  MONGO_DB        = {MONGO_DB}")
    print(f"  OPENMETEO_BASE  = {OPENMETEO_BASE}")
    print(f"  AIRPORTS_CSV    = {AIRPORTS_CSV}")
    print(f"  MONGO_URI set?  = {bool(MONGO_URI)}")
    print(f"  NEON_URI set?   = {bool(NEON_URI)}")
    try:
        mongo_client()
        print("  [OK] MongoDB reachable")
    except RuntimeError as exc:
        print(f"  [FAIL] {exc}")
    try:
        neon_engine()
        print("  [OK] PostgreSQL reachable")
    except RuntimeError as exc:
        print(f"  [FAIL] {exc}")
