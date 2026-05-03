# Aviation Safety Intelligence
 
End-to-end analytics pipeline that ingests three semi-/un-structured aviation
datasets (NTSB accident records, BTS on-time performance, and Open-Meteo
historical weather) into MongoDB, cleans and joins them in PostgreSQL (Neon),
and serves the results through an interactive Plotly Dash dashboard.
 
---
 
## 1. Setup
 
```bash
# 1. Install dependencies
pip install -r requirements.txt
 
# 2. Copy the env template and fill in real credentials
cp .env.example .env
$EDITOR .env
```
 
### MongoDB Atlas
 
1. Create a free M0 cluster at [cloud.mongodb.com](https://cloud.mongodb.com).
2. Add your IP to the access list, create a DB user.
3. Copy the SRV connection string into `MONGO_URI`.
 
### Neon PostgreSQL
 
1. Create a project at [neon.tech](https://neon.tech) (free tier is fine).
2. Copy the pooled connection string into `NEON_URI`.
3. **Make sure the URI ends with `?sslmode=require`** — without it
   `psycopg2` will silently fail to TLS-upgrade on some machines.
 
---
 
## 2. Run order
 
```bash
# Neha — NTSB incidents (XML semi-structured)
python neha_ntsb_incidents.py
 
# Shreya — BTS on-time performance (large CSV/zip)
python shreya_bts_flights.py
 
# Varshini — Open-Meteo weather enrichment (API)
# Tip: run in a separate terminal, in parallel with member 2.
# It hits the API ~3,000 times and takes 5-10 minutes.
python varshini_weather.py
 
# ETL — build joined analytical tables in Postgres
python etl_pipeline.py
 
# Dashboard — open http://localhost:8050
python dashboard.py
```
 
---
 
## 3. Data sources
 
| Source | URL | Format |
|---|---|---|
| NTSB Aviation Accident Database | <https://data.ntsb.gov/avdata/FileDirectory/DownloadFile?fileID=C%3A%5CUsers%5CPublic%5CDocuments%5CAVALL_CSV.zip> | CSV → converted to XML |
| NTSB CAROL fallback (if main URL 404s) | <https://data.ntsb.gov/carol-main-public/landing-page> | n/a |
| BTS Reporting Carrier On-Time Performance | <https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YEAR}_…> | CSV in zip |
| Open-Meteo Historical Archive API | <https://archive-api.open-meteo.com/v1/archive> | JSON |
| OurAirports reference data | <https://ourairports.com/data/airports.csv> | CSV |
 
---
 
## 4. Research questions
 
| # | Question | Where it is answered |
|---|---|---|
| RQ1 | Do regional carriers experience disproportionately worse outcomes in IMC weather compared to major airlines? | `varshini_weather.py` (Viz 1 & 2) and Dashboard **Tab 1** |
| RQ2 | Have specific phases of flight become safer over time, and where has the biggest improvement happened? | `neha_ntsb_incidents.py` (Viz 1 & 3) and Dashboard **Tab 2** |
| RQ3 | Which US states are geographic hotspots for aviation incidents and is terrain (latitude / mountains) a contributing factor? | `neha_ntsb_incidents.py` (Viz 4), `varshini_weather.py` (Viz 4), Dashboard **Tab 3** |
 
---
 
## 5. Architecture
 
```
                   ┌────────────────────┐
   NTSB CSV ───►   │ member1 ► Mongo    │ ──► incidents
                   ├────────────────────┤
   BTS CSV  ───►   │ member2 ► Mongo    │ ──► flights
                   ├────────────────────┤
Open-Meteo ───►   │ member3 ► Mongo    │ ──► weather_raw
                   └─────────┬──────────┘
                             │ (clean + transform in pandas)
                             ▼
                   ┌────────────────────┐
                   │  Neon PostgreSQL   │
                   │  incidents_clean   │
                   │  carrier_stats     │
                   │  flights_sample    │
                   │  incidents_weather │
                   └─────────┬──────────┘
                             │ etl_pipeline.py
                             ▼
                   ┌────────────────────┐
                   │  incident_enriched │  (view)
                   │  rq1_*  rq2_*  rq3_* (materialised)
                   └─────────┬──────────┘
                             │
                             ▼
                       Plotly Dash
```
 
---
 
## 6. Notes / gotchas
 
- The NTSB direct download URL changes from time to time; both
  `neha_ntsb_incidents.py` and `etl_pipeline.py` will fall back to
  the CAROL landing page and print which endpoint succeeded.
- Open-Meteo is rate-limited; the script sleeps `0.1s` between calls
  and uses `tqdm` so you can see progress.
- All Postgres writes use `if_exists='replace'` so the pipeline is
  idempotent — re-running will not duplicate rows.
- **NTSB XML Parsing**: The M1 pipeline (`neha_ntsb_incidents.py`) uses case-insensitive fallback mappings to correctly extract fields like `air_carrier` and `injury_severity` regardless of XML tag casing variations.
- **PostgreSQL Limits**: The pipeline uses `chunksize=500` and avoids `method="multi"` when writing large tables (like `incidents_clean`) to bypass the psycopg2 65,535 parameter limit.
- **Carrier Name Joins**: M2 automatically maps BTS 2-letter carrier codes to full names (e.g., `DL` to `DELTA AIR LINES`) so they join correctly with the NTSB records in the analytical layer.
- **Schema Management**: The ETL pipeline uses `DROP TABLE ... CASCADE` to cleanly drop base tables and any dependent analytical views (`incident_enriched`) before rebuilding them.
- **Dashboard Fallbacks**: The dashboard and analytical views (`rq1`, `rq2`, `rq3`) are designed with robust fallbacks. If weather or carrier data is sparse, the visualizations will gracefully fall back to base tables (`incidents_clean` and `incidents_weather`) instead of failing.
 
Cloud: MongoDB Cloud
 # TeamL-Project
