# NFL Analytics Pipeline
 
A production-style NFL data pipeline built as a portfolio project. Raw play-by-play data flows through ingestion, DuckDB, dbt transformations, Prefect orchestration, and a Streamlit dashboard.
 
```
nflreadpy → Parquet → DuckDB → dbt → Prefect → Streamlit
```
 
---
 
## Tech Stack
 
| Layer | Tool | Reason |
|---|---|---|
| Data source | `nflreadpy` | Official successor to `nfl_data_py`; Python 3.12 compatible; Polars-native |
| Raw storage | Parquet (snappy) | Columnar, fast reads, regenerable |
| Query engine | DuckDB | OLAP, reads Parquet natively, no server required |
| Transformation | dbt | Three-layer model structure, testing, auto-docs |
| Orchestration | Prefect | Scheduled weekly pipeline runs |
| Dashboard | Streamlit | Interactive QB analytics dashboard |
 
---
 
## Project Structure
 
```
NFL-Projects/
├── ingestion/
│   ├── pull_pbp.py          # Pull PBP data from nflverse, write Parquet
│   └── load_duckdb.py       # Register Parquet files as DuckDB views
├── nfl_analytics/           # dbt project
│   └── models/
│       ├── staging/         # stg_pbp.sql — clean & rename raw columns
│       ├── intermediate/    # int_qb_plays.sql — per-play passing metrics
│       └── marts/           # mart_qb_season.sql — QB season aggregates
├── flows/
│   └── nfl_pipeline.py      # Prefect orchestration flow
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── db/
│   └── nfl.duckdb           # DuckDB file (gitignored, regenerable)
├── data/
│   └── raw/                 # Parquet files (gitignored, regenerable)
├── queries/
│   └── qb_completion.sql    # Validated Phase 2 reference query
├── profiles.yml.example     # dbt profile template
└── requirements.txt
```
 
---
 
## Setup
 
### 1. Clone and create virtual environment
 
```bash
git clone https://github.com/hsamala688/NFL_Projects.git
cd NFL-Projects
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
 
### 2. Configure dbt profile
 
Copy the template and edit the path to match your local setup:
 
```bash
cp profiles.yml.example ~/.dbt/profiles.yml
```
 
The profile should point to `db/nfl.duckdb` using an absolute path:
 
```yaml
nfl_analytics:
  outputs:
    dev:
      type: duckdb
      path: /your/path/to/NFL-Projects/db/nfl.duckdb
      threads: 4
  target: dev
```
 
### 3. Run the full pipeline
 
```bash
# Pull raw data
python ingestion/pull_pbp.py --seasons 2025
 
# Register DuckDB views
python ingestion/load_duckdb.py
 
# Run dbt models and tests
cd nfl_analytics && dbt run && dbt test
```
 
### 4. Launch the dashboard
 
```bash
# From repo root
streamlit run dashboard/app.py
```
 
---
 
## dbt Models
 
### Staging — `stg_pbp`
Selects and renames relevant columns from the raw `pbp` DuckDB view. No business logic, just cleaning.
 
### Intermediate — `int_qb_plays`
Filters to regular season passing plays (`complete_pass`, `incomplete_pass`, or `interception`), computes per-play metrics (EPA, air yards, CPOE, WPA), and classifies plays into distance buckets (short / medium / long).
 
### Marts — `mart_qb_season`
Aggregates to QB season summary. Minimum 100 attempts threshold. Columns:
 
| Column | Description |
|---|---|
| `passer_player_name` | QB name |
| `season` | Season year |
| `completions` | Total completions |
| `attempts` | Total attempts |
| `completion_pct` | Completion percentage |
| `avg_epa` | Average EPA per play |
| `avg_air_yards` | Average air yards per attempt |
| `avg_cpoe` | Average Completion Percentage Over Expected |
| `interceptions` | Total interceptions |
| `total_wpa` | Total Win Probability Added |
 
---
 
## Orchestration
 
The Prefect flow at `flows/nfl_pipeline.py` wraps the full pipeline into four tasks:
 
```
pull_pbp_task → load_duckdb_task → dbt_run_task → dbt_test_task
```
 
```bash
# Manual run
python flows/nfl_pipeline.py --run-now
 
# Start scheduler (Tuesdays at 6am)
python flows/nfl_pipeline.py
```
 
---
 
## Dashboard
 
The Streamlit dashboard reads directly from `mart_qb_season` in DuckDB and renders:
 
- **Season selector** — switch between available seasons
- **QB Leaderboard** — sortable table of all QBs with all metrics
- **Scatter plot** — completion % vs avg EPA to identify efficient QBs
- **Bar chart** — top 10 QBs by avg EPA
- **QB detail view** — select any QB and see their full stat line
---
 
## Notes
 
- `db/nfl.duckdb` and `data/raw/` are gitignored — both are fully regenerable by running the bootstrap commands above
- DuckDB uses file-level locking: only one write connection can be open at a time. Close any open connections (e.g. in a Python console) before running `dbt run`
- `mart_qb_season` is a dbt view, not a materialized table — it recomputes on every query, which is fine at this data scale