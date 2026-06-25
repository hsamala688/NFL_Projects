# NFL Analytics Pipeline

A production-style NFL analytics platform spanning three disciplines: a data-engineering pipeline, a machine-learning layer, and an AI-engineering layer. Raw play-by-play data flows from ingestion through DuckDB and dbt, an XGBoost completion-probability model produces situation-adjusted CPOE rankings, and a natural-language analyst agent makes the whole thing queryable in plain English through a web app.

```
nflreadpy -> Parquet -> DuckDB -> dbt -> Prefect -> XGBoost -> ADK agent -> FastAPI + React
                                          |
                                          +-> Streamlit dashboard
```

The project carries a thesis: raw completion percentage is misleading because quarterbacks face different throw difficulty, and CPOE (completion percentage over expected) is the honest metric. Every layer exists to make that thesis usable, ending with an agent that recognizes when a question is secretly about expectation-adjusted performance and reaches for the model instead of the raw stat.

---

## The Three Layers

**Data engineering.** Ingestion, columnar storage, a query engine, a tested transformation project, and scheduled orchestration. This is the foundation everything else queries.

**Machine learning.** A gradient-boosted classifier that predicts pass completion from pre-snap situation, trained era-neutrally across 20 seasons, producing model-adjusted CPOE rankings that differ meaningfully from raw completion leaderboards.

**AI engineering.** A text-to-SQL analyst agent over the DuckDB marts, with the saved model exposed as a callable tool, served through a FastAPI backend and a React frontend.

---

## Tech Stack

| Layer | Tool | Reason |
|---|---|---|
| Languages | Python 3.12, SQL, TypeScript | Pipeline and ML in Python, transforms in SQL, frontend in TypeScript |
| Data source | `nflreadpy` | Official successor to `nfl_data_py`; Python 3.12 compatible; Polars-native |
| Raw storage | Parquet (snappy) | Columnar, fast reads, regenerable |
| Query engine | DuckDB | OLAP, reads Parquet natively, no server required |
| Transformation | dbt | Three-layer model structure, testing, auto-docs |
| Orchestration | Prefect | Scheduled weekly pipeline runs |
| ML | XGBoost, scikit-learn | Gradient-boosted completion model with a logistic baseline |
| Agent | Google ADK | Text-to-SQL analyst agent with the model exposed as a tool |
| Backend | FastAPI | Serves the agent behind a demo-mode and live-agent split |
| Frontend | React + TypeScript | Web interface for the agent |
| Dashboard | Streamlit, Plotly | Interactive QB analytics dashboard |

---

## Project Structure

```
NFL-Projects/
├── ingestion/
│   ├── pull_pbp.py          # Pull PBP data from nflverse, write Parquet
│   └── load_duckdb.py       # Register Parquet files as DuckDB views
├── nfl_analytics/           # dbt project
│   └── models/
│       ├── staging/         # stg_pbp.sql, clean & rename raw columns
│       ├── intermediate/    # int_qb_plays.sql, per-play passing metrics
│       └── marts/           # mart_qb_season.sql, QB season aggregates
├── flows/
│   └── nfl_pipeline.py      # Prefect orchestration flow
├── ml/
│   ├── train.py             # Feature engineering, model training, evaluation
│   ├── predict.py           # Per-play predictions, QB CPOE comparison table
│   └── calibration_plot.png # Calibration curve output from train.py
├── agent/
│   ├── semantic_layer.py    # Data dictionary the agent reads before querying
│   └── tools.py             # query_marts, get_definition, run_cpoe
├── backend/                 # FastAPI app wrapping the agent
├── frontend/                # React + TypeScript web app
├── dashboard/
│   └── app.py               # Streamlit dashboard (Season Stats + ML CPOE tabs)
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
# Pull raw data (2006 is the model floor, where charted features like air_yards exist)
python ingestion/pull_pbp.py --seasons 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025

# Register DuckDB views
python ingestion/load_duckdb.py

# Run dbt models and tests
cd nfl_analytics && dbt run && dbt test
```

### 4. Train the ML model

```bash
# Trains on 2006-2024, holds out 2025, saves xgb_model.json
python ml/train.py
```

### 5. Launch the dashboard

```bash
# From repo root, loads the saved model and runs live predictions
streamlit run dashboard/app.py
```

---

## Layer 1: Data Engineering

### Ingestion and storage
`ingestion/pull_pbp.py` pulls play-by-play data via `nflreadpy` and writes one snappy-compressed Parquet file per season. `ingestion/load_duckdb.py` registers each season as a named DuckDB view plus a unified `pbp` view that globs all seasons. DuckDB reads the Parquet files directly, so `db/nfl.duckdb` stores only metadata and the underlying data stays in regenerable Parquet.

The data spans 20 seasons (2006 to 2025), 469,122 pass attempts. The 2006 floor is deliberate: charted features like `air_yards` and `pass_location` become available that year, which is what the model needs. Raw stats like EPA reach further back, but 2006 is the model floor. `nflreadpy` normalizes all team codes to current franchise identifiers, so historical codes (SD, STL, OAK) never appear.

### dbt models

**Staging (`stg_pbp`).** Selects and renames relevant columns from the raw `pbp` DuckDB view. No business logic, just cleaning.

**Intermediate (`int_qb_plays`).** Filters to regular-season passing plays, computes per-play metrics (EPA, air yards, CPOE, WPA), classifies plays into distance buckets (short, medium, long), and exposes all ML feature columns. One row per pass attempt.

**Marts (`mart_qb_season`).** Aggregates to a QB season summary with a minimum 100-attempt threshold baked in.

| Column | Description |
|---|---|
| `passer_player_name` | QB name |
| `season` | Season year |
| `completions` | Total completions |
| `attempts` | Total attempts |
| `completion_pct` | Completion percentage |
| `avg_epa` | Average EPA per play |
| `avg_air_yards` | Average air yards per attempt |
| `avg_cpoe` | Average Completion Percentage Over Expected (nflfastR built-in) |
| `interceptions` | Total interceptions |
| `total_wpa` | Total Win Probability Added |

### Orchestration
The Prefect flow at `flows/nfl_pipeline.py` wraps the full pipeline into four tasks:

```
pull_pbp_task -> load_duckdb_task -> dbt_run_task -> dbt_test_task
```

```bash
# Manual run
python flows/nfl_pipeline.py --run-now

# Start scheduler (Tuesdays at 6am)
python flows/nfl_pipeline.py
```

---

## Layer 2: Machine Learning

An XGBoost binary classifier predicts whether a given pass attempt will be completed, using only pre-snap situational features. A logistic regression model serves as the baseline to quantify what the gradient-boosted model adds.

**Features:** down, yards to go, field position, air yards, pass location (left/middle/right), score differential, time remaining, shotgun, no-huddle, and season.

**Target:** `is_complete` (0/1)

**Training data:** 2006-2024 regular-season passing plays
**Holdout:** 2025 regular season

| Model | AUC |
|---|---|
| Logistic Regression (baseline) | 0.670 |
| XGBoost | 0.699 |

### Era-neutrality
Training across 20 seasons introduces a confound: passing has gotten easier over time, so a model blind to era systematically biases CPOE against older quarterbacks. The fix is adding `season` as a feature, then running an era-neutrality check that confirms the model's CPOE has no residual drift across seasons. Air yards remains the dominant predictor, confirming that depth of target is the strongest situational driver of completion.

The roughly 0.70 AUC ceiling is consistent with published CPOE models on the same data. The remaining unpredictability reflects genuine uncertainty (coverage, separation, execution) not visible in pre-snap features.

### CPOE rankings
`predict.py` applies the saved model per-play and aggregates by QB to compute model CPOE (actual completion percentage minus model-expected), then compares the model ranking against the raw completion ranking. The gap between the two rankings is the whole point: quarterbacks on easy throw schedules rank far lower once difficulty is accounted for.

Two distinct CPOE numbers live in this project, and they are not the same thing:
- `avg_cpoe` (mart column) is nflfastR's built-in CPOE from nflverse. It is a column you query.
- `model_cpoe` is this project's metric, produced only by applying the saved XGBoost model. It is never a column.

---

## Layer 3: AI Engineering

### Analyst agent
A natural-language analyst agent (Google ADK) sits over the DuckDB marts and translates football questions into validated SQL. It reads a semantic layer (`agent/semantic_layer.py`), a data dictionary covering column meanings, the dbt renames, the baked-in filters, and the abstention boundary, before writing any query.

Three tools:
- `query_marts(sql)` executes a read-only SELECT against the whitelisted views (`mart_qb_season`, `int_qb_plays`). SELECT only; anything that is not a read is parsed out and rejected. All connections are `read_only=True`, so the agent never holds a write lock.
- `get_definition(term)` reads the semantic layer.
- `run_cpoe(filters)` is the differentiator. It applies the saved XGBoost model to a filtered slice of plays and returns the CPOE aggregation, so the agent can answer expectation-adjusted questions rather than only raw aggregates.

The product judgment lives in tool selection. A question like "who is the most accurate QB" is really about expectation-adjusted performance. A naive bot answers with the raw completion leaderboard and is confidently wrong. The agent recognizes the question is about CPOE, calls `run_cpoe`, and explains why the raw ranking would mislead. The two-CPOE distinction above is the conflation most likely to break that selection, so it is stated explicitly in both the semantic layer and the agent instruction.

Every answer returns a structured object, not loose prose: the plain-English answer, the evidence rows behind it, the tool calls that ran, a confidence, and the caveats for what the data cannot capture. Every stated number traces to a row in the evidence, which is how hallucinated stats are kept out.

### Web app
A FastAPI backend wraps the agent and a React + TypeScript frontend consumes it. The public path serves cached demo responses with no live LLM calls, while the live agent is gated behind an API key. The agent queries a sandboxed DuckDB serving database that materializes the mart views into a self-contained file with external file and network access disabled, kept as a deliberate hardening showcase.

---

## Dashboard

The Streamlit dashboard reads directly from DuckDB and renders two tabs:

**Season Stats**
- Season selector to switch between available seasons
- QB Leaderboard, a sortable table of all QBs with all metrics
- Scatter plot of completion % vs avg EPA to identify efficient QBs
- Bar chart of the top 10 QBs by avg EPA
- QB detail view to select any QB and see their full stat line, including model CPOE

**ML Model CPOE**
- CPOE leaderboard of QB rankings by model-adjusted completion % over expected
- Diverging bar chart of the top and bottom 10 QBs by model CPOE
- Scatter plot of raw completion % vs model CPOE to surface which QBs are padding stats with easy throws vs genuinely outperforming their situations

The dashboard loads the saved model at startup and runs live per-play predictions to generate each QB's model CPOE, with no stale CSV required.

---

## Notes

- `db/nfl.duckdb`, `data/raw/`, and `ml/xgb_model.json` are gitignored, and all are fully regenerable by running the bootstrap commands above
- DuckDB uses file-level locking: only one write connection can be open at a time. Close any open connections before running `dbt run`
- `mart_qb_season` is a dbt view, not a materialized table, so it recomputes on every query, which is fine at this data scale
- The raw interception column is `interception` (singular); the plural exists only as the aggregated mart column
