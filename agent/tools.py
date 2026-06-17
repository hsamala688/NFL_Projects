from pathlib import Path
import duckdb
import sqlglot
from sqlglot import exp
import xgboost as xgb
from ml.predict import prepare_features, compute_qb_cpoe, MODEL_PATH, DB_PATH

# Single source of truth. Do not redefine the whitelist here.
from agent.semantic_layer import COLUMNS, METRICS, QUERYABLE_VIEWS

_VIEWS = set(QUERYABLE_VIEWS)
_DB_PATH = Path(__file__).resolve().parent.parent / "db" / "nfl.duckdb"

def _validate(sql: str) -> None:
    """Reject anything that is not a single SELECT over the whitelisted views."""
    try:
        statements = sqlglot.parse(sql, dialect="duckdb")
    except sqlglot.errors.ParseError as e:
        raise ValueError(f"Could not parse SQL: {e}")

    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Exactly one statement is allowed.")

    stmt = statements[0]
    if not isinstance(stmt, exp.Select):
        raise ValueError("Only SELECT statements are allowed.")

    # CTE aliases are not real tables, so remove them before the whitelist check.
    cte_names = {cte.alias_or_name for cte in stmt.find_all(exp.CTE)}
    referenced = {t.name for t in stmt.find_all(exp.Table)} - cte_names
    illegal = referenced - _VIEWS
    if illegal:
        raise ValueError(f"Query references tables outside the whitelist: {sorted(illegal)}")


def query_marts(sql: str) -> dict:
    """
    Run a read-only SELECT against the NFL marts and return the rows.
    Use for raw stats and aggregates. sql must be a single SELECT against
    mart_qb_season or int_qb_plays. Returns status plus rows, or an error message.
    """

    try: 
        _validate(sql)
        with duckdb.connect(str(_DB_PATH), read_only=True) as con:
            cur = con.execute(sql)
            columns = [c[0] for c in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]
            return {"status": "success", "rows": rows}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

# completion_pct_over_expected was renamed to avg_cpoe in the semantic layer, so no concern that it doesn't show up in listing
def get_definition(term: str) -> str:
    """
    Look up the meaning of a metric or column from the semantic layer.

    term: a metric name (e.g. 'CPOE', 'EPA') or a column name (e.g. 'air_yards').
    Returns the definition text, or a message listing known terms if not found.
    """

    q = term.strip().lower()
    hits = []

    # metric keys can be slash-separated aliases, e.g. "EPA / avg_epa"
    for key, desc in METRICS.items():
        aliases = [a.strip().lower() for a in key.split("/")]
        if any(q in alias or alias in q for alias in aliases):
            hits.append(f"{key}: {desc}")

    # column names across both queryable views
    for view, cols in COLUMNS.items():
        for name, meta in cols.items():
            if q in name.lower():
                was = f" (was {meta['renamed_from']})" if "renamed_from" in meta else ""
                hits.append(f"{name} in {view}{was}: {meta['meaning']}")

    if hits:
        return "\n\n".join(hits)

    known_metrics = ", ".join(METRICS.keys())
    known_cols = ", ".join(sorted({c for cols in COLUMNS.values() for c in cols}))
    return (f"No definition found for '{term}'. "f"Known metric terms: {known_metrics}. Known columns: {known_cols}.")

_model = None


def _load_model():
    """Load the saved XGBoost model once and reuse it across calls."""
    global _model
    if _model is None:
        m = xgb.XGBClassifier()
        m.load_model(str(MODEL_PATH))
        _model = m
    return _model


def run_cpoe(season: int = 2025) -> dict:
    """
    Rank QBs by model-based CPOE for a season, accounting for throw difficulty.

    Use this for accuracy or expectation-adjusted questions: who is actually good,
    who outperforms or underperforms expectation. Applies the project's XGBoost
    model to that season's pass attempts and returns, per QB with at least 100
    attempts: actual_pct, expected_pct, model_cpoe, raw_rank, model_rank, rank_diff.
    Returns status plus rows, or an error message.
    """
    try:
        conn = duckdb.connect(str(DB_PATH), read_only=True)
        df = conn.execute(
            """
            SELECT passer_player_name, season, week, down, yards_to_go,
                   yards_from_endzone, air_yards, pass_location, score_differential,
                   game_seconds_remaining, shotgun, no_huddle, is_complete
            FROM int_qb_plays
            WHERE season = ?
            """,
            [season],
        ).df()
        conn.close()

        if df.empty:
            return {"status": "error", "error_message": f"No plays found for season {season}."}

        df, feature_cols = prepare_features(df)
        df["completion_prob"] = _load_model().predict_proba(df[feature_cols])[:, 1].astype("float64")
        qb = compute_qb_cpoe(df)
        return {"status": "success", "rows": qb.to_dict(orient="records")}
    
    except Exception as e:
        return {"status": "error", "error_message": str(e)}