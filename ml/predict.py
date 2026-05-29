import duckdb
import pandas as pd
import xgboost as xgb
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
DB_PATH    = Path(__file__).parent.parent / "db" / "nfl.duckdb"
MODEL_PATH = Path(__file__).parent / "xgb_model.json"

# ── must match train.py exactly ───────────────────────────────────────────────
NUMERIC_FEATURES = [
    "down",
    "yards_to_go",
    "yards_from_endzone",
    "air_yards",
    "score_differential",
    "game_seconds_remaining",
    "shotgun",
    "no_huddle",
]
TARGET = "is_complete"


def load_2025_plays() -> pd.DataFrame:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute("""
        SELECT
            passer_player_name,
            season,
            week,
            down,
            yards_to_go,
            yards_from_endzone,
            air_yards,
            pass_location,
            score_differential,
            game_seconds_remaining,
            shotgun,
            no_huddle,
            is_complete
        FROM int_qb_plays
        WHERE season = 2025
    """).df()
    conn.close()
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = NUMERIC_FEATURES + ["pass_location", TARGET]
    df = df.dropna(subset=feature_cols).copy()

    # one-hot encode pass_location — must match train.py
    dummies = pd.get_dummies(df["pass_location"], prefix="loc", dtype=float)
    df = pd.concat([df.drop(columns=["pass_location"]), dummies], axis=1)

    # ensure all three loc columns exist even if one value is missing in this slice
    for col in ["loc_left", "loc_middle", "loc_right"]:
        if col not in df.columns:
            df[col] = 0.0

    feature_cols = NUMERIC_FEATURES + ["loc_left", "loc_middle", "loc_right"]
    return df, feature_cols


def compute_qb_cpoe(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each QB in 2025:
      - actual_pct:   raw completion percentage
      - expected_pct: mean model-predicted probability across their attempts
      - model_cpoe:   actual_pct minus expected_pct
      - raw_rank:     rank by actual completion %
      - model_rank:   rank by model CPOE
      - rank_diff:    how many spots the QB moves between the two rankings
    """
    qb = df.groupby("passer_player_name").agg(
        attempts        =("is_complete", "count"),
        actual_pct      =("is_complete", "mean"),
        expected_pct    =("completion_prob", "mean"),
    ).reset_index()

    qb["model_cpoe"] = qb["actual_pct"] - qb["expected_pct"]

    # min 100 attempts — consistent with mart_qb_season threshold
    qb = qb[qb["attempts"] >= 100].copy()

    qb["actual_pct"]   = (qb["actual_pct"]   * 100).round(1)
    qb["expected_pct"] = (qb["expected_pct"]  * 100).round(1)
    qb["model_cpoe"]   = (qb["model_cpoe"]    * 100).round(1)

    qb["raw_rank"]   = qb["actual_pct"].rank(ascending=False).astype(int)
    qb["model_rank"] = qb["model_cpoe"].rank(ascending=False).astype(int)
    qb["rank_diff"]  = qb["raw_rank"] - qb["model_rank"]

    return qb.sort_values("model_rank")


def main():
    print("Loading 2025 plays...")
    df = load_2025_plays()
    print(f"  Plays loaded: {len(df):,}")

    df, feature_cols = prepare_features(df)
    print(f"  After dropping nulls: {len(df):,}")

    print("\nLoading model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    print("Generating per-play completion probabilities...")
    df["completion_prob"] = model.predict_proba(df[feature_cols])[:, 1]

    print("\n── QB Completion % vs Model Expected (2025, min 100 attempts) ──\n")
    qb_summary = compute_qb_cpoe(df)
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 120)
    print(qb_summary.to_string(index=False))

    out_path = Path(__file__).parent / "qb_cpoe_2025.csv"
    qb_summary.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()