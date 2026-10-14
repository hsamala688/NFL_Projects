import argparse

import duckdb
import pandas as pd
import xgboost as xgb
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
DB_PATH    = Path(__file__).parent.parent / "db" / "nfl.duckdb"
MODEL_PATH = Path(__file__).parent / "xgb_model.json"

# ── feature contract ──────────────────────────────────────────────────────────
# The saved model records the exact columns, in the exact order, it was
# fitted on. Reading that list back is the single source of truth for
# inference, so predict.py and the dashboard cannot drift away from
# train.py through hand-maintenance.
TARGET = "is_complete"
CATEGORICAL_FEATURE = "pass_location"
ONE_HOT_PREFIX = "loc"

_feature_names: list[str] | None = None


def load_feature_names(model_path: Path = MODEL_PATH) -> list[str]:
    """Read the authoritative ordered feature list off the saved model."""
    global _feature_names
    if _feature_names is None:
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        names = booster.feature_names
        if not names:
            raise ValueError(
                f"{model_path} records no feature_names. Retrain with "
                f"train.py, which writes them when it saves the model."
            )
        if len(names) != booster.num_features():
            raise ValueError(
                f"{model_path} is inconsistent: {len(names)} feature_names "
                f"but num_feature is {booster.num_features()}."
            )
        _feature_names = list(names)
    return _feature_names


def load_plays(season: int) -> pd.DataFrame:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute(
        """
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
        WHERE season = ?
        """,
        [season],
    ).df()
    conn.close()
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build exactly the columns the model was fitted on, in the model's order.

    Returns the frame plus the ordered feature list, so callers select with
    df[feature_cols] rather than assuming a layout.
    """
    feature_cols = load_feature_names()

    # Everything the model wants that is not one-hot encoded comes straight
    # off the query, so those are the columns a null disqualifies a play on.
    raw_cols = [c for c in feature_cols if not c.startswith(f"{ONE_HOT_PREFIX}_")]
    df = df.dropna(subset=raw_cols + [CATEGORICAL_FEATURE, TARGET]).copy()

    dummies = pd.get_dummies(
        df[CATEGORICAL_FEATURE], prefix=ONE_HOT_PREFIX, dtype=float
    )
    df = pd.concat([df.drop(columns=[CATEGORICAL_FEATURE]), dummies], axis=1)

    # The model fixes which one-hot columns exist. Supply as zeros any the
    # slice never saw. A category the model was never trained on simply goes
    # unselected here, instead of shifting every column to its right.
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df, feature_cols


def compute_qb_cpoe(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each QB in the season:
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
    parser = argparse.ArgumentParser(
        description="Rank QBs by model-based CPOE for a season."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="Season to score. air_yards was not charted before 2006.",
    )
    season = parser.parse_args().season

    print(f"Loading {season} plays...")
    df = load_plays(season)
    print(f"  Plays loaded: {len(df):,}")

    df, feature_cols = prepare_features(df)
    print(f"  After dropping nulls: {len(df):,}")

    print("\nLoading model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    print("Generating per-play completion probabilities...")
    df["completion_prob"] = model.predict_proba(df[feature_cols])[:, 1]

    print(f"\n── QB Completion % vs Model Expected ({season}, min 100 attempts) ──\n")
    qb_summary = compute_qb_cpoe(df)
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 120)
    print(qb_summary.to_string(index=False))

    out_path = Path(__file__).parent / f"qb_cpoe_{season}.csv"
    qb_summary.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()