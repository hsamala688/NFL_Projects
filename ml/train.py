import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent / "db" / "nfl.duckdb"

# ── config ───────────────────────────────────────────────────────────────────
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
CATEGORICAL_FEATURES = ["pass_location"]
TARGET = "is_complete"


def load_data() -> pd.DataFrame:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute("""
        SELECT
            season,
            week,
            passer_player_name,
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
    """).df()
    conn.close()
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # drop nulls on any feature or target column
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    df = df.dropna(subset=feature_cols).copy()

    # one-hot encode pass_location (drop_first=False so all three are explicit)
    dummies = pd.get_dummies(df["pass_location"], prefix="loc", dtype=float)
    df = pd.concat([df.drop(columns=["pass_location"]), dummies], axis=1)

    return df


def split_data(df: pd.DataFrame):
    train = df[df["season"].isin([2020, 2021, 2022, 2023, 2024])]
    test  = df[df["season"] == 2025]

    # final feature list: numerics + one-hot columns
    feature_cols = NUMERIC_FEATURES + [c for c in df.columns if c.startswith("loc_")]

    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test  = test[feature_cols]
    y_test  = test[TARGET]

    return X_train, X_test, y_train, y_test, feature_cols


def evaluate(name: str, y_true, y_prob) -> None:
    auc  = roc_auc_score(y_true, y_prob)
    loss = log_loss(y_true, y_prob)
    print(f"\n{name}")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Log-loss: {loss:.4f}")


def plot_calibration(y_true, prob_lr, prob_xgb) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, probs in [("Logistic Regression", prob_lr), ("XGBoost", prob_xgb)]:
        fraction_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10)
        ax.plot(mean_pred, fraction_pos, marker="o", label=name)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of completions")
    ax.set_title("Calibration Plot — Pass Completion Model")
    ax.legend()
    plt.tight_layout()

    out_path = Path(__file__).parent / "calibration_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nCalibration plot saved to {out_path}")
    plt.show()


def main():
    print("Loading data...")
    df = load_data()
    print(f"  Raw rows: {len(df):,}")

    df = prepare_features(df)
    print(f"  After dropping nulls: {len(df):,}")

    X_train, X_test, y_train, y_test, feature_cols = split_data(df)
    print(f"  Train: {len(X_train):,} plays (2023–2024)")
    print(f"  Test:  {len(X_test):,} plays (2025)")

    # ── logistic regression ───────────────────────────────────────────────────
    print("\nTraining logistic regression baseline...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    evaluate("Logistic Regression", y_test, prob_lr)

    # ── xgboost ──────────────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    evaluate("XGBoost", y_test, prob_xgb)

    # ── feature importance ────────────────────────────────────────────────────
    print("\nXGBoost feature importances:")
    importances = pd.Series(
        xgb_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print(importances.to_string())

    # ── calibration plot ─────────────────────────────────────────────────────
    plot_calibration(y_test, prob_lr, prob_xgb)

    model_path = Path(__file__).parent / "xgb_model.json"
    xgb_model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()