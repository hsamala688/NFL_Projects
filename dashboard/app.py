import streamlit as st
import duckdb
import pandas as pd
import xgboost as xgb
import plotly.express as px
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
DB_PATH    = Path(__file__).parent.parent / "db" / "nfl.duckdb"
MODEL_PATH = Path(__file__).parent.parent / "ml" / "xgb_model.json"

# ── feature list — must match train.py exactly ────────────────────────────────
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

# ── data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_season_data():
    """Load mart_qb_season from DuckDB — used by the Season Stats tab."""
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute("SELECT * FROM mart_qb_season").df()
    conn.close()
    return df


@st.cache_resource
def load_model():
    """Load the XGBoost model once and keep it in memory across interactions.
    cache_resource is used (not cache_data) because model objects are not
    serializable by Streamlit's data cache."""
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


@st.cache_data
def load_ml_predictions(_model):
    """
    Query int_qb_plays for the 2025 season, run predictions through the
    saved XGBoost model, and return the aggregated QB CPOE table.

    The leading underscore on _model tells Streamlit not to try to hash the
    XGBoost object (which would fail). The cache key is based on the other
    arguments — since there are none, this runs exactly once per session.
    """
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute("""
        SELECT
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
        WHERE season = 2025
    """).df()
    conn.close()

    # Drop nulls on all feature + target columns — mirrors predict.py
    feature_cols = NUMERIC_FEATURES + ["pass_location", "is_complete"]
    df = df.dropna(subset=feature_cols).copy()

    # One-hot encode pass_location — must match train.py exactly
    dummies = pd.get_dummies(df["pass_location"], prefix="loc", dtype=float)
    df = pd.concat([df.drop(columns=["pass_location"]), dummies], axis=1)

    # Safety check: guarantee all three loc columns exist even if a value
    # is absent in this slice (mirrors the safety check in predict.py)
    for col in ["loc_left", "loc_middle", "loc_right"]:
        if col not in df.columns:
            df[col] = 0.0

    final_features = NUMERIC_FEATURES + ["loc_left", "loc_middle", "loc_right"]

    # Generate per-play completion probabilities
    df["completion_prob"] = _model.predict_proba(df[final_features])[:, 1]

    # Aggregate to QB level — mirrors compute_qb_cpoe() in predict.py
    qb = df.groupby("passer_player_name").agg(
        attempts     =("is_complete", "count"),
        actual_pct   =("is_complete", "mean"),
        expected_pct =("completion_prob", "mean"),
    ).reset_index()

    qb["model_cpoe"] = qb["actual_pct"] - qb["expected_pct"]

    # Min 100 attempts — consistent with mart_qb_season threshold
    qb = qb[qb["attempts"] >= 100].copy()

    # Convert to percentages
    qb["actual_pct"]   = (qb["actual_pct"]   * 100).round(1)
    qb["expected_pct"] = (qb["expected_pct"]  * 100).round(1)
    qb["model_cpoe"]   = (qb["model_cpoe"]    * 100).round(1)

    qb["raw_rank"]   = qb["actual_pct"].rank(ascending=False).astype(int)
    qb["model_rank"] = qb["model_cpoe"].rank(ascending=False).astype(int)
    qb["rank_diff"]  = qb["raw_rank"] - qb["model_rank"]

    return qb.sort_values("model_rank").reset_index(drop=True)


# ── app ───────────────────────────────────────────────────────────────────────

st.title("🏈 NFL QB Analytics Dashboard")
st.caption("Data: nflverse play-by-play | Pipeline: nflreadpy → DuckDB → dbt")

# Load ML model and predictions here — above both tabs — so Tab 1's
# QB detail view can access qb_ml without depending on Tab 2 having run first.
# Both functions are cached so this costs nothing on subsequent interactions.
model = load_model()
qb_ml = load_ml_predictions(model)

tab1, tab2 = st.tabs(["📊 Season Stats", "🤖 ML — Model CPOE"])

# ── Tab 1: Season Stats ───────────────────────────────────────────────────────
with tab1:
    df = load_season_data()

    seasons = sorted(df["season"].unique(), reverse=True)
    selected_season = st.selectbox("Season", seasons)

    df_season = df[df["season"] == selected_season]

    # Leaderboard Table
    st.subheader("QB Leaderboard")

    display_cols = {
        "passer_player_name": "QB",
        "completions":        "Cmp",
        "attempts":           "Att",
        "completion_pct":     "Cmp%",
        "avg_epa":            "Avg EPA",
        "avg_air_yards":      "Avg AY",
        "avg_cpoe":           "CPOE",
        "interceptions":      "INTs",
        "total_wpa":          "Total WPA",
    }

    leaderboard = (
        df_season[list(display_cols.keys())]
        .rename(columns=display_cols)
        .sort_values("Avg EPA", ascending=False)
        .reset_index(drop=True)
    )

    leaderboard["Cmp%"]      = leaderboard["Cmp%"].round(1)
    leaderboard["Avg EPA"]   = leaderboard["Avg EPA"].round(3)
    leaderboard["Avg AY"]    = leaderboard["Avg AY"].round(1)
    leaderboard["CPOE"]      = leaderboard["CPOE"].round(2)
    leaderboard["Total WPA"] = leaderboard["Total WPA"].round(2)

    st.dataframe(leaderboard, use_container_width=True, hide_index=True)

    # Scatter Plot
    st.subheader("Completion % vs Avg EPA")
    st.caption("Each point is a QB with 100+ attempts. Hover for name.")

    fig = px.scatter(
        df_season,
        x="completion_pct",
        y="avg_epa",
        text="passer_player_name",
        hover_name="passer_player_name",
        hover_data={"completion_pct": ":.1f", "avg_epa": ":.3f", "attempts": True},
        labels={"completion_pct": "Completion %", "avg_epa": "Avg EPA"},
    )
    fig.update_traces(textposition="top center", marker=dict(size=8))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Bar Chart
    st.subheader("Top 10 QBs by Avg EPA")

    top10 = df_season.nlargest(10, "avg_epa").sort_values("avg_epa", ascending=True)
    fig2 = px.bar(
        top10,
        x="avg_epa",
        y="passer_player_name",
        orientation="h",
        labels={"avg_epa": "Avg EPA", "passer_player_name": "QB"},
        text=top10["avg_epa"].round(3),
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # QB Detail View
    st.subheader("QB Detail View")

    qb_list = sorted(df_season["passer_player_name"].unique())
    selected_qb = st.selectbox("Select a QB", qb_list)

    qb_row = df_season[df_season["passer_player_name"] == selected_qb].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Completion %", f"{qb_row['completion_pct']:.1f}%")
    col2.metric("Avg EPA",      f"{qb_row['avg_epa']:.3f}")
    col3.metric("CPOE",         f"{qb_row['avg_cpoe']:.2f}")
    col4.metric("Total WPA",    f"{qb_row['total_wpa']:.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Completions",   int(qb_row["completions"]))
    col6.metric("Attempts",      int(qb_row["attempts"]))
    col7.metric("Avg Air Yards", f"{qb_row['avg_air_yards']:.1f}")
    col8.metric("INTs",          int(qb_row["interceptions"]))

    # ML CPOE row — only available for 2025, min 100 attempts
    st.divider()
    ml_row = qb_ml[qb_ml["passer_player_name"] == selected_qb]

    if selected_season != 2025:
        st.caption("🤖 Model CPOE data is only available for the 2025 season.")
    elif ml_row.empty:
        st.caption(f"🤖 {selected_qb} does not meet the 100-attempt minimum for model CPOE.")
    else:
        ml = ml_row.iloc[0]
        st.caption("🤖 Model CPOE — XGBoost completion probability model (2025)")

        rank_delta = int(ml["rank_diff"])

        col9, col10, col11, col12 = st.columns(4)
        col9.metric("Model CPOE", f"{ml['model_cpoe']:+.1f}%")
        col10.metric("Expected Cmp%", f"{ml['expected_pct']:.1f}%")
        col11.metric("Model Rank", f"#{int(ml['model_rank'])}", delta=f"{rank_delta:+d} spots vs raw",)
        col12.metric("Raw Rank", f"#{int(ml['raw_rank'])}")


# ── Tab 2: ML — Model CPOE ────────────────────────────────────────────────────
with tab2:
    st.subheader("Model CPOE — 2025 Season")
    st.caption(
        "Completion % Over Expected (CPOE) from a trained XGBoost model. "
        "Positive = completing passes above what the situation predicts. "
        "Min 100 attempts."
    )

    # ── Leaderboard table ─────────────────────────────────────────────────────
    st.subheader("QB CPOE Leaderboard")

    ml_display_cols = {
        "passer_player_name": "QB",
        "attempts":           "Att",
        "actual_pct":         "Actual Cmp%",
        "expected_pct":       "Expected Cmp%",
        "model_cpoe":         "Model CPOE",
        "raw_rank":           "Raw Rank",
        "model_rank":         "Model Rank",
        "rank_diff":          "Rank Δ",
    }

    ml_leaderboard = (
        qb_ml[list(ml_display_cols.keys())]
        .rename(columns=ml_display_cols)
        .reset_index(drop=True)
    )

    st.dataframe(ml_leaderboard, use_container_width=True, hide_index=True)

    # ── CPOE Bar Chart — top 10 and bottom 10 ─────────────────────────────────
    st.subheader("Top & Bottom 10 QBs by Model CPOE")

    top10_cpoe    = qb_ml.nlargest(10, "model_cpoe")
    bottom10_cpoe = qb_ml.nsmallest(10, "model_cpoe")
    bar_df = pd.concat([top10_cpoe, bottom10_cpoe]).sort_values("model_cpoe", ascending=True)

    fig3 = px.bar(
        bar_df,
        x="model_cpoe",
        y="passer_player_name",
        orientation="h",
        color="model_cpoe",
        color_continuous_scale=["#d62728", "#ffffff", "#2ca02c"],
        color_continuous_midpoint=0,
        labels={"model_cpoe": "Model CPOE", "passer_player_name": "QB"},
        text=bar_df["model_cpoe"].apply(lambda v: f"{v:+.1f}"),
    )
    fig3.update_traces(textposition="outside")
    fig3.update_layout(height=550, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Scatter: raw completion % vs model CPOE ───────────────────────────────
    st.subheader("Raw Completion % vs Model CPOE")
    st.caption(
        "QBs above the horizontal zero line are outperforming their situations. "
        "QBs to the right have high raw completion % — but that may reflect easy throw schedules."
    )

    fig4 = px.scatter(
        qb_ml,
        x="actual_pct",
        y="model_cpoe",
        text="passer_player_name",
        hover_name="passer_player_name",
        hover_data={
            "actual_pct":   ":.1f",
            "expected_pct": ":.1f",
            "model_cpoe":   ":.1f",
            "attempts":     True,
        },
        labels={"actual_pct": "Raw Completion %", "model_cpoe": "Model CPOE"},
    )

    # Zero reference line — QBs above this line are outperforming expectations
    fig4.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
    fig4.update_traces(textposition="top center", marker=dict(size=8))
    fig4.update_layout(height=520)
    st.plotly_chart(fig4, use_container_width=True)