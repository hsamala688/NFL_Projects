import streamlit as st
import duckdb
import pandas as pd
from pathlib import Path

# --- Data loading ---
# Path is built relative to this file's location, so it works no matter
# what directory you run `streamlit run` from.
DB_PATH = Path(__file__).parent.parent / "db" / "nfl.duckdb"

@st.cache_data
def load_data():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute("SELECT * FROM mart_qb_season").df()
    conn.close()
    return df

df = load_data()

# --- Header ---
st.title("🏈 NFL QB Analytics Dashboard")
st.caption("Data: nflverse play-by-play | Pipeline: nflreadpy → DuckDB → dbt")

seasons = sorted(df["season"].unique(), reverse=True)
selected_season = st.selectbox("Season", seasons)

df_season = df[df["season"] == selected_season]

# --- Leaderboard Table ---
st.subheader("QB Leaderboard")

display_cols = {
    "passer_player_name": "QB",
    "completions": "Cmp",
    "attempts": "Att",
    "completion_pct": "Cmp%",
    "avg_epa": "Avg EPA",
    "avg_air_yards": "Avg AY",
    "avg_cpoe": "CPOE",
    "interceptions": "INTs",
    "total_wpa": "Total WPA",
}

leaderboard = (
    df_season[list(display_cols.keys())]
    .rename(columns=display_cols)
    .sort_values("Avg EPA", ascending=False)
    .reset_index(drop=True)
)

# Round floats for readability
leaderboard["Cmp%"] = leaderboard["Cmp%"].round(1)
leaderboard["Avg EPA"] = leaderboard["Avg EPA"].round(3)
leaderboard["Avg AY"] = leaderboard["Avg AY"].round(1)
leaderboard["CPOE"] = leaderboard["CPOE"].round(2)
leaderboard["Total WPA"] = leaderboard["Total WPA"].round(2)

st.dataframe(leaderboard, use_container_width=True, hide_index=True)

# --- Scatter Plot ---
st.subheader("Completion % vs Avg EPA")
st.caption("Each point is a QB with 100+ attempts. Hover for name.")

import plotly.express as px

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

# --- Bar Chart ---
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

# --- QB Detail View ---
st.subheader("QB Detail View")

qb_list = sorted(df_season["passer_player_name"].unique())
selected_qb = st.selectbox("Select a QB", qb_list)

qb_row = df_season[df_season["passer_player_name"] == selected_qb].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Completion %", f"{qb_row['completion_pct']:.1f}%")
col2.metric("Avg EPA", f"{qb_row['avg_epa']:.3f}")
col3.metric("CPOE", f"{qb_row['avg_cpoe']:.2f}")
col4.metric("Total WPA", f"{qb_row['total_wpa']:.2f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Completions", int(qb_row["completions"]))
col6.metric("Attempts", int(qb_row["attempts"]))
col7.metric("Avg Air Yards", f"{qb_row['avg_air_yards']:.1f}")
col8.metric("INTs", int(qb_row["interceptions"]))