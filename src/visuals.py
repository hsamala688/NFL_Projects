import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------

def _create_football_field(orientation: str = "horizontal") -> go.Figure:
    """
    Draws a blank NFL football field using Plotly shapes.

    Parameters
    ----------
    orientation : str
        "horizontal"  → sideline-to-sideline on Y axis (0–53.3),
                         goal-line-to-goal-line on X axis (0–100).
                         Used by the passing chart (dots across the field).
        "vertical"    → field runs bottom-to-top on Y axis (0–60 visible yards),
                         hash-width on X axis (0–53.3).
                         Used by the heatmap (zones stacked vertically).

    Returns
    -------
    go.Figure with field shapes pre-drawn.
    """
    fig = go.Figure()

    if orientation == "horizontal":
        # ── Field background ──────────────────────────────────────────────
        fig.add_shape(
            type="rect", x0=0, y0=0, x1=100, y1=53.3,
            fillcolor="#1a472a", line=dict(width=0), layer="below"
        )
        # Outer boundary
        fig.add_shape(
            type="rect", x0=0, y0=0, x1=100, y1=53.3,
            line=dict(color="white", width=3), fillcolor="rgba(0,0,0,0)"
        )
        # End zones
        for x0, x1 in [(0, 10), (90, 100)]:
            fig.add_shape(
                type="rect", x0=x0, y0=0, x1=x1, y1=53.3,
                fillcolor="#145214", line=dict(width=0), layer="below"
            )
        # Yard lines every 10 yards + numbers
        for yard in range(10, 100, 10):
            fig.add_shape(
                type="line", x0=yard, y0=0, x1=yard, y1=53.3,
                line=dict(color="white", width=1.5)
            )
            label = yard if yard <= 50 else 100 - yard
            for y_pos, angle in [(4, 0), (49.3, 180)]:
                fig.add_annotation(
                    x=yard, y=y_pos, text=str(label), showarrow=False,
                    font=dict(color="white", size=11, family="Arial Black"),
                    textangle=angle
                )
        # Hash marks
        for yard in range(0, 101):
            for y0, y1 in [(18.5, 19.5), (33.8, 34.8)]:
                fig.add_shape(
                    type="line", x0=yard, y0=y0, x1=yard, y1=y1,
                    line=dict(color="white", width=1)
                )
        # Goal lines
        for x in [10, 90]:
            fig.add_shape(
                type="line", x0=x, y0=0, x1=x, y1=53.3,
                line=dict(color="white", width=3)
            )

    else:  # vertical
        # ── Field background ──────────────────────────────────────────────
        fig.add_shape(
            type="rect", x0=0, y0=0, x1=53.3, y1=60,
            fillcolor="#1a472a", line=dict(width=0), layer="below"
        )
        fig.add_shape(
            type="rect", x0=0, y0=0, x1=53.3, y1=60,
            line=dict(color="white", width=3), fillcolor="rgba(0,0,0,0)"
        )
        # Yard lines every 10 yards + labels on both sides
        for yard in [10, 20, 30, 40, 50, 60]:
            fig.add_shape(
                type="line", x0=0, y0=yard, x1=53.3, y1=yard,
                line=dict(color="white", width=2)
            )
            for x_pos in [3, 50.3]:
                fig.add_annotation(
                    x=x_pos, y=yard, text=str(yard), showarrow=False,
                    font=dict(color="white", size=10, family="Times New Roman")
                )
        # Hash marks every 5 yards
        for yard in range(0, 61, 5):
            for x0, x1 in [(18.5, 19.5), (33.8, 34.8)]:
                fig.add_shape(
                    type="line", x0=x0, y0=yard, x1=x1, y1=yard,
                    line=dict(color="white", width=1)
                )

    return fig


def _add_jitter(
    x_series: pd.Series,
    y_series: pd.Series,
    jitter: float = 1.2
) -> tuple[list, list]:
    """
    Spread overlapping dots radially so stacked passes are visible.

    Parameters
    ----------
    x_series, y_series : pd.Series
        Raw target coordinates.
    jitter : float
        Maximum radius of spread in yards.

    Returns
    -------
    (new_x, new_y) : tuple of lists
    """
    df = pd.DataFrame({"x": x_series, "y": y_series})
    new_x, new_y = [], []

    for (x, y), group in df.groupby(["x", "y"]):
        n = len(group)
        if n == 1:
            new_x.append(x)
            new_y.append(y)
        else:
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            radii = np.random.uniform(0.3 * jitter, jitter, n)
            new_x.extend(x + np.cos(angles) * radii)
            new_y.extend(y + np.sin(angles) * radii)

    return new_x, new_y


def _compute_target_coords(row: pd.Series) -> tuple[float, float]:
    """
    Converts nflverse pass_location + air_yards + yardline_100 into
    (x, y) field coordinates for the horizontal passing chart.

    X = yards from the left end zone (0 = left goal line, 100 = right).
    Y = lateral position on field (0 = bottom sideline, 53.3 = top).
    """
    y_map = {"left": 13.25, "middle": 26.65, "right": 40.05}
    y = y_map.get(str(row.get("pass_location", "middle")).lower(), 26.65)

    air = row.get("air_yards")
    los = row.get("yardline_100", 50)  # yards to opponent end zone

    if pd.notna(air):
        # yardline_100 counts from the offense's perspective (100 = own goal)
        x = 100 - los + float(air)
    else:
        x = 100 - float(los)

    return float(np.clip(x, 0, 100)), y


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def create_passing_chart(
    df: pd.DataFrame,
    passer_name: str,
    title: str | None = None,
) -> go.Figure:
    """
    Scatter-dot passing chart on a horizontal football field.

    Each pass is a dot color-coded by outcome:
        🟢 Green   → Completion
        ⚪ White   → Incomplete
        🔵 Blue    → Touchdown
        🔴 Red     → Interception

    Parameters
    ----------
    df : pd.DataFrame
        nflverse play-by-play data (Polars or Pandas).
        Required columns: passer_player_name, complete_pass, touchdown,
        interception, pass_location, air_yards, yardline_100,
        receiver_player_name, week, qtr, down, ydstogo, yards_gained.
    passer_name : str
        As it appears in nflverse, e.g. "B.Nix", "J.Herbert".
    title : str, optional
        Chart title. Auto-generated from passer_name if not provided.

    Returns
    -------
    go.Figure
    """
    # ── Normalize to Pandas if Polars DF passed in ────────────────────────
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    plays = df[
        (df["passer_player_name"] == passer_name) &
        (df["pass_attempt"] == 1)
    ].copy()

    if plays.empty:
        raise ValueError(
            f"No pass attempts found for '{passer_name}'. "
            "Check the passer_player_name format (e.g. 'B.Nix')."
        )

    # ── Compute target coordinates ────────────────────────────────────────
    coords = plays.apply(_compute_target_coords, axis=1)
    plays["target_x"] = [c[0] for c in coords]
    plays["target_y"] = [c[1] for c in coords]
    plays["plot_x"], plays["plot_y"] = _add_jitter(
        plays["target_x"], plays["target_y"]
    )

    # ── Segment by outcome ────────────────────────────────────────────────
    tds    = plays[plays["touchdown"] == 1]
    ints   = plays[plays["interception"] == 1]
    comps  = plays[(plays["complete_pass"] == 1) & (plays["touchdown"] == 0)]
    incomp = plays[(plays["complete_pass"] == 0) & (plays["interception"] == 0)]

    fig = _create_football_field("horizontal")

    # Common hover columns (safe defaults for missing cols)
    def safe_col(frame, col, default="N/A"):
        return frame[col] if col in frame.columns else default

    # ── Touchdowns ────────────────────────────────────────────────────────
    if not tds.empty:
        fig.add_trace(go.Scatter(
            x=tds["plot_x"], y=tds["plot_y"],
            mode="markers", name="Touchdown",
            marker=dict(size=16, color="#3B82F6",
                        line=dict(width=2, color="white"), opacity=0.95),
            customdata=tds[[
                "yardline_100", "receiver_player_name", "week",
                "qtr", "down", "ydstogo", "yards_gained",
                "air_yards", "pass_location"
            ]].values,
            hovertemplate=(
                "<b>🏈 TOUCHDOWN → %{customdata[1]}</b><br>"
                "Week %{customdata[2]:.0f} · Q%{customdata[3]:.0f}<br>"
                "Down & Distance: %{customdata[4]:.0f} & %{customdata[5]:.0f}<br>"
                "Yards Gained: %{customdata[6]:.0f} · Air Yards: %{customdata[7]:.0f}<br>"
                "Location: %{customdata[8]}<br>"
                "LOS: %{customdata[0]:.0f} yds to end zone"
                "<extra></extra>"
            ),
            hoverlabel=dict(bgcolor="#3B82F6", font_color="white", font_size=13)
        ))

    # ── Interceptions ─────────────────────────────────────────────────────
    if not ints.empty:
        fig.add_trace(go.Scatter(
            x=ints["plot_x"], y=ints["plot_y"],
            mode="markers", name="Interception",
            marker=dict(size=16, color="#EF4444",
                        line=dict(width=2, color="white"), opacity=0.95),
            customdata=ints[[
                "yardline_100", "receiver_player_name", "week",
                "qtr", "down", "ydstogo", "air_yards", "pass_location"
            ]].values,
            hovertemplate=(
                "<b>🚫 INTERCEPTION → %{customdata[1]}</b><br>"
                "Week %{customdata[2]:.0f} · Q%{customdata[3]:.0f}<br>"
                "Down & Distance: %{customdata[4]:.0f} & %{customdata[5]:.0f}<br>"
                "Air Yards: %{customdata[6]:.0f} · Location: %{customdata[7]}<br>"
                "LOS: %{customdata[0]:.0f} yds to end zone"
                "<extra></extra>"
            ),
            hoverlabel=dict(bgcolor="#EF4444", font_color="white", font_size=13)
        ))

    # ── Completions ───────────────────────────────────────────────────────
    if not comps.empty:
        fig.add_trace(go.Scatter(
            x=comps["plot_x"], y=comps["plot_y"],
            mode="markers", name="Complete",
            marker=dict(size=13, color="#22C55E",
                        line=dict(width=1.5, color="white"), opacity=0.85),
            customdata=comps[[
                "yardline_100", "receiver_player_name", "week",
                "qtr", "down", "ydstogo", "yards_gained",
                "air_yards", "pass_location"
            ]].values,
            hovertemplate=(
                "<b>✅ Complete → %{customdata[1]}</b><br>"
                "Week %{customdata[2]:.0f} · Q%{customdata[3]:.0f}<br>"
                "Down & Distance: %{customdata[4]:.0f} & %{customdata[5]:.0f}<br>"
                "Yards Gained: %{customdata[6]:.0f} · Air Yards: %{customdata[7]:.0f}<br>"
                "Location: %{customdata[8]}<br>"
                "LOS: %{customdata[0]:.0f} yds to end zone"
                "<extra></extra>"
            ),
            hoverlabel=dict(bgcolor="#166534", font_color="white", font_size=13)
        ))

    # ── Incomplete ────────────────────────────────────────────────────────
    if not incomp.empty:
        fig.add_trace(go.Scatter(
            x=incomp["plot_x"], y=incomp["plot_y"],
            mode="markers", name="Incomplete",
            marker=dict(size=12, color="white",
                        line=dict(width=1.5, color="#6B7280"), opacity=0.65),
            customdata=incomp[[
                "yardline_100", "receiver_player_name", "week",
                "qtr", "down", "ydstogo", "air_yards", "pass_location"
            ]].values,
            hovertemplate=(
                "<b>❌ Incomplete → %{customdata[1]}</b><br>"
                "Week %{customdata[2]:.0f} · Q%{customdata[3]:.0f}<br>"
                "Down & Distance: %{customdata[4]:.0f} & %{customdata[5]:.0f}<br>"
                "Air Yards: %{customdata[6]:.0f} · Location: %{customdata[7]}<br>"
                "LOS: %{customdata[0]:.0f} yds to end zone"
                "<extra></extra>"
            ),
            hoverlabel=dict(bgcolor="#374151", font_color="white", font_size=13)
        ))

    # ── Stats summary for title ───────────────────────────────────────────
    total   = len(plays)
    comp_n  = int(plays["complete_pass"].sum())
    comp_pct = (comp_n / total * 100) if total else 0
    td_n    = int(plays["touchdown"].sum())
    int_n   = int(plays["interception"].sum())
    yards   = int(plays["yards_gained"].sum()) if "yards_gained" in plays.columns else 0
    weeks   = sorted(plays["week"].unique()) if "week" in plays.columns else []
    week_range = (f"Weeks {min(weeks)}–{max(weeks)}" if len(weeks) > 1
                  else f"Week {weeks[0]}" if weeks else "")

    auto_title = title or f"{passer_name} Passing Chart"
    subtitle = (
        f"{week_range} | {comp_n}/{total} ({comp_pct:.1f}%) | "
        f"{yards} Yds | {td_n} TDs | {int_n} INTs"
    )

    fig.update_layout(
        title={
            "text": f"{auto_title}<br><sub>{subtitle}</sub>",
            "x": 0.5, "xanchor": "center",
            "font": {"size": 22, "color": "white", "family": "Arial Black"}
        },
        xaxis=dict(range=[-2, 102], showgrid=False, zeroline=False,
                   showticklabels=False, title=""),
        yaxis=dict(range=[-3, 56], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", scaleratio=1, title=""),
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="white"),
        height=600,
        hovermode="closest",
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="white", borderwidth=1,
            font=dict(size=13)
        )
    )
    return fig


# ---------------------------------------------------------------------------

def create_heatmap_chart(
    df: pd.DataFrame,
    passer_name: str,
    title: str | None = None,
) -> go.Figure:
    """
    Zone-based passing heatmap on a vertical football field.

    Divides the field into 6 zones (Short/Deep × Left/Middle/Right) and
    colors them by target volume. Hover shows top receivers per zone.

    Parameters
    ----------
    df : pd.DataFrame
        nflverse play-by-play data (Polars or Pandas).
        Required columns: passer_player_name (or desc), pass_length,
        pass_location, receiver_player_name, week.
    passer_name : str
        As it appears in nflverse, e.g. "B.Nix".
    title : str, optional
        Chart title.

    Returns
    -------
    go.Figure
    """
    # ── Normalize ─────────────────────────────────────────────────────────
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    # Filter to passer — support both nflverse column and desc-based
    if "passer_player_name" in df.columns:
        plays = df[df["passer_player_name"] == passer_name].copy()
    else:
        plays = df[df["desc"].str.contains(passer_name, na=False)].copy()

    if plays.empty:
        raise ValueError(f"No plays found for '{passer_name}'.")

    plays["pass_length"]   = plays["pass_length"].str.lower().fillna("unknown")
    plays["pass_location"] = plays["pass_location"].str.lower().fillna("unknown")

    # ── Zone geometry ─────────────────────────────────────────────────────
    SHORT_Y = (10, 30)
    DEEP_Y  = (30, 60)
    zones = {
        ("short", "left"):   {"x": [0,     17.77], "y": SHORT_Y},
        ("short", "middle"): {"x": [17.77, 35.53], "y": SHORT_Y},
        ("short", "right"):  {"x": [35.53, 53.3],  "y": SHORT_Y},
        ("deep",  "left"):   {"x": [0,     17.77], "y": DEEP_Y},
        ("deep",  "middle"): {"x": [17.77, 35.53], "y": DEEP_Y},
        ("deep",  "right"):  {"x": [35.53, 53.3],  "y": DEEP_Y},
    }

    # ── Per-zone stats ────────────────────────────────────────────────────
    zone_stats = {}
    for (depth, location), bounds in zones.items():
        zone_plays = plays[
            (plays["pass_length"]   == depth) &
            (plays["pass_location"] == location)
        ]
        count     = len(zone_plays)
        comp_rate = zone_plays["complete_pass"].mean() if count > 0 else 0.0

        recv_counts = (
            zone_plays["receiver_player_name"]
            .value_counts()
            .head(5)
        )
        recv_text = "<br>".join(
            f"{name}: {cnt}" for name, cnt in recv_counts.items()
        ) or "No targets"

        zone_stats[(depth, location)] = {
            "count":     count,
            "comp_rate": comp_rate,
            "receivers": recv_text,
        }

    max_count = max(s["count"] for s in zone_stats.values()) or 1

    # ── Build figure ──────────────────────────────────────────────────────
    fig = _create_football_field("vertical")

    # LOS marker
    los_y = 10
    fig.add_shape(
        type="line", x0=0, y0=los_y, x1=53.3, y1=los_y,
        line=dict(color="#60A5FA", width=5), layer="above"
    )
    fig.add_annotation(
        x=26.65, y=los_y + 1.5, text="<b>LOS</b>", showarrow=False,
        font=dict(color="#60A5FA", size=16, family="Times New Roman"),
        bgcolor="rgba(0,0,0,0.8)", borderpad=4
    )

    # ── Draw zones ────────────────────────────────────────────────────────
    for (depth, location), bounds in zones.items():
        stats     = zone_stats[(depth, location)]
        count     = stats["count"]
        comp_rate = stats["comp_rate"]
        intensity = count / max_count

        # Color: red → yellow → green by volume
        if intensity < 0.5:
            r = 255
            g = int(255 * intensity * 2)
            b = 0
        else:
            r = int(255 * (1 - (intensity - 0.5) * 2))
            g = 255
            b = 0
        fill_color = f"rgba({r},{g},{b},0.72)"

        cx = (bounds["x"][0] + bounds["x"][1]) / 2
        cy = (bounds["y"][0] + bounds["y"][1]) / 2

        # Invisible hover point
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="markers",
            marker=dict(size=0.1, opacity=0),
            hovertemplate=(
                f"<b>{depth.upper()} {location.upper()}</b><br>"
                f"Targets: {count}<br>"
                f"Comp %: {comp_rate*100:.1f}%<br>"
                f"<b>Top Receivers:</b><br>{stats['receivers']}"
                "<extra></extra>"
            ),
            hoverlabel=dict(
                bgcolor="rgba(30,30,30,0.95)",
                font_color="white", font_size=13,
                font_family="Times New Roman"
            ),
            showlegend=False
        ))

        # Colored zone rectangle
        fig.add_shape(
            type="rect",
            x0=bounds["x"][0], y0=bounds["y"][0],
            x1=bounds["x"][1], y1=bounds["y"][1],
            fillcolor=fill_color,
            line=dict(color="white", width=2),
            layer="below"
        )

        # Zone annotation
        comp_label = f"{comp_rate*100:.0f}% comp" if count > 0 else "—"
        fig.add_annotation(
            x=cx, y=cy,
            text=(
                f"<b>{count}</b><br>"
                f"{depth.upper()}<br>"
                f"{location.upper()}<br>"
                f"<sub>{comp_label}</sub>"
            ),
            showarrow=False,
            font=dict(size=13, color="white", family="Arial Black"),
            bgcolor="rgba(0,0,0,0.65)",
            borderpad=8
        )

    # ── Summary stats ─────────────────────────────────────────────────────
    total   = sum(s["count"] for s in zone_stats.values())
    weeks   = sorted(plays["week"].unique()) if "week" in plays.columns else []
    week_range = (f"Weeks {min(weeks)}–{max(weeks)}" if len(weeks) > 1
                  else f"Week {weeks[0]}" if weeks else "")

    auto_title = title or f"{passer_name} Target Distribution"
    subtitle   = f"{week_range} | {total} Total Targets"

    fig.update_layout(
        title={
            "text": f"{auto_title}<br><sub>{subtitle}</sub>",
            "x": 0.5, "xanchor": "center",
            "font": {"size": 22, "color": "white", "family": "Times New Roman"}
        },
        xaxis=dict(range=[-2, 55.3], showgrid=False, zeroline=False,
                   showticklabels=False),
        yaxis=dict(range=[-2, 62], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="white"),
        width=600, height=800,
        hovermode="closest"
    )
    return fig


# ---------------------------------------------------------------------------
# CONVENIENCE: per-week breakdowns
# ---------------------------------------------------------------------------

def save_weekly_charts(
    df: pd.DataFrame,
    passer_name: str,
    chart_type: str = "passing",
    output_dir: str = ".",
) -> list[str]:
    """
    Generates one chart per week and saves as HTML files.

    Parameters
    ----------
    df : pd.DataFrame
        Full nflverse PBP dataframe.
    passer_name : str
        e.g. "B.Nix"
    chart_type : str
        "passing" → scatter dot chart
        "heatmap" → zone heatmap
    output_dir : str
        Directory to write HTML files.

    Returns
    -------
    List of file paths written.
    """
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    if "passer_player_name" in df.columns:
        plays = df[df["passer_player_name"] == passer_name]
    else:
        plays = df[df["desc"].str.contains(passer_name, na=False)]

    weeks = sorted(plays["week"].unique())
    paths = []

    slug = passer_name.replace(".", "_").lower()
    fn_map = {"passing": create_passing_chart, "heatmap": create_heatmap_chart}
    chart_fn = fn_map.get(chart_type, create_passing_chart)

    for week in weeks:
        week_df = plays[plays["week"] == week]
        if week_df.empty:
            continue
        fig  = chart_fn(week_df, passer_name, title=f"{passer_name} — Week {week}")
        path = f"{output_dir}/{slug}_{chart_type}_week_{week}.html"
        fig.write_html(path)
        paths.append(path)
        print(f"  ✅ Saved {path}")

    return paths