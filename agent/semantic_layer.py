"""
Semantic layer for the NFL analyst agent.

Single source of truth for what the agent knows about the schema before it
writes a query. Authored as structured data, rendered to a prompt string by
render_for_prompt(). The structured form is also importable directly, so the
Phase 5 query critic can pull the view whitelist or filter rules without
re-typing them.

Scope note: only mart_qb_season and int_qb_plays are exposed. stg_pbp is
intentionally not queryable. It carries the full raw nflverse surface, which
widens what the agent can get wrong, and everything the agent needs is already
present in the two views below.

Ranges below are the observed min/max over the 1999 to 2025 data, taken from
DuckDB SUMMARIZE. Re-run SUMMARIZE after ingesting a new season to refresh them.
"""

# Views the agent is allowed to read. query_marts rejects anything else.
QUERYABLE_VIEWS = ["mart_qb_season", "int_qb_plays"]


# Column dictionary: view -> column -> metadata.
#   meaning: plain English
#   type: SQL/python type
#   range: observed range over 1999 to 2025 (from SUMMARIZE)
#   renamed_from: the raw nflverse column it was renamed from, where applicable
COLUMNS = {
    "mart_qb_season": {
        "season": {
            "meaning": "NFL season year. Data covers 1999 to 2025.",
            "type": "int",
            "range": "1999 to 2025",
        },
        "passer_player_id": {
            "meaning": "nflverse unique player id (GSIS id) for the quarterback.",
            "type": "string",
            "range": "one value per QB",
        },
        "passer_player_name": {
            "meaning": "Quarterback name, first initial plus last name (e.g. B.Nix).",
            "type": "string",
            "range": "one row per QB per season",
        },
        "team": {
            "meaning": "Possession team abbreviation for the QB-season (e.g. DEN). 32 distinct current abbreviations; nflreadpy normalizes all historical codes (e.g. SD to LAC, STL to LA, OAK to LV).",
            "type": "string",
            "range": "32 distinct current abbreviations (nflreadpy normalizes all historical codes to current ones, e.g. SD to LAC, STL to LA, OAK to LV)",
            "renamed_from": "possession_team",
        },
        "attempts": {
            "meaning": "Pass attempts in the QB-season. One attempt is one row in int_qb_plays.",
            "type": "int",
            "range": "100 to 733 (100 minimum enforced by construction)",
        },
        "completions": {
            "meaning": "Completed passes (attempts where is_complete = 1).",
            "type": "int",
            "range": "44 to 490",
        },
        "completion_pct": {
            "meaning": (
                "Raw completion percentage, completions / attempts times 100, one decimal. "
                "Misleading on its own because it ignores throw difficulty. This is the metric "
                "the project argues against using alone. For 'most accurate QB' style questions, "
                "prefer model_cpoe from the run_cpoe tool."
            ),
            "type": "float",
            "range": "38.6 to 74.4",
        },
        "avg_epa": {
            "meaning": "Mean Expected Points Added per play. The leaderboard is sorted by this descending.",
            "type": "float",
            "range": "-0.483 to 0.549",
        },
        "avg_air_yards": {
            "meaning": "Mean air yards per attempt. A rough proxy for how aggressive the throw diet is. Null for seasons before 2006 (air_yards was not charted).",
            "type": "float",
            "range": "5.1 to 12.9 (null for seasons before 2006; about 26% of QB-seasons are null)",
        },
        "avg_yac": {
            "meaning": "Mean yards after catch on completed passes only. Null for seasons before 2006.",
            "type": "float",
            "range": "-1.0 to 11.7 (null for seasons before 2006; about 24% of QB-seasons are null)",
        },
        "avg_cpoe": {
            "meaning": (
                "Mean of nflverse's BUILT-IN completion percentage over expected (the cpoe field). "
                "This is NOT the project's model_cpoe. It comes from nflfastR's own model, not the "
                "XGBoost model. See METRICS for the distinction and which tool produces which. "
                "Null for seasons before 2006 (cpoe was not charted)."
            ),
            "type": "float",
            "range": "-14.35 to 10.78 percentage points (null for seasons before 2006; about 26% of QB-seasons are null)",
        },
        "interceptions": {
            "meaning": "Total interceptions thrown in the QB-season.",
            "type": "int",
            "range": "0 to 30",
        },
        "total_wpa": {
            "meaning": "Sum of Win Probability Added across the QB's plays.",
            "type": "float",
            "range": "-2.783 to 7.359",
        },
    },
    "int_qb_plays": {
        "play_id": {
            "meaning": "Identifier of the play within its game. Sequential within a game, not unique across games.",
            "type": "int",
            "range": "54 to 5461 within a game",
        },
        "game_id": {
            "meaning": "Unique game identifier (e.g. 2025_01_DEN_SEA).",
            "type": "string",
            "range": "one value per game (approx 5,158 games, 1999 to 2025)",
        },
        "season": {
            "meaning": "NFL season year. Data covers 1999 to 2025.",
            "type": "int",
            "range": "1999 to 2025",
        },
        "week": {
            "meaning": "Regular-season week.",
            "type": "int",
            "range": "1 to 18",
        },
        "possession_team": {
            "meaning": "Offense (passing) team abbreviation on the play. 32 distinct current abbreviations; nflreadpy normalizes all historical codes to current ones.",
            "type": "string",
            "range": "32 distinct current abbreviations (nflreadpy normalizes all historical codes to current ones)",
        },
        "passer_player_id": {
            "meaning": "nflverse unique player id (GSIS id) for the quarterback.",
            "type": "string",
            "range": "one value per QB",
        },
        "passer_player_name": {
            "meaning": "Quarterback name, first initial plus last name (e.g. B.Nix).",
            "type": "string",
            "range": "non-null by construction",
        },
        "down": {
            "meaning": "The down on which the pass was attempted.",
            "type": "int",
            "range": "1 to 4",
        },
        "yards_to_go": {
            "meaning": "Yards needed for a first down at the snap.",
            "type": "int",
            "range": "1 to 50",
            "renamed_from": "ydstogo",
        },
        "yards_from_endzone": {
            "meaning": "Distance to the opponent end zone at the snap (field position).",
            "type": "int",
            "range": "1 to 99",
            "renamed_from": "yardline_100",
        },
        "is_complete": {
            "meaning": "Completion label, 1 if the pass was caught else 0. The ML target.",
            "type": "int (0/1)",
            "range": "{0, 1}",
            "renamed_from": "complete_pass",
        },
        "is_interception": {
            "meaning": "1 if the pass was intercepted else 0.",
            "type": "int (0/1)",
            "range": "{0, 1}",
            "renamed_from": "interception",
        },
        "air_yards": {
            "meaning": "Yards the ball traveled in the air past the line of scrimmage. Dominant completion-probability feature. Null before 2006 (not charted); about 25% null overall.",
            "type": "float",
            "range": "-93 to 78 (mostly 0 to 50; large negatives are laterals or charting quirks; null before 2006, about 25% null overall)",
        },
        "yards_after_catch": {
            "meaning": "Yards gained after the catch. Null on non-completions, so it exists for about 65% of rows.",
            "type": "float",
            "range": "-72 to 91 on completions (about 53% null overall; higher rate because the column is absent before 2006)",
        },
        "yards_gained": {
            "meaning": "Total yards gained on the play.",
            "type": "float",
            "range": "-24 to 99",
        },
        "epa": {
            "meaning": "Expected Points Added on the play.",
            "type": "float",
            "range": "-13.15 to 8.93 per play",
        },
        "win_probability_added": {
            "meaning": "Change in win probability attributable to the play.",
            "type": "float",
            "range": "-1.0 to 1.0 per play",
            "renamed_from": "wpa",
        },
        "completion_pct_over_expected": {
            "meaning": (
                "nflverse's BUILT-IN per-play CPOE in percentage points (actual completion minus "
                "nflfastR's expected completion probability). This is the per-play source of avg_cpoe "
                "in the mart. It is NOT the project's model_cpoe. See METRICS. "
                "Null before 2006 (not charted); about 27% null overall."
            ),
            "type": "float",
            "range": "-92 to 85 per play (null before 2006, about 27% null overall)",
            "renamed_from": "cpoe",
        },
        "pass_location": {
            "meaning": "Side of the field targeted. One-hot encoded for the model into loc_left, loc_middle, loc_right. Null before 2006 (not charted); about 25% null overall.",
            "type": "string",
            "range": "{left, middle, right} (null before 2006, about 25% null overall)",
        },
        "score_differential": {
            "meaning": "Possession team score minus opponent score at the snap. Negative means trailing.",
            "type": "int",
            "range": "-59 to 59",
        },
        "game_seconds_remaining": {
            "meaning": "Seconds remaining in the game at the snap.",
            "type": "int",
            "range": "0 to 3600",
        },
        "shotgun": {
            "meaning": "1 if the play was run from shotgun formation else 0.",
            "type": "int (0/1)",
            "range": "{0, 1}",
        },
        "no_huddle": {
            "meaning": "1 if the play was run no-huddle else 0.",
            "type": "int (0/1)",
            "range": "{0, 1}",
        },
        "distance_bucket": {
            "meaning": "Derived grouping of yards_to_go: short (<= 3), medium (<= 7), long (> 7).",
            "type": "string",
            "range": "{short, medium, long}",
        },
    },
}


# Filter rules the agent must respect. Stated once, here, so it cannot reinvent
# them wrong. These reflect what the dbt models actually do, not generic advice.
FILTER_RULES = [
    "int_qb_plays already contains only regular-season passing plays. The passing-play filter "
    "(complete_pass = 1 OR incomplete_pass = 1 OR interception = 1) and the season_type = 'REG' "
    "filter are applied upstream in dbt. Do not re-apply them. The raw columns complete_pass, "
    "incomplete_pass, pass_attempt, and season_type are NOT present in int_qb_plays; only the "
    "outcome flags is_complete and is_interception survive. Referencing the raw columns will error.",
    "In int_qb_plays, one row is one pass attempt. attempts = count of rows; completions = count "
    "of rows where is_complete = 1.",
    "mart_qb_season already contains only QB-seasons with at least 100 attempts. The threshold is "
    "enforced by a HAVING clause in the model. Do not re-apply it.",
    "For QB-season aggregates (completion %, EPA, air yards, interceptions, WPA), query "
    "mart_qb_season directly rather than re-aggregating int_qb_plays, so the 100-attempt rule and "
    "the model's rounding stay consistent.",
    "Use int_qb_plays for per-play or situational slices (by down, distance_bucket, pass_location, "
    "field position, etc.) that the season mart does not pre-aggregate.",
    "yards_after_catch is null on non-completions and completion_pct_over_expected is null on a "
    "small share of plays. SQL AVG ignores nulls, but COUNT(*) and COUNT(column) will differ on "
    "these columns. Use AVG for means and be explicit about the denominator when counting.",
]


# Metric definitions. The agent reads these instead of guessing. The two CPOEs
# are the conflation most likely to break tool selection, so they are stated
# explicitly and tied to the tool that produces each.
METRICS = {
    "model_cpoe": (
        "The project's headline metric. Actual completion percentage minus the XGBoost model's "
        "expected completion probability over the same plays. Positive means the QB completes more "
        "than the difficulty of their throws predicts. Produced ONLY by the run_cpoe tool, never by "
        "querying a column. Use this for expectation-adjusted questions such as 'who is the most "
        "accurate QB once you account for what they were asked to throw'."
    ),
    "avg_cpoe / completion_pct_over_expected": (
        "nflverse's built-in CPOE, available as columns (avg_cpoe in mart_qb_season, "
        "completion_pct_over_expected per play in int_qb_plays). Conceptually similar to model_cpoe "
        "but from nflfastR's own model, not the project's XGBoost model. Prefer model_cpoe for "
        "expectation-adjusted questions. Mention avg_cpoe only when the user asks about it directly."
    ),
    "completion_pct": (
        "Raw completion percentage. Useful as a factual lookup but misleading as a measure of skill "
        "because it ignores throw difficulty and schedule. When a question is really about accuracy "
        "or who is 'good', surface model_cpoe and explain why the raw ranking would mislead."
    ),
    "EPA / avg_epa": "Expected Points Added. The change in expected points from a play, measuring play value in points.",
    "WPA / total_wpa / win_probability_added": "Win Probability Added. The change in win probability attributable to a play.",
    "outperforming expectation": "model_cpoe greater than 0. The QB completes more passes than the situation predicts.",
}


# The abstention boundary. The play-by-play does not contain these, so the agent
# answers the measurable part and explicitly declines the rest, naming what would
# be needed.
OUT_OF_SCOPE = [
    "injuries and player health",
    "defensive coverage and scheme",
    "offensive line performance, pressure, and pass protection",
    "weather and field conditions",
    "play-calling intent or coaching decisions",
    "receiver separation, drops, and anything requiring film or tracking data",
]


def render_for_prompt() -> str:
    """Render the structured layer into the string injected into the agent's system prompt."""
    lines = ["# NFL schema and rules\n"]
    lines.append(
        f"You may issue read-only SELECT queries against these views only: "
        f"{', '.join(QUERYABLE_VIEWS)}. Any other table, and any non-SELECT statement, is rejected.\n"
    )

    for view, cols in COLUMNS.items():
        lines.append(f"\n## {view}")
        for name, meta in cols.items():
            rename = f" (renamed from {meta['renamed_from']})" if "renamed_from" in meta else ""
            lines.append(f"- {name}{rename}: {meta['meaning']} ")

    lines.append("\n## Filter rules")
    lines += [f"- {rule}" for rule in FILTER_RULES]

    lines.append("\n## Metric definitions")
    lines += [f"- {term}: {desc}" for term, desc in METRICS.items()]

    lines.append("\n## Out of scope (abstain on these, name what would be needed)")
    lines += [f"- {item}" for item in OUT_OF_SCOPE]

    return "\n".join(lines)


if __name__ == "__main__":
    print(render_for_prompt())