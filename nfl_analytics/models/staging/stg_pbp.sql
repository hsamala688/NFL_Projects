with source as (
    select * from {{ source('nflverse', 'pbp') }}
)

select
    -- identifiers
    play_id,
    game_id,
    season,
    week,
    season_type,
    home_team,
    away_team,
    posteam                 as possession_team,
    defteam                 as defense_team,

    -- play context
    down,
    ydstogo                 as yards_to_go,
    yardline_100            as yards_from_endzone,
    play_type,
    quarter_seconds_remaining,
    half_seconds_remaining,
    game_seconds_remaining,

    -- passer info
    passer_player_id,
    passer_player_name,

    -- pass outcome columns
    complete_pass,
    incomplete_pass,
    interception,
    pass_attempt,
    air_yards,
    yards_after_catch,
    yards_gained,

    -- advanced metrics
    epa,
    wp                      as win_probability,
    wpa                     as win_probability_added,
    cpoe                    as completion_pct_over_expected

from source
