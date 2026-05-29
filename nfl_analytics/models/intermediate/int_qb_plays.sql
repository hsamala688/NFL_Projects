with passing_plays as (
    select * from {{ ref('stg_pbp') }}
    where (complete_pass = 1 or incomplete_pass = 1 or interception = 1)
        and passer_player_name is not null
        and season_type = 'REG'
)

select
    play_id,
    game_id,
    season,
    week,
    possession_team,
    passer_player_id,
    passer_player_name,
    down,
    yards_to_go,
    yards_from_endzone,
    complete_pass                                    as is_complete,
    interception                                     as is_interception,
    air_yards,
    yards_after_catch,
    yards_gained,
    epa,
    win_probability_added,
    completion_pct_over_expected,
    case
        when yards_to_go <= 3  then 'short'
        when yards_to_go <= 7  then 'medium'
        else 'long'
    end                                              as distance_bucket
from passing_plays
