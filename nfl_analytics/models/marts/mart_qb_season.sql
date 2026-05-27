with qb_plays as (
    select * from {{ ref('int_qb_plays') }}
)

select
    season,
    passer_player_id,
    passer_player_name,
    possession_team                                         as team,

    -- volume
    count(*)                                                as attempts,
    sum(is_complete)                                        as completions,

    -- efficiency
    round(sum(is_complete) / count(*)::double * 100, 1)    as completion_pct,
    round(avg(epa), 3)                                      as avg_epa,
    round(avg(air_yards), 1)                                as avg_air_yards,
    round(avg(yards_after_catch)
          filter (where is_complete = 1), 1)                as avg_yac,
    round(avg(completion_pct_over_expected), 2)             as avg_cpoe,

    -- turnover
    sum(is_interception)                                    as interceptions,

    -- win impact
    round(sum(win_probability_added), 3)                    as total_wpa

from qb_plays
group by season, passer_player_id, passer_player_name, possession_team
having count(*) >= 100   -- same 100-attempt threshold as qb_completion.sql
order by avg_epa desc
