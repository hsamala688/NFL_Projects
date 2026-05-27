SELECT
    passer_player_name,
    COUNT(*)  FILTER (WHERE complete_pass = 1)                          AS completions,
    COUNT(*)  FILTER (WHERE complete_pass = 1 OR incomplete_pass = 1)   AS attempts,
    ROUND(
        completions * 100.0 / NULLIF(attempts, 0)
    , 1)                                                                 AS completion_pct,
    ROUND(AVG(air_yards) FILTER (WHERE complete_pass = 1 OR incomplete_pass = 1), 1) AS avg_air_yards
FROM pbp
WHERE passer_player_name IS NOT NULL
  AND season_type = 'REG'
GROUP BY passer_player_name
HAVING attempts >= 100
ORDER BY completion_pct DESC;