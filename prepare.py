# Builds one row per team and season (outputs/team_seasons.csv).
# Each row has the targets of that season and features we already know
# the day before the season starts: the team and its history, the coach and
# their history, and the players of the team and their history.
# Run make_db.py first, then: python prepare.py

import sqlite3

import pandas as pd

DB_PATH = "outputs/hockey.db"
FIRST_YEAR = 1980
NEXT_YEAR = 2010  # season we have to predict (not in the data)

conn = sqlite3.connect(DB_PATH)


# ---------- 1. team stats of each season ----------
# The NEXT_YEAR season is not in the data, so we copy the teams of the
# last season with empty stats. Their features come from the seasons before.

conn.executescript(f"""
DROP TABLE IF EXISTS team_stats;
CREATE TEMP TABLE team_stats AS
SELECT year, tmID, franchID, confID, divID, name, G, W, Pts, GF, GA, rank, playoff,
       Pts * 1.0 / (2 * G) AS pts_pct,
       GF * 1.0 / G AS gf_pg,
       GA * 1.0 / G AS ga_pg,
       PPG * 1.0 / PPC AS pp_pct,
       1 - PKG * 1.0 / PKC AS pk_pct,
       PIM * 1.0 / G AS pim_pg,
       playoff IS NOT NULL AS made_playoffs
FROM Teams
WHERE lgID = 'NHL'

UNION ALL

SELECT {NEXT_YEAR}, tmID, franchID, confID, divID, name, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
       NULL, NULL, NULL, NULL, NULL, NULL, NULL
FROM Teams
WHERE lgID = 'NHL' AND year = {NEXT_YEAR - 1};
""")


# ---------- 2. players of each season and their history ----------
# Players of the team this season (known before the season starts),
# with their own stats from the season before, on any team.
# The data lists every player who played for the team during the season,
# including call-ups and trades. Bad teams use more players, so counting all
# of them would leak how the season went. We only use the 18 best players
# by last season's points (about one lineup).
# 2004 is missing, so for 2005 the season before is 2003.

conn.executescript("""
DROP TABLE IF EXISTS player_year;
CREATE TEMP TABLE player_year AS
SELECT playerID, year, SUM(GP) AS GP, SUM(G) AS G, SUM(Pts) AS Pts
FROM Scoring
WHERE lgID = 'NHL'
GROUP BY playerID, year;

DROP TABLE IF EXISTS roster;
CREATE TEMP TABLE roster AS
SELECT DISTINCT playerID, year, tmID,
       CASE WHEN year = 2005 THEN 2003 ELSE year - 1 END AS prev_year
FROM Scoring
WHERE lgID = 'NHL';

DROP TABLE IF EXISTS player_stats;
CREATE TEMP TABLE player_stats AS
WITH players AS (
    SELECT r.year, r.tmID, r.playerID,
           r.year - m.birthYear AS age,
           last.GP AS last_GP, last.G AS last_G, last.Pts AS last_Pts,
           ROW_NUMBER() OVER (PARTITION BY r.year, r.tmID ORDER BY last.Pts DESC) AS pts_order,
           (SELECT COUNT(*) FROM AwardsPlayers a WHERE a.playerID = r.playerID AND a.year < r.year) AS awards
    FROM roster r
    LEFT JOIN Master m ON m.playerID = r.playerID
    LEFT JOIN player_year last ON last.playerID = r.playerID AND last.year = r.prev_year
)
SELECT year, tmID,
       AVG(age) AS avg_age,
       COALESCE(SUM(last_Pts), 0) AS last_season_pts,
       AVG(CASE WHEN pts_order <= 3 AND last_GP > 0 THEN last_Pts * 1.0 / last_GP END) AS top3_pts_pg,
       SUM(last_G >= 20) AS n_20_goal_players,
       SUM(awards) AS player_awards
FROM players
WHERE pts_order <= 18
GROUP BY year, tmID;

DROP TABLE IF EXISTS goalie_stats;
CREATE TEMP TABLE goalie_stats AS
WITH goalie_year AS (
    SELECT playerID, year, SUM(GA) AS GA, SUM(SA) AS SA
    FROM Goalies
    WHERE lgID = 'NHL'
    GROUP BY playerID, year
),
goalie_roster AS (
    SELECT DISTINCT playerID, year, tmID,
           CASE WHEN year = 2005 THEN 2003 ELSE year - 1 END AS prev_year
    FROM Goalies
    WHERE lgID = 'NHL'
)
SELECT r.year, r.tmID,
       1 - SUM(last.GA) * 1.0 / SUM(last.SA) AS goalie_save_pct
FROM goalie_roster r
JOIN goalie_year last ON last.playerID = r.playerID AND last.year = r.prev_year
WHERE last.SA > 0
GROUP BY r.year, r.tmID;
""")

conn.executescript("""
DROP TABLE IF EXISTS playoff_wins;
CREATE TEMP TABLE playoff_wins AS
SELECT year, tmID, W AS playoff_wins
FROM TeamsPost
WHERE lgID = 'NHL';
""")


# ---------- 3. coaches of each season ----------
# first_coach = coach at the start of the season, last_coach = coach at the end.
# stint > 1 means the coach was changed during the season (target of task b).

conn.executescript("""
DROP TABLE IF EXISTS coach_season;
CREATE TEMP TABLE coach_season AS
WITH c AS (SELECT * FROM Coaches WHERE lgID = 'NHL'),
     stints AS (SELECT year, tmID, MAX(stint) AS max_stint FROM c GROUP BY year, tmID)
SELECT st.year, st.tmID, st.max_stint,
       (SELECT MIN(coachID) FROM c WHERE c.year = st.year AND c.tmID = st.tmID AND c.stint = 1) AS first_coach,
       (SELECT MIN(coachID) FROM c WHERE c.year = st.year AND c.tmID = st.tmID AND c.stint = st.max_stint) AS last_coach
FROM stints st;
""")


# ---------- 4. put everything together, with last season's values ----------
# LAG(x) = value of x in the previous season of the same franchise.
# franchID stays the same when a team moves city. 2004 is missing, so 2005 looks at 2003.

query = f"""
WITH seasons AS (
    SELECT t.*, p.avg_age, p.last_season_pts, p.top3_pts_pg,
           p.n_20_goal_players, p.player_awards,
           g.goalie_save_pct,
           COALESCE(pw.playoff_wins, 0) AS playoff_wins,
           c.max_stint, c.first_coach, c.last_coach,
           CASE WHEN t.Pts IS NOT NULL THEN
               ROW_NUMBER() OVER (PARTITION BY t.year, t.confID
                                  ORDER BY t.rank = 1 DESC, t.Pts DESC, t.W DESC, t.GF - t.GA DESC)
           END AS conf_rank,
           COUNT(*) OVER (PARTITION BY t.year, t.confID) AS conf_teams
    FROM team_stats t
    LEFT JOIN player_stats p ON p.year = t.year AND p.tmID = t.tmID
    LEFT JOIN goalie_stats g ON g.year = t.year AND g.tmID = t.tmID
    LEFT JOIN playoff_wins pw ON pw.year = t.year AND pw.tmID = t.tmID
    LEFT JOIN coach_season c ON c.year = t.year AND c.tmID = t.tmID
),
with_prev AS (
    SELECT *,
        LAG(pts_pct)           OVER w AS prev_pts_pct,
        LAG(pts_pct, 2)        OVER w AS prev2_pts_pct,
        AVG(pts_pct)           OVER (w ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS avg3_pts_pct,
        LAG((conf_rank - 1.0) / (conf_teams - 1)) OVER w AS prev_rank_norm,
        LAG(gf_pg)             OVER w AS prev_gf_pg,
        LAG(ga_pg)             OVER w AS prev_ga_pg,
        LAG(pp_pct)            OVER w AS prev_pp_pct,
        LAG(pk_pct)            OVER w AS prev_pk_pct,
        LAG(pim_pg)            OVER w AS prev_pim_pg,
        LAG(made_playoffs)     OVER w AS prev_made_playoffs,
        LAG(playoff_wins)      OVER w AS prev_playoff_wins,
        LAG(max_stint) OVER w > 1 AS prev_coach_changed,
        LAG(last_coach)        OVER w AS prev_last_coach
    FROM seasons
    WINDOW w AS (PARTITION BY franchID ORDER BY year)
)
SELECT
    year, franchID, tmID, name,
    conf.Fullname AS conference,
    div.Fullname AS division,

    -- targets
    pts_pct,
    conf_rank,
    (conf_rank - 1.0) / (conf_teams - 1) AS rank_norm,  -- 0 = first, 1 = last
    max_stint > 1 AS coach_changed,

    -- team, last season
    prev_pts_pct, avg3_pts_pct,
    prev_pts_pct - prev2_pts_pct AS pts_pct_trend,
    prev_rank_norm, prev_gf_pg, prev_ga_pg, prev_pp_pct, prev_pk_pct, prev_pim_pg,
    prev_made_playoffs, prev_playoff_wins,

    -- players of this season, with their stats from the season before
    avg_age, last_season_pts, top3_pts_pg, n_20_goal_players,
    player_awards, goalie_save_pct,

    -- coach at the start of the season (career stats only from seasons before)
    first_coach AS coach,
    first_coach != prev_last_coach AS coach_new,
    prev_coach_changed,
    (SELECT COUNT(DISTINCT c.year) FROM Coaches c JOIN Teams t ON t.year = c.year AND t.tmID = c.tmID
     WHERE c.coachID = s.first_coach AND t.franchID = s.franchID AND c.year < s.year) AS coach_seasons_with_team,
    (SELECT COUNT(DISTINCT year) FROM Coaches WHERE coachID = s.first_coach AND year < s.year) AS coach_seasons,
    (SELECT SUM(w) * 1.0 / SUM(g) FROM Coaches WHERE coachID = s.first_coach AND year < s.year) AS coach_win_pct,
    (SELECT COALESCE(SUM(postw), 0) FROM Coaches WHERE coachID = s.first_coach AND year < s.year) AS coach_playoff_wins,
    (SELECT COUNT(*) FROM AwardsCoaches WHERE coachID = s.first_coach AND year < s.year) AS coach_awards

FROM with_prev s
LEFT JOIN abbrev conf ON conf.Type = 'Conference' AND conf.Code = s.confID
LEFT JOIN abbrev div ON div.Type = 'Division' AND div.Code = s.divID
WHERE year >= {FIRST_YEAR}
ORDER BY year, confID, conf_rank
"""

df = pd.read_sql(query, conn)

# NEXT_YEAR: the coach and the players of the new season are not in the data yet
next_year = df["year"] == NEXT_YEAR
not_known_yet = ["coach", "coach_new", "coach_seasons_with_team", "coach_seasons",
                 "coach_win_pct", "coach_playoff_wins", "coach_awards"]
df.loc[next_year, not_known_yet] = None


# ---------- 5. names that are easier to read ----------

df = df.rename(columns={
    "franchID": "franchise_id",
    "tmID": "team_id",
    "name": "team",
    # targets
    "pts_pct": "points_%",
    "conf_rank": "conference_rank",
    "rank_norm": "rank_0_to_1",
    "coach_changed": "coach_changed_in_season",
    # team, last season
    "prev_pts_pct": "last_season_points_%",
    "avg3_pts_pct": "last_3_seasons_points_%",
    "pts_pct_trend": "last_season_points_%_change",
    "prev_rank_norm": "last_season_rank_0_to_1",
    "prev_gf_pg": "last_season_goals_for_per_game",
    "prev_ga_pg": "last_season_goals_against_per_game",
    "prev_pp_pct": "last_season_power_play_%",
    "prev_pk_pct": "last_season_penalty_kill_%",
    "prev_pim_pg": "last_season_penalty_minutes_per_game",
    "prev_made_playoffs": "last_season_made_playoffs",
    "prev_playoff_wins": "last_season_playoff_wins",
    # players
    "avg_age": "players_avg_age",
    "last_season_pts": "players_last_season_points",
    "top3_pts_pg": "top3_players_points_per_game",
    "n_20_goal_players": "players_with_20_goals",
    "player_awards": "players_awards",
    "goalie_save_pct": "goalies_last_season_save_%",
    # coach
    "coach": "coach_id",
    "coach_new": "coach_new_this_season",
    "prev_coach_changed": "last_season_coach_changed",
    "coach_seasons": "coach_seasons_total",
    "coach_win_pct": "coach_win_%",
})

# yes/no columns and counts as whole numbers (1 instead of 1.0), the rest with 3 decimals
whole_numbers = ["conference_rank", "coach_changed_in_season", "last_season_made_playoffs",
                 "last_season_playoff_wins", "players_last_season_points", "players_with_20_goals",
                 "players_awards", "coach_new_this_season", "last_season_coach_changed",
                 "coach_seasons_with_team", "coach_seasons_total", "coach_playoff_wins", "coach_awards"]
df[whole_numbers] = df[whole_numbers].astype("Int64")
df = df.round(3)

df.to_sql("team_seasons", conn, if_exists="replace", index=False)
conn.close()

df.to_csv("outputs/team_seasons.csv", index=False)
print("Saved outputs/team_seasons.csv:", df.shape[0], "rows,", df.shape[1], "columns")
print("Seasons:", df["year"].min(), "to", df["year"].max())
print("\nMissing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])
