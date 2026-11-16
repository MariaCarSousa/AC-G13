# Builds one row per team and season (outputs/team_seasons.csv), for every
# league and season in the data (1909 on) plus the next seasons we have to predict.
# Each row has the targets of that season and features we already know
# the day before the season starts: the team and its history, the coach and
# their history, and the players of the team and their history.
# Run make_db.py first, then: python prepare.py

import sqlite3

import pandas as pd

DB_PATH = "outputs/hockey.db"

conn = sqlite3.connect(DB_PATH)
# The teacher will test the next seasons (2010 to 2013). Any season added to Teams
# is used as it is. To be able to run the code before that, we also add the next
# 4 seasons after the last one in Teams, copying the NHL teams of that season with
# empty stats. Their features come from the seasons we already have.
LAST_YEAR = conn.execute("SELECT MAX(year) FROM Teams").fetchone()[0]
NEXT_YEARS = range(LAST_YEAR + 1, LAST_YEAR + 5)


# ---------- 1. team stats of each season ----------

copies = "\nUNION ALL\n".join(f"""
SELECT {y}, lgID, tmID, franchID, confID, divID, name, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
       NULL, NULL, NULL, NULL, NULL, NULL, NULL
FROM Teams
WHERE lgID = 'NHL' AND year = {LAST_YEAR}""" for y in NEXT_YEARS)

conn.executescript(f"""
DROP TABLE IF EXISTS team_stats;
CREATE TEMP TABLE team_stats AS
SELECT year, lgID, tmID, franchID, confID, divID, name, G, W, Pts, GF, GA, rank, playoff,
       GF - GA AS goal_diff,
       Pts * 1.0 / (2 * G) AS pts_pct,
       GF * 1.0 / G AS gf_pg,
       GA * 1.0 / G AS ga_pg,
       PPG * 1.0 / PPC AS pp_pct,
       1 - PKG * 1.0 / PKC AS pk_pct,
       PIM * 1.0 / G AS pim_pg,
       playoff IS NOT NULL AS made_playoffs
FROM Teams
UNION ALL
{copies};

-- seasons we have results for, and for every season the last one of those before it
-- (2004 is missing, so for 2005 it is 2003; for 2012 it is 2009 if we have no 2010 and 2011 results)
DROP TABLE IF EXISTS known_seasons;
CREATE TEMP TABLE known_seasons AS
SELECT DISTINCT year FROM team_stats WHERE Pts IS NOT NULL;

DROP TABLE IF EXISTS prev_season;
CREATE TEMP TABLE prev_season AS
SELECT y.year, (SELECT MAX(k.year) FROM known_seasons k WHERE k.year < y.year) AS prev_year
FROM (SELECT DISTINCT year FROM team_stats) y;
""")


# ---------- 2. players of each season and their history ----------
# Players of the team this season (known before the season starts),
# with their own stats from the season before, on any team.
# The data lists every player who played for the team during the season,
# including call-ups and trades. Bad teams use more players, so counting all
# of them would leak how the season went. We only use the 18 best players
# by last season's points (about one lineup).
# "Season before" = the last season we have results for (prev_season table).

conn.executescript("""
DROP TABLE IF EXISTS player_year;
CREATE TEMP TABLE player_year AS
SELECT playerID, year, SUM(GP) AS GP, SUM(G) AS G, SUM(Pts) AS Pts
FROM Scoring
GROUP BY playerID, year;

DROP TABLE IF EXISTS roster;
CREATE TEMP TABLE roster AS
SELECT DISTINCT s.playerID, s.year, s.tmID, p.prev_year
FROM Scoring s
JOIN prev_season p ON p.year = s.year;

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
    SELECT playerID, year, SUM(GA) AS GA, SUM(SA) AS SA, SUM(Min) AS Min
    FROM Goalies
    GROUP BY playerID, year
),
goalie_roster AS (
    SELECT DISTINCT g.playerID, g.year, g.tmID, p.prev_year
    FROM Goalies g
    JOIN prev_season p ON p.year = g.year
)
SELECT r.year, r.tmID,
       1 - SUM(CASE WHEN last.SA > 0 THEN last.GA END) * 1.0
         / SUM(CASE WHEN last.SA > 0 THEN last.SA END) AS goalie_save_pct,
       SUM(CASE WHEN last.Min > 0 THEN last.GA END) * 60.0
         / SUM(CASE WHEN last.Min > 0 THEN last.Min END) AS goalie_gaa
FROM goalie_roster r
JOIN goalie_year last ON last.playerID = r.playerID AND last.year = r.prev_year
GROUP BY r.year, r.tmID;
""")

conn.executescript("""
DROP TABLE IF EXISTS playoff_wins;
CREATE TEMP TABLE playoff_wins AS
SELECT year, tmID, W AS playoff_wins
FROM TeamsPost;
""")


# ---------- 3. coaches of each season ----------
# first_coach = coach at the start of the season, last_coach = coach at the end.
# stint > 1 means the coach was changed during the season (target of task b).

conn.executescript("""
DROP TABLE IF EXISTS coach_season;
CREATE TEMP TABLE coach_season AS
WITH c AS (SELECT * FROM Coaches),
     stints AS (SELECT year, tmID, MAX(stint) AS max_stint FROM c GROUP BY year, tmID)
SELECT st.year, st.tmID, st.max_stint,
       (SELECT MIN(coachID) FROM c WHERE c.year = st.year AND c.tmID = st.tmID AND c.stint = 1) AS first_coach,
       (SELECT MIN(coachID) FROM c WHERE c.year = st.year AND c.tmID = st.tmID AND c.stint = st.max_stint) AS last_coach
FROM stints st;
""")


# ---------- 4. put everything together, with last season's values ----------
# Teams are ranked with the information we have about them (teacher feedback):
#   - division known: one ranking per division (inside its conference)
#   - only the conference known: one ranking per conference
#   - nothing known: one ranking of all the teams of the league
# Each league (NHL, WHA, ...) is ranked on its own. Order: points, wins, goal difference.
#
# "Last season" = the last season of the same franchise that we have results for
# (franchID stays the same when a team moves city). 2004 is missing, so 2005 looks
# at 2003, and a season like 2012 looks at 2009 if we don't have 2010 and 2011.

query = f"""
WITH seasons AS (
    SELECT t.*, p.avg_age, p.last_season_pts, p.top3_pts_pg,
           p.n_20_goal_players, p.player_awards,
           g.goalie_save_pct, g.goalie_gaa,
           COALESCE(pw.playoff_wins, 0) AS playoff_wins,
           c.max_stint, c.first_coach, c.last_coach,
           CASE WHEN t.divID IS NOT NULL THEN COALESCE(t.confID, '') || '/' || t.divID
                ELSE COALESCE(t.confID, '') END AS group_key
    FROM team_stats t
    LEFT JOIN player_stats p ON p.year = t.year AND p.tmID = t.tmID
    LEFT JOIN goalie_stats g ON g.year = t.year AND g.tmID = t.tmID
    LEFT JOIN playoff_wins pw ON pw.year = t.year AND pw.tmID = t.tmID
    LEFT JOIN coach_season c ON c.year = t.year AND c.tmID = t.tmID
),
ranked AS (
    SELECT *,
           CASE WHEN Pts IS NOT NULL THEN
               ROW_NUMBER() OVER (PARTITION BY year, lgID, group_key ORDER BY Pts DESC, W DESC, goal_diff DESC)
           END AS group_rank,
           COUNT(*) OVER (PARTITION BY year, lgID, group_key) AS group_teams
    FROM seasons
),
-- only the seasons with results, with a few values of the season before them
known AS (
    SELECT *,
           (group_rank - 1.0) / (group_teams - 1) AS rank_norm,
           LAG(pts_pct) OVER w AS lag_pts_pct,
           AVG(pts_pct) OVER (w ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS last3_pts_pct
    FROM ranked
    WHERE Pts IS NOT NULL
    WINDOW w AS (PARTITION BY franchID ORDER BY year)
),
with_prev AS (
    SELECT s.*,
           k.pts_pct AS prev_pts_pct,
           k.last3_pts_pct AS avg3_pts_pct,
           k.pts_pct - k.lag_pts_pct AS pts_pct_trend,
           k.rank_norm AS prev_rank_norm,
           k.gf_pg AS prev_gf_pg,
           k.ga_pg AS prev_ga_pg,
           k.pp_pct AS prev_pp_pct,
           k.pk_pct AS prev_pk_pct,
           k.pim_pg AS prev_pim_pg,
           k.made_playoffs AS prev_made_playoffs,
           k.playoff_wins AS prev_playoff_wins,
           k.max_stint > 1 AS prev_coach_changed,
           k.last_coach AS prev_last_coach
    FROM ranked s
    LEFT JOIN known k ON k.franchID = s.franchID
        AND k.year = (SELECT MAX(k2.year) FROM known k2 WHERE k2.franchID = s.franchID AND k2.year < s.year)
)
SELECT
    year, lgID, franchID, tmID, name,
    -- full name from abbrev, or the value as it is if it is not there
    COALESCE(conf.Fullname, s.confID) AS conference,
    COALESCE(div.Fullname, s.divID) AS division,
    CASE WHEN s.divID IS NOT NULL AND s.confID IS NOT NULL
             THEN COALESCE(conf.Fullname, s.confID) || ' - ' || COALESCE(div.Fullname, s.divID)
         WHEN s.divID IS NOT NULL THEN COALESCE(div.Fullname, s.divID)
         WHEN s.confID IS NOT NULL THEN COALESCE(conf.Fullname, s.confID)
         ELSE lgID || ' (all teams)'
    END AS ranking_group,

    -- targets
    pts_pct, W, goal_diff,
    group_rank,
    (group_rank - 1.0) / (group_teams - 1) AS rank_norm,  -- 0 = first, 1 = last
    max_stint > 1 AS coach_changed,

    -- team, last season
    prev_pts_pct, avg3_pts_pct, pts_pct_trend,
    prev_rank_norm, prev_gf_pg, prev_ga_pg, prev_pp_pct, prev_pk_pct, prev_pim_pg,
    prev_made_playoffs, prev_playoff_wins,

    -- players of this season, with their stats from the season before
    avg_age, last_season_pts, top3_pts_pg, n_20_goal_players,
    player_awards, goalie_save_pct, goalie_gaa,

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
ORDER BY year, lgID, ranking_group, group_rank
"""

df = pd.read_sql(query, conn)

# If the coach of a season is not in the data (always for NEXT_YEARS, and a few
# old seasons), the coach columns would say 0 seasons, 0 awards, ... so we
# leave them empty instead. When the teacher adds the coaches, they are used.
no_coach = df["coach"].isna()
not_known_yet = ["coach_new", "coach_seasons_with_team", "coach_seasons",
                 "coach_win_pct", "coach_playoff_wins", "coach_awards"]
df.loc[no_coach, not_known_yet] = None


# ---------- 5. names that are easier to read ----------

df = df.rename(columns={
    "lgID": "league",
    "franchID": "franchise_id",
    "tmID": "team_id",
    "name": "team",
    # targets
    "pts_pct": "points_%",
    "W": "wins",
    "goal_diff": "goal_difference",
    "group_rank": "rank_in_group",
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
    "goalie_gaa": "goalies_last_season_goals_against_per_60",
    # coach
    "coach": "coach_id",
    "coach_new": "coach_new_this_season",
    "prev_coach_changed": "last_season_coach_changed",
    "coach_seasons": "coach_seasons_total",
    "coach_win_pct": "coach_win_%",
})

# yes/no columns and counts as whole numbers (1 instead of 1.0), the rest with 3 decimals
whole_numbers = ["wins", "goal_difference", "rank_in_group", "coach_changed_in_season", "last_season_made_playoffs",
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
