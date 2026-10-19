# Exploratory data analysis for tasks (a) and (b).
# Run from the project folder: python eda.py
# Prints the main numbers and saves the plots in outputs/.

import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_table

teams = load_table("Teams")
coaches = load_table("Coaches")


# ---------- Teams ----------

print("Seasons:", teams["year"].min(), "to", teams["year"].max())
print("Teams per league:")
print(teams["lgID"].value_counts())

# only the NHL from 1980 on (after the WHA merger)
nhl = teams[(teams["lgID"] == "NHL") & (teams["year"] >= 1980)].copy()

missing_years = set(range(1980, 2010)) - set(nhl["year"])
print("\nMissing seasons:", missing_years)  # 2004 lockout

# games per season are not always the same, so we use points %
nhl["pts_pct"] = nhl["Pts"] / (2 * nhl["G"])

# rank in the data is the division rank, task (a) is the conference rank.
# Official NHL standings: division winners go first, then the other teams by points.
# We checked this against the first round of the playoffs (1 vs 8, 2 vs 7, ...):
# it matches in 60 of 64 conference-seasons, the other 4 are teams tied on points and wins.
def conference_rank(df, div_winner_col, sort_cols):
    df = df.sort_values(["year", "confID", div_winner_col] + sort_cols, ascending=False)
    return df.groupby(["year", "confID"]).cumcount() + 1


nhl["div_winner"] = nhl["rank"] == 1
nhl["goal_diff"] = nhl["GF"] - nhl["GA"]
nhl["conf_rank"] = conference_rank(nhl, "div_winner", ["Pts", "W", "goal_diff"])

# previous season of the same team (franchID stays the same when a team moves city)
nhl = nhl.sort_values(["franchID", "year"])
nhl["prev_pts_pct"] = nhl.groupby("franchID")["pts_pct"].shift(1)
nhl["prev_playoff"] = nhl.groupby("franchID")["playoff"].shift(1)


# ---------- Task (a): ranking ----------

a = nhl[nhl["prev_pts_pct"].notna()].copy()
print("\nCorrelation points % last season vs this season:", round(a["pts_pct"].corr(a["prev_pts_pct"]), 2))

# baseline: same order as last season (best team of each division last season goes first)
a["prev_div_winner"] = a.groupby(["year", "divID"])["prev_pts_pct"].rank(ascending=False, method="first") == 1
a["baseline_rank"] = conference_rank(a, "prev_div_winner", ["prev_pts_pct"])
a["rank_error"] = (a["conf_rank"] - a["baseline_rank"]).abs()
print("Baseline mean rank error per season:")
print(a[a["year"] >= 2001].groupby("year")["rank_error"].mean().round(2))

plt.scatter(a["prev_pts_pct"], a["pts_pct"], alpha=0.3, s=10)
plt.xlabel("points % last season")
plt.ylabel("points % this season")
plt.title("Points % last season vs this season")
plt.savefig("outputs/eda_points_last_vs_this.png")
plt.close()


# ---------- Task (b): coach changed during the season ----------

nhl_coaches = coaches[coaches["lgID"] == "NHL"].sort_values(["year", "tmID", "stint"])

coach_season = nhl_coaches.groupby(["year", "tmID"]).agg(
    max_stint=("stint", "max"),
    first_coach=("coachID", "first"),
    last_coach=("coachID", "last"),
).reset_index()
coach_season["changed_in_season"] = coach_season["max_stint"] > 1

b = nhl.merge(coach_season, on=["year", "tmID"])
b = b.sort_values(["franchID", "year"])
b["prev_last_coach"] = b.groupby("franchID")["last_coach"].shift(1)
b = b[b["prev_pts_pct"].notna()].copy()

# new coach before the season starts (known on prediction day)
b["new_coach"] = b["first_coach"] != b["prev_last_coach"]
b["prev_made_playoffs"] = b["prev_playoff"].notna()

print("\nTeams that changed coach during the season:", round(b["changed_in_season"].mean() * 100, 1), "%")
print("Number of teams per season (2001-2009):")
print(b[b["year"] >= 2001].groupby("year")["changed_in_season"].sum())

print("\nChanged during season, by playoffs last season:")
print(b.groupby("prev_made_playoffs")["changed_in_season"].mean().round(2))
print("\nChanged during season, by new coach before the season:")
print(b.groupby("new_coach")["changed_in_season"].mean().round(2))

b["prev_group"] = pd.qcut(b["prev_pts_pct"], 5, labels=["worst", "2", "3", "4", "best"])
by_group = b.groupby("prev_group", observed=True)["changed_in_season"].mean()
print("\nChanged during season, by points % last season (5 groups):")
print(by_group.round(2))

by_group.plot.bar()
plt.xlabel("points % last season")
plt.ylabel("share of teams")
plt.title("Coach changed during the season")
plt.tight_layout()
plt.savefig("outputs/eda_coach_change_by_last_season.png")
plt.close()
