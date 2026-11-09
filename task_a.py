# Task (a): predict the final regular-season ranking of each conference.
# We predict the points % of each team and then sort the teams with the
# official rule (division winners first, then the others by points).
# Sliding window: to test season Y we train on the 9 seasons before it.
# Run prepare.py first, then: python task_a.py

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

WINDOW = 9
TEST_YEARS = [y for y in range(1995, 2010) if y != 2004]  # 2004 lockout
NEXT_YEAR = 2010

FEATURES = [
    # team, last season
    "prev_pts_pct", "avg3_pts_pct", "pts_pct_trend", "prev_rank_norm",
    "prev_gf_pg", "prev_ga_pg", "prev_pp_pct", "prev_pk_pct", "prev_pim_pg",
    "prev_made_playoffs", "prev_playoff_wins",
    # players
    "avg_age", "last_season_pts", "top3_pts_pg", "n_20_goal_players",
    "player_awards", "goalie_save_pct",
    # coach
    "coach_new", "prev_coach_changed", "coach_seasons_with_team", "coach_seasons",
    "coach_win_pct", "coach_playoff_wins", "coach_awards",
]

df = pd.read_csv("outputs/team_seasons.csv")
# the csv is sorted by the real rank, so we sort by team name to not
# leak the answer when two teams get the same prediction
df = df.sort_values(["year", "tmID"]).reset_index(drop=True)


# Imputation and scaling are inside the pipeline, so they are fitted
# on the training seasons only.
def make_models():
    return {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        "knn": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsRegressor(n_neighbors=15)),
        "tree": make_pipeline(SimpleImputer(strategy="median"), DecisionTreeRegressor(max_depth=4, random_state=0)),
        "forest": make_pipeline(SimpleImputer(strategy="median"),
                                RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0)),
        "boosting": make_pipeline(SimpleImputer(strategy="median"),
                                  GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                                            random_state=0)),
    }


# Turns predicted points % into a rank inside each conference (0 = first, 1 = last).
# The team with the best prediction in each division is a division winner and goes first.
# Ties (trees give the same value to many teams) are broken by last season's points %.
def predicted_rank(season, pred):
    s = season.copy()
    s["pred"] = pred
    s["tie_break"] = s["prev_pts_pct"].fillna(0)
    s = s.sort_values(["pred", "tie_break"], ascending=False)
    s["div_winner"] = ~s.duplicated("divID")
    s = s.sort_values(["confID", "div_winner", "pred", "tie_break"], ascending=[True, False, False, False])
    s["pred_conf_rank"] = s.groupby("confID").cumcount() + 1
    conf_teams = s.groupby("confID")["tmID"].transform("count")
    s["pred_rank_norm"] = (s["pred_conf_rank"] - 1) / (conf_teams - 1)
    return s


# Training rows for season `year`: the last WINDOW seasons that exist before it
# (2004 is missing, so for example 2010 uses 2000-2003 and 2005-2009).
def training_seasons(year):
    seasons = sorted(df.loc[df["year"] < year, "year"].unique())[-WINDOW:]
    return df[df["year"].isin(seasons)]


# ---------- sliding window over the test seasons ----------

rows = []
for year in TEST_YEARS:
    train = training_seasons(year)
    test = df[df["year"] == year]

    preds = {"baseline": test["prev_pts_pct"].fillna(train["pts_pct"].mean())}
    for name, model in make_models().items():
        model.fit(train[FEATURES], train["pts_pct"])
        preds[name] = model.predict(test[FEATURES])

    for name, pred in preds.items():
        s = predicted_rank(test, pred)
        rows.append({
            "year": year,
            "model": name,
            "rank_error": (s["pred_rank_norm"] - s["rank_norm"]).abs().mean(),
            "pts_pct_mae": (s["pred"] - s["pts_pct"]).abs().mean(),
            "spearman": s.groupby("confID")[["pred_rank_norm", "rank_norm"]]
                         .corr(method="spearman").xs("pred_rank_norm", level=1)["rank_norm"].mean(),
        })

results = pd.DataFrame(rows)
results.to_csv("outputs/task_a_results.csv", index=False)

print("Mean rank error per season (0 to 1 scale, lower is better):")
by_year = results.pivot(index="year", columns="model", values="rank_error")
print(by_year.round(3))

summary = results.groupby("model")[["rank_error", "pts_pct_mae", "spearman"]].mean()
summary["seasons_better_than_baseline"] = (by_year.lt(by_year["baseline"], axis=0)).sum()
summary = summary.sort_values("rank_error")
print("\nAverage over", len(TEST_YEARS), "test seasons:")
print(summary.round(3))

by_year.plot(marker="o")
plt.ylabel("mean rank error (0 to 1)")
plt.title("Task (a): rank error per test season")
plt.tight_layout()
plt.savefig("outputs/task_a_error_per_season.png")
plt.close()


# ---------- final prediction for the next season ----------
# Train on the last WINDOW seasons with the best model and predict NEXT_YEAR.
# The coach of NEXT_YEAR is not in the data, so those columns are filled
# with the training median for now.

best = summary.index[0] if summary.index[0] != "baseline" else summary.index[1]
train = training_seasons(NEXT_YEAR)
test = df[df["year"] == NEXT_YEAR]
model = make_models()[best]
model.fit(train[FEATURES], train["pts_pct"])
final = predicted_rank(test, model.predict(test[FEATURES]))
final = final[["confID", "pred_conf_rank", "pred_rank_norm", "name", "divID", "pred"]]
final = final.rename(columns={"pred": "pred_pts_pct"})
final.to_csv(f"outputs/task_a_prediction_{NEXT_YEAR}.csv", index=False)

print(f"\nPrediction for {NEXT_YEAR} with {best} (trained on {train['year'].min()}-{train['year'].max()}):")
print(final.round(3).to_string(index=False))
