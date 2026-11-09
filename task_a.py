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
    "last_season_points_%", "last_3_seasons_points_%", "last_season_points_%_change",
    "last_season_rank_0_to_1", "last_season_goals_for_per_game", "last_season_goals_against_per_game",
    "last_season_power_play_%", "last_season_penalty_kill_%", "last_season_penalty_minutes_per_game",
    "last_season_made_playoffs", "last_season_playoff_wins",
    # players
    "players_avg_age", "players_last_season_points", "top3_players_points_per_game",
    "players_with_20_goals", "players_awards", "goalies_last_season_save_%",
    # coach
    "coach_new_this_season", "last_season_coach_changed", "coach_seasons_with_team",
    "coach_seasons_total", "coach_win_%", "coach_playoff_wins", "coach_awards",
]

df = pd.read_csv("outputs/team_seasons.csv")
# the csv is sorted by the real rank, so we sort by team id to not
# leak the answer when two teams get the same prediction
df = df.sort_values(["year", "team_id"]).reset_index(drop=True)


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
    s["tie_break"] = s["last_season_points_%"].fillna(0)
    s = s.sort_values(["pred", "tie_break"], ascending=False)
    s["div_winner"] = ~s.duplicated("division")
    s = s.sort_values(["conference", "div_winner", "pred", "tie_break"], ascending=[True, False, False, False])
    s["pred_conf_rank"] = s.groupby("conference").cumcount() + 1
    conf_teams = s.groupby("conference")["team_id"].transform("count")
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

    preds = {"baseline": test["last_season_points_%"].fillna(train["points_%"].mean())}
    for name, model in make_models().items():
        model.fit(train[FEATURES], train["points_%"])
        preds[name] = model.predict(test[FEATURES])

    for name, pred in preds.items():
        s = predicted_rank(test, pred)
        rows.append({
            "year": year,
            "model": name,
            "rank_error": (s["pred_rank_norm"] - s["rank_0_to_1"]).abs().mean(),
            "pts_pct_mae": (s["pred"] - s["points_%"]).abs().mean(),
            "spearman": s.groupby("conference")[["pred_rank_norm", "rank_0_to_1"]]
                         .corr(method="spearman").xs("pred_rank_norm", level=1)["rank_0_to_1"].mean(),
        })

results = pd.DataFrame(rows)

print("Mean rank error per season (0 to 1 scale, lower is better):")
by_year = results.pivot(index="year", columns="model", values="rank_error")
print(by_year.round(3))

# csv files with names that are easier to read
MODEL_NAMES = {
    "baseline": "Baseline (last season)", "ridge": "Ridge", "knn": "kNN",
    "tree": "Decision tree", "forest": "Random forest", "boosting": "Gradient boosting",
}
readable = results.rename(columns={
    "year": "test_season",
    "rank_error": "rank_error_0_to_1",
    "pts_pct_mae": "points_%_error",
    "spearman": "rank_correlation",
})
readable["model"] = readable["model"].map(MODEL_NAMES)
readable.round(3).to_csv("outputs/task_a_results.csv", index=False)
by_year.rename(columns=MODEL_NAMES).round(3).to_csv("outputs/task_a_rank_error_per_season.csv",
                                                    index_label="test_season")

summary = results.groupby("model")[["rank_error", "pts_pct_mae", "spearman"]].mean()
summary["seasons_better_than_baseline"] = (by_year.lt(by_year["baseline"], axis=0)).sum()
summary = summary.sort_values("rank_error")
print("\nAverage over", len(TEST_YEARS), "test seasons:")
print(summary.round(3))

by_year.rename(columns=MODEL_NAMES).plot(marker="o")
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
model.fit(train[FEATURES], train["points_%"])
final = predicted_rank(test, model.predict(test[FEATURES]))

final = pd.DataFrame({
    "conference": final["conference"],
    "predicted_rank": final["pred_conf_rank"],
    "team": final["team"],
    "division": final["division"],
    "predicted_rank_0_to_1": final["pred_rank_norm"].round(3),
    "predicted_points_%": final["pred"].round(3),
})
final.to_csv(f"outputs/task_a_prediction_{NEXT_YEAR}.csv", index=False)

print(f"\nPrediction for {NEXT_YEAR} with {MODEL_NAMES[best]} "
      f"(trained on {train['year'].min()}-{train['year'].max()}):")
print(final.to_string(index=False))
