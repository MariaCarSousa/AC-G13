# Task (a): predict the final regular-season ranking of the teams.
# We predict the points % of each team and then sort the teams inside their
# ranking group (conference, or division / whole league in older seasons)
# with the official rule: division winners first, then the others by points.
# Sliding window: to test season Y we train on the 9 seasons before it.
#
# Run prepare.py first, then:
#   python task_a.py 1975   predicts the ranking of one season (any season with
#                           9 seasons before it, also one added by the teacher)
#   python task_a.py        tests all the models on every season with results

import sys

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
# Model used for the predictions. Ridge was the best on the recent seasons
# (1995-2009) when we tested all the models (python task_a.py).
FINAL_MODEL = "ridge"
RECENT = range(1995, 2010)

FEATURES = [
    # team, last season
    "last_season_points_%", "last_3_seasons_points_%", "last_season_points_%_change",
    "last_season_rank_0_to_1", "last_season_goals_for_per_game", "last_season_goals_against_per_game",
    "last_season_power_play_%", "last_season_penalty_kill_%", "last_season_penalty_minutes_per_game",
    "last_season_made_playoffs", "last_season_playoff_wins",
    # players
    "players_avg_age", "players_last_season_points", "top3_players_points_per_game",
    "players_with_20_goals", "players_awards", "goalies_last_season_save_%",
    "goalies_last_season_goals_against_per_60",
    # coach
    "coach_new_this_season", "last_season_coach_changed", "coach_seasons_with_team",
    "coach_seasons_total", "coach_win_%", "coach_playoff_wins", "coach_awards",
]

MODEL_NAMES = {
    "baseline": "Baseline (last season)", "ridge": "Ridge", "knn": "kNN",
    "tree": "Decision tree", "forest": "Random forest", "boosting": "Gradient boosting",
}

df = pd.read_csv("outputs/team_seasons.csv")
# the csv is sorted by the real rank, so we sort by team id to not
# leak the answer when two teams get the same prediction
df = df.sort_values(["year", "team_id"]).reset_index(drop=True)

# we can predict every season that has WINDOW seasons before it (2004 is missing,
# lockout), and test the ones that also have results
seasons = sorted(df["year"].unique())
PREDICTABLE = seasons[WINDOW:]
with_results = set(df.loc[df["rank_in_group"].notna(), "year"])
TEST_YEARS = [y for y in PREDICTABLE if y in with_results]


# Imputation and scaling are inside the pipeline, so they are fitted
# on the training seasons only. Some columns are empty in the old seasons
# (no power play or shots stats), so the imputer fills them with a constant
# and the models just ignore them.
def imputer():
    return SimpleImputer(strategy="median", keep_empty_features=True)


def make_models():
    return {
        "ridge": make_pipeline(imputer(), StandardScaler(), Ridge(alpha=1.0)),
        "knn": make_pipeline(imputer(), StandardScaler(), KNeighborsRegressor(n_neighbors=15)),
        "tree": make_pipeline(imputer(), DecisionTreeRegressor(max_depth=4, random_state=0)),
        "forest": make_pipeline(imputer(),
                                RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0,
                                                      n_jobs=-1)),
        "boosting": make_pipeline(imputer(),
                                  GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                                            random_state=0)),
    }


# Turns predicted points % into a rank inside each ranking group (0 = first, 1 = last).
# The team with the best prediction in each division is a division winner and goes first.
# Ties (trees give the same value to many teams) are broken by last season's points %.
def predicted_rank(season, pred):
    s = season.copy()
    s["pred"] = pred
    s["tie_break"] = s["last_season_points_%"].fillna(0)
    s = s.sort_values(["pred", "tie_break"], ascending=False)
    s["div_winner"] = ~s.duplicated(["ranking_group", "division"])
    s = s.sort_values(["ranking_group", "div_winner", "pred", "tie_break"], ascending=[True, False, False, False])
    s["pred_rank"] = s.groupby("ranking_group").cumcount() + 1
    group_teams = s.groupby("ranking_group")["team_id"].transform("count")
    s["pred_rank_norm"] = (s["pred_rank"] - 1) / (group_teams - 1)
    return s


# Training rows for season `year`: the last WINDOW seasons that exist before it
# (2004 is missing, so for example 2010 uses 2000-2003 and 2005-2009).
def training_seasons(year):
    before = [y for y in seasons if y < year][-WINDOW:]
    return df[df["year"].isin(before)]


# Trains every model on the seasons before `year` and ranks the teams of `year`.
# Returns one ranked table per model.
def predict_season(year):
    train = training_seasons(year)
    test = df[df["year"] == year]
    preds = {"baseline": test["last_season_points_%"].fillna(train["points_%"].mean())}
    for name, model in make_models().items():
        model.fit(train[FEATURES], train["points_%"])
        preds[name] = model.predict(test[FEATURES])
    return {name: predicted_rank(test, pred) for name, pred in preds.items()}


def scores(s):
    return {
        "rank_error": (s["pred_rank_norm"] - s["rank_0_to_1"]).abs().mean(),
        "pts_pct_mae": (s["pred"] - s["points_%"]).abs().mean(),
        "spearman": s.groupby("ranking_group")[["pred_rank_norm", "rank_0_to_1"]]
                     .corr(method="spearman").xs("pred_rank_norm", level=1)["rank_0_to_1"].mean(),
    }


# ---------- one season: python task_a.py 1975 ----------
# Prediction of the final model, saved to outputs/task_a_prediction_<year>.csv.
# If the season already has results, we also show the real rank and the error.
# Coaches or players that are not in the data are filled with the training median.

if len(sys.argv) > 1:
    year = int(sys.argv[1])
    if year not in PREDICTABLE:
        sys.exit(f"Season {year} can't be predicted: it needs {WINDOW} seasons before it "
                 f"(possible: {PREDICTABLE[0]}-{PREDICTABLE[-1]}, no 2004).")
    ranked = predict_season(year)
    has_results = year in with_results

    final = ranked[FINAL_MODEL]
    final = pd.DataFrame({
        "ranking_group": final["ranking_group"],
        "predicted_rank": final["pred_rank"],
        "real_rank": final["rank_in_group"].astype("Int64"),
        "team": final["team"],
        "division": final["division"],
        "predicted_rank_0_to_1": final["pred_rank_norm"].round(3),
        "predicted_points_%": final["pred"].round(3),
    })
    if not has_results:
        final = final.drop(columns="real_rank")
    final.to_csv(f"outputs/task_a_prediction_{year}.csv", index=False)

    train_years = training_seasons(year)["year"]
    print(f"Season {year}: prediction with {MODEL_NAMES[FINAL_MODEL]} "
          f"(trained on {train_years.min()}-{train_years.max()})")
    for group, rows in final.groupby("ranking_group"):
        print(f"\n{group}")
        print(rows.drop(columns="ranking_group").to_string(index=False))
    print(f"\nSaved outputs/task_a_prediction_{year}.csv")

    if has_results:
        print("\nRank error of every model in this season (0 to 1, lower is better):")
        for name, s in ranked.items():
            print(f"  {MODEL_NAMES[name]:<24} {scores(s)['rank_error']:.3f}")
    sys.exit()


# ---------- sliding window over all the test seasons ----------

rows = []
for year in TEST_YEARS:
    for name, s in predict_season(year).items():
        rows.append({"year": year, "model": name, **scores(s)})
results = pd.DataFrame(rows)
by_year = results.pivot(index="year", columns="model", values="rank_error")

# csv files with names that are easier to read
readable = results.rename(columns={
    "year": "test_season",
    "rank_error": "rank_error_0_to_1",
    "pts_pct_mae": "points_%_mae",
})
readable["model"] = readable["model"].map(MODEL_NAMES)
readable.round(3).to_csv("outputs/task_a_results.csv", index=False)
by_year.rename(columns=MODEL_NAMES).round(3).to_csv("outputs/task_a_rank_error_per_season.csv",
                                                    index_label="test_season")


def summary_of(years):
    part = results[results["year"].isin(years)]
    part_by_year = by_year.loc[by_year.index.isin(years)]
    summary = part.groupby("model")[["rank_error", "pts_pct_mae", "spearman"]].mean()
    summary["seasons_better_than_baseline"] = part_by_year.lt(part_by_year["baseline"], axis=0).sum()
    return summary.sort_values("rank_error")


# the league changed a lot, so we also look at each era on its own
eras = {
    f"all seasons ({TEST_YEARS[0]}-{TEST_YEARS[-1]})": TEST_YEARS,
    f"{TEST_YEARS[0]}-1966 (before the 1967 expansion)": range(TEST_YEARS[0], 1967),
    "1967-1994": range(1967, 1995),
    f"{RECENT[0]}-{RECENT[-1]} (recent)": RECENT,
}
print("Mean rank error (0 to 1 scale, lower is better)")
for era, years in eras.items():
    n = sum(y in years for y in TEST_YEARS)
    print(f"\n{era}, {n} test seasons:")
    print(summary_of(years).round(3))

by_year.rename(columns=MODEL_NAMES).plot(figsize=(10, 5))
plt.ylabel("mean rank error (0 to 1)")
plt.title("Task (a): rank error per test season")
plt.tight_layout()
plt.savefig("outputs/task_a_error_per_season.png")
plt.close()


# is FINAL_MODEL still the best one on the recent seasons?
best = [m for m in summary_of(RECENT).index if m != "baseline"][0]
print(f"\nBest model on {RECENT[0]}-{RECENT[-1]}: {MODEL_NAMES[best]} (FINAL_MODEL is {MODEL_NAMES[FINAL_MODEL]})")
print("To predict a season: python task_a.py <year>")
