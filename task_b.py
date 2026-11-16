# Task (b): predict which teams will change coach in the middle of the season.
# The target is coach_changed_in_season (the team had more than one coach that season).
# Only ~17% of the team-seasons have a change, so the classes are imbalanced and
# the model uses class weights (computed on the training seasons only).
# The model gives a probability of a change, and we predict a change for the k teams
# with the highest probability, where k = usual number of changes per season.
#
# Model: random forest. We tested logistic regression, kNN, a decision tree, a random
# forest and gradient boosting on 1995-2009 (training on the 9 seasons before each
# test season) and the random forest had the best F1, so we only study it.
#
# Training window: for every test season Y from 2004 to 2009 (2004 is missing,
# lockout) we train on the last 9, 7, 5, 3 and 1 seasons before Y and compare the
# F1. We want a window that is better than the baseline with a small variance,
# more than one with the best mean.
#
# Run prepare.py first, then:
#   python task_b.py        compares the training windows on 2005-2009
#   python task_b.py 2012   predicts one season, trained only on the seasons we
#                           have results for (2009 and before)

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

WINDOWS = [9, 7, 5, 3, 1]
# Window used for the predictions, chosen with python task_b.py: 3 and 7 seasons
# were better than random k teams in the most test seasons (4 of 5), and 3 has
# the smallest variance
WINDOW = 3
STUDY_YEARS = range(2004, 2010)
TARGET = "coach_changed_in_season"

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

df = pd.read_csv("outputs/team_seasons.csv")
df = df.sort_values(["year", "team_id"]).reset_index(drop=True)

# seasons with results (we can train on them and test them)
known = sorted(df.loc[df[TARGET].notna(), "year"].unique())
study_years = [y for y in STUDY_YEARS if y in known]


# Training rows for season `year`: the last `window` seasons with results before it.
# For 2012 that is the seasons up to 2009, because we don't have 2010 and 2011.
def training_seasons(year, window):
    before = [y for y in known if y < year][-window:]
    train = df[df["year"].isin(before)]
    return train[train[TARGET].notna()]


def make_model():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         RandomForestClassifier(n_estimators=200, min_samples_leaf=10,
                                                class_weight="balanced", random_state=0, n_jobs=-1))


# Probability of a coach change for every team of `year`, and the prediction:
# a change for the k teams with the highest probability, where k is the share of
# teams that changed coach in the training seasons times the number of teams.
def predict(year, window):
    train = training_seasons(year, window)
    test = df[df["year"] == year]
    y = train[TARGET].astype(int)
    k = round(y.mean() * len(test))
    if y.nunique() < 2:  # no coach changes at all in the training seasons
        prob = np.zeros(len(test))
    else:
        prob = make_model().fit(train[FEATURES], y).predict_proba(test[FEATURES])[:, 1]
    pred = np.zeros(len(test), dtype=int)
    pred[np.argsort(-prob, kind="stable")[:k]] = 1
    return test, prob, pred, k


# F1 of the prediction and of the two baselines:
#   - no changes (always F1 = 0)
#   - k random teams: on average k * changes / teams of them are right
def scores(test, prob, pred, k):
    y = test[TARGET].astype(int).values
    right = ((pred == 1) & (y == 1)).sum()
    changes = y.sum()
    random_right = k * changes / len(y)
    return {
        "real_changes": changes,
        "predicted_changes": pred.sum(),
        "right_changes": right,
        "forest_f1": 2 * right / (pred.sum() + changes) if pred.sum() + changes > 0 else 0,
        "random_k_f1": 2 * random_right / (k + changes) if k + changes > 0 else 0,
        "no_changes_f1": 0.0,
        "forest_auc": roc_auc_score(y, prob) if 0 < changes < len(y) else np.nan,
    }


# ---------- one season: python task_b.py 2012 ----------

if len(sys.argv) > 1:
    year = int(sys.argv[1])
    if year not in df["year"].values or len([y for y in known if y < year]) < WINDOW:
        sys.exit(f"Season {year} can't be predicted: it must be in the data (up to {df['year'].max()}) "
                 f"and have {WINDOW} seasons with results before it.")
    test, prob, pred, k = predict(year, WINDOW)
    has_results = test[TARGET].notna().all()

    final = pd.DataFrame({
        "team": test["team"],
        "coach_at_start": test["coach_id"],
        "change_probability": prob.round(3),
        "predicted_change": pred,
    })
    if has_results:
        final["real_change"] = test[TARGET].astype(int)
    final = final.sort_values("change_probability", ascending=False)
    final.to_csv(f"outputs/task_b_prediction_{year}.csv", index=False)

    train_years = training_seasons(year, WINDOW)["year"]
    print(f"Season {year}: random forest trained on the last {WINDOW} seasons with results "
          f"({train_years.min()}-{train_years.max()}), change predicted for the top {k} teams\n")
    print(final.to_string(index=False))
    print(f"\nSaved outputs/task_b_prediction_{year}.csv")

    if has_results:
        s = scores(test, prob, pred, k)
        print(f"\nRight {s['right_changes']} of {s['real_changes']} changes, F1 {s['forest_f1']:.3f} "
              f"(random k teams {s['random_k_f1']:.3f}), AUC {s['forest_auc']:.3f}")
    sys.exit()


# ---------- compare the training windows on 2005-2009 ----------

rows = []
for year in study_years:
    for window in WINDOWS:
        rows.append({"test_season": year, "training_seasons": window, **scores(*predict(year, window))})
results = pd.DataFrame(rows)
results.round(3).to_csv("outputs/task_b_windows.csv", index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
summary = results.groupby("training_seasons").agg(
    mean=("forest_f1", "mean"),
    variance=("forest_f1", "var"),
    std=("forest_f1", "std"),
    worst_season=("forest_f1", "min"),
    mean_auc=("forest_auc", "mean"),
    right_changes=("right_changes", "sum"),
    real_changes=("real_changes", "sum"),
)
summary["seasons_better_than_random"] = (results["forest_f1"] > results["random_k_f1"]) \
    .groupby(results["training_seasons"]).sum()
random_mean = results.groupby("training_seasons")["random_k_f1"].mean()

print(f"F1 of the random forest per training window, test seasons {study_years[0]}-{study_years[-1]} "
      f"({len(study_years)} seasons, 2004 is missing). Higher is better.\n")
print(summary.round(4))
print("\nBaselines: no changes F1 = 0; random k teams F1 (mean):", random_mean.round(4).to_dict())

# Choice (teacher): we want a window that is better than the baseline in every
# season (or in as many as possible) with a small variance, even if its mean
# is not the best. So: the windows better than random k teams in the most seasons,
# and among them the one with the smallest variance.
most = summary["seasons_better_than_random"].max()
candidates = summary[summary["seasons_better_than_random"] == most]
best = candidates["variance"].idxmin()
print(f"\nWindows better than random k teams in {most} of {len(study_years)} seasons: {list(candidates.index)}")
print(f"Best window (smallest variance among them): {best} seasons")
print(f"WINDOW used for the predictions: {WINDOW}")

by_year = results.pivot(index="test_season", columns="training_seasons", values="forest_f1")
by_year.columns = [f"Random forest, {w} seasons" for w in by_year.columns]
ax = by_year.plot(figsize=(9, 5), marker="o")
results[results["training_seasons"] == WINDOW].set_index("test_season")["random_k_f1"].plot(
    ax=ax, color="black", linestyle="--", label=f"Random k teams ({WINDOW} seasons)")
ax.legend()
plt.ylabel("F1")
plt.title("Task (b): F1 per test season and training window")
plt.tight_layout()
plt.savefig("outputs/task_b_windows.png")
plt.close()
print("\nSaved outputs/task_b_windows.csv and outputs/task_b_windows.png")
