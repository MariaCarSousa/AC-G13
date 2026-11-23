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
# Training window and min_samples_leaf (the smallest number of team-seasons in a
# leaf of each tree; small = deep trees that can overfit, big = simple trees): for
# every test season Y from 2004 to 2009 (2004 is missing, lockout) we train on the
# last 9, 7, 5, 3 and 1 seasons before Y, with several values of min_samples_leaf,
# and compare the F1. The teacher tests on 2010-2013, so 2005-2009 works as our
# validation seasons and we tune both on them. We want a combination that is
# better than the baseline in as many seasons as possible with a small variance,
# more than one with the best mean.
#
# Run prepare.py first, then:
#   python task_b.py        compares the training windows and min_samples_leaf on 2005-2009
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
LEAF_SIZES = [1, 5, 10, 20, 40]
# Window and min_samples_leaf used for the predictions, chosen with python task_b.py
# (with the features below): 1 with 5 seasons was better than random k teams in all
# 5 test seasons and had the smallest variance among those. min_samples_leaf 1 means
# deep trees, which alone would overfit, but the forest averages 200 of them.
WINDOW = 5
LEAF_SIZE = 1
STUDY_YEARS = range(2004, 2010)
TARGET = "coach_changed_in_season"

# Features chosen by feature_selection.py (filter method, on the seasons 1995-2003):
# features related to the target (|Spearman| >= 0.05) and, of each very correlated pair
# (|Spearman| > 0.8), the one more related to the target. On 2005-2009 this set was
# better than using all the features (see outputs/feature_selection/summary.csv).
FEATURES = [
    "last_season_goals_against_per_game",
    "coach_seasons_with_team",
    "coach_new_this_season",
    "last_season_made_playoffs",
    "last_season_penalty_kill_%",
    "last_season_rank_0_to_1",
    "coach_win_%",
    "players_awards",
    "last_season_penalty_minutes_per_game",
    "last_season_coach_changed",
    "last_season_league_penalty_minutes_per_game",
    "last_season_points_%_change",
    "coach_awards",
    "coach_seasons_total",
    "last_3_seasons_points_%",
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


def make_model(leaf_size):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         RandomForestClassifier(n_estimators=200, min_samples_leaf=leaf_size,
                                                class_weight="balanced", random_state=0, n_jobs=-1))


# Probability of a coach change for every team of `year`, and the prediction:
# a change for the k teams with the highest probability, where k is the share of
# teams that changed coach in the training seasons times the number of teams.
def predict(year, window, leaf_size=LEAF_SIZE):
    train = training_seasons(year, window)
    test = df[df["year"] == year]
    y = train[TARGET].astype(int)
    k = round(y.mean() * len(test))
    if y.nunique() < 2:  # no coach changes at all in the training seasons
        prob = np.zeros(len(test))
    else:
        prob = make_model(leaf_size).fit(train[FEATURES], y).predict_proba(test[FEATURES])[:, 1]
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


# The code below only runs with python task_b.py (other scripts can import the
# functions above without running it).
if __name__ == "__main__":
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
        print(f"Season {year}: random forest (min_samples_leaf {LEAF_SIZE}) trained on the last {WINDOW} seasons with results "
              f"({train_years.min()}-{train_years.max()}), change predicted for the top {k} teams\n")
        print(final.to_string(index=False))
        print(f"\nSaved outputs/task_b_prediction_{year}.csv")

        if has_results:
            s = scores(test, prob, pred, k)
            print(f"\nRight {s['right_changes']} of {s['real_changes']} changes, F1 {s['forest_f1']:.3f} "
                  f"(random k teams {s['random_k_f1']:.3f}), AUC {s['forest_auc']:.3f}")
        sys.exit()


    # ---------- compare the training windows and min_samples_leaf on 2005-2009 ----------

    rows = []
    for year in study_years:
        for leaf_size in LEAF_SIZES:
            for window in WINDOWS:
                rows.append({"test_season": year, "min_samples_leaf": leaf_size, "training_seasons": window,
                             **scores(*predict(year, window, leaf_size))})
    results = pd.DataFrame(rows)
    results.round(3).to_csv("outputs/task_b_windows.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    summary = results.groupby(["min_samples_leaf", "training_seasons"]).agg(
        mean=("forest_f1", "mean"),
        variance=("forest_f1", "var"),
        std=("forest_f1", "std"),
        worst_season=("forest_f1", "min"),
        mean_auc=("forest_auc", "mean"),
        right_changes=("right_changes", "sum"),
        real_changes=("real_changes", "sum"),
    )
    summary["seasons_better_than_random"] = (results["forest_f1"] > results["random_k_f1"]) \
        .groupby([results["min_samples_leaf"], results["training_seasons"]]).sum()
    random_mean = results.groupby("training_seasons")["random_k_f1"].mean()

    print(f"F1 of the random forest per min_samples_leaf and training window, test seasons {study_years[0]}-{study_years[-1]} "
          f"({len(study_years)} seasons, 2004 is missing). Higher is better.\n")
    print(summary.round(4))
    print("\nBaselines: no changes F1 = 0; random k teams F1 (mean):", random_mean.round(4).to_dict())

    # Choice (teacher): we want a window that is better than the baseline in every
    # season (or in as many as possible) with a small variance, even if its mean
    # is not the best. So: the (min_samples_leaf, window) pairs better than random k
    # teams in the most seasons, and among them the one with the smallest variance.
    # Note: with many combinations and only 5 seasons, the best one is partly luck,
    # so these numbers are a bit optimistic. The real test is 2010-2013.
    most = summary["seasons_better_than_random"].max()
    candidates = summary[summary["seasons_better_than_random"] == most]
    best = candidates["variance"].idxmin()
    print(f"\n(min_samples_leaf, window) better than random k teams in {most} of {len(study_years)} seasons: "
          f"{list(candidates.index)}")
    print(f"Best (smallest variance among them): min_samples_leaf {best[0]}, {best[1]} seasons")
    print(f"Used for the predictions: min_samples_leaf {LEAF_SIZE}, {WINDOW} seasons")

    chosen = results[results["min_samples_leaf"] == LEAF_SIZE]
    by_year = chosen.pivot(index="test_season", columns="training_seasons", values="forest_f1")
    by_year.columns = [f"Random forest, {w} seasons" for w in by_year.columns]
    ax = by_year.plot(figsize=(9, 5), marker="o")
    chosen[chosen["training_seasons"] == WINDOW].set_index("test_season")["random_k_f1"].plot(
        ax=ax, color="black", linestyle="--", label=f"Random k teams ({WINDOW} seasons)")
    ax.legend()
    plt.ylabel("F1")
    plt.title(f"Task (b): F1 per test season and training window (min_samples_leaf {LEAF_SIZE})")
    plt.tight_layout()
    plt.savefig("outputs/task_b_windows.png")
    plt.close()
    print("\nSaved outputs/task_b_windows.csv and outputs/task_b_windows.png")

    # ---------- how sure are we? bootstrap confidence interval ----------
    # Lab 2 (8.4): resample the test set with replacement many times and recompute the
    # metric. We resample the teams inside each season (so every season keeps its size)
    # and look at model - baseline. If the 95% interval does not include 0, the gain
    # is unlikely to be luck. Limitation: teams of the same season are not independent
    # (Lab 2 suggests resampling whole groups), but 5 seasons are too few to resample.

    # F1 is not a mean over teams, so each resample recomputes it. The baseline is k
    # random teams: on average k * changes / teams of them are right in each season.
    def f1_vs_random(sample):
        per_season = sample.groupby("season").agg(right=("right", "sum"), predicted=("pred", "sum"),
                                                  changes=("y", "sum"), teams=("y", "size"))
        random_right = (per_season["predicted"] * per_season["changes"] / per_season["teams"]).sum()
        total = per_season["predicted"].sum() + per_season["changes"].sum()
        return 2 * per_season["right"].sum() / total - 2 * random_right / total


    def bootstrap(per_team, n=1000, seed=0):
        rng = np.random.default_rng(seed)
        return np.array([f1_vs_random(per_team.groupby("season").sample(frac=1, replace=True,
                                                                        random_state=rng.integers(10**9)))
                         for _ in range(n)])


    per_team = []
    for year in study_years:
        test, prob, pred, k = predict(year, WINDOW)
        y = test[TARGET].astype(int).values
        per_team.append(pd.DataFrame({"season": year, "y": y, "pred": pred, "right": (y == 1) & (pred == 1)}))
    per_team = pd.concat(per_team)
    diffs = bootstrap(per_team)
    low, high = np.percentile(diffs, [2.5, 97.5])
    print(f"\nForest - random k teams F1 (min_samples_leaf {LEAF_SIZE}, {WINDOW} seasons, {len(per_team)} team-seasons):")
    print(f"  {f1_vs_random(per_team):+.3f}, 95% interval [{low:+.3f}, {high:+.3f}],"
          f" forest better in {(diffs > 0).mean():.0%} of the resamples")
    print("  -> " + ("the interval is above 0: the forest is better than random"
                     if low > 0 else "the interval includes 0: the gain could be luck"))
