# Task (a): predict the final regular-season ranking of the teams.
# We predict the points % of each team with Ridge regression and then sort the
# teams inside their ranking group. The group depends on what we know about the
# teams (teacher feedback): one ranking per division if we know the divisions,
# per conference if we only know the conferences, else one ranking of all teams.
#
# Model: Ridge. We tested Ridge, kNN, a decision tree, a random forest and
# gradient boosting on 1995-2009 (training on the 9 seasons before each test
# season) and Ridge had the lowest mean rank error, so we only study Ridge.
#
# Training window and Ridge alpha: for every test season Y from 2004 to 2009
# (2004 is missing, lockout) we train on the last 9, 7, 5, 3 and 1 seasons before
# Y, with several values of alpha, and compare the rank error. The teacher tests
# on 2010-2013, so 2005-2009 works as our validation seasons (the closest ones to
# 2010-2013) and we tune both on them. We want a combination that is better than
# the baseline in as many seasons as possible, with a small variance, more than
# one with the best mean.
#
# Run prepare.py first, then:
#   python task_a.py        compares the training windows and alphas on 2005-2009
#   python task_a.py 2012   predicts the ranking of one season, trained only on
#                           the seasons we have results for (2009 and before)

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WINDOWS = [9, 7, 5, 3, 1]
ALPHAS = [0.1, 1, 10, 100, 1000]
# Window and alpha used for the predictions, chosen with python task_a.py: alpha 100
# with 5 seasons was better than the baseline in 4 of 5 test seasons (the most) and
# had the smallest variance among those
WINDOW = 5
ALPHA = 100
STUDY_YEARS = range(2004, 2010)

# Features chosen by feature_selection.py (filter method, on the seasons 1995-2003):
# features related to the target (|Spearman| >= 0.05) and, of each very correlated pair
# (|Spearman| > 0.8), the one more related to the target. On 2005-2009 this set was
# better than using all the features (see outputs/feature_selection/summary.csv).
FEATURES = [
    "last_season_points_%",
    "last_season_goals_against_per_game",
    "last_season_playoff_wins",
    "coach_win_%",
    "players_last_season_points",
    "last_season_penalty_kill_%",
    "last_season_goals_for_per_game",
    "players_awards",
    "players_avg_age",
    "last_3_seasons_penalty_minutes_per_game",
    "last_season_power_play_%",
    "coach_playoff_wins",
    "coach_awards",
    "goalies_last_season_save_%",
    "coach_seasons_with_team",
    "last_season_league_penalty_minutes_per_game",
    "players_last_season_penalty_minutes_per_game",
    "players_avg_weight",
    "last_season_coach_changed",
    "last_season_points_%_change",
    "players_avg_height",
]

df = pd.read_csv("outputs/team_seasons.csv")
# the csv is sorted by the real rank, so we sort by team id to not
# leak the answer when two teams get the same prediction
df = df.sort_values(["year", "team_id"]).reset_index(drop=True)

# seasons with results (we can train on them and test them)
known = sorted(df.loc[df["points_%"].notna(), "year"].unique())
study_years = [y for y in STUDY_YEARS if y in known]


# Training rows for season `year`: the last `window` seasons with results before it.
# For 2012 that is the seasons up to 2009, because we don't have 2010 and 2011.
def training_seasons(year, window):
    before = [y for y in known if y < year][-window:]
    return df[df["year"].isin(before)]


# Imputation and scaling are fitted on the training seasons only.
def make_model(alpha):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=alpha))


# Ranking groups for each level of information:
#   "given" = what the data has (division, else conference, else all teams)
#   "conference" = as if we only knew the conferences
#   "all teams" = as if we only knew the teams
def groups(season, level):
    if level == "given":
        return season["ranking_group"]
    if level == "conference":
        return season["conference"].fillna(season["league"] + " (all teams)")
    return season["league"] + " (all teams)"


# Sorts the teams inside each group by `score` (higher = better), ties broken
# by the `tie_break` columns. Returns the rank on a 0-1 scale (0 = first).
def rank_0_to_1(season, group, score, tie_break):
    s = season.assign(group=group, score=score)
    s = s.sort_values(["score"] + tie_break, ascending=False)
    rank = s.groupby("group").cumcount()
    teams = s.groupby("group")["team_id"].transform("count")
    return (rank / (teams - 1)).reindex(season.index)


# Real rank: points, then wins, then goal difference (like the NHL).
def real_rank(season, level):
    return rank_0_to_1(season, groups(season, level), season["points_%"], ["wins", "goal_difference"])


# Predicted rank from a predicted points %. Ties broken by last season's points %.
def predicted_rank(season, level, pred):
    tie = season.assign(tie=season["last_season_points_%"].fillna(0))
    return rank_0_to_1(tie, groups(season, level), pred, ["tie"])


# Predicted points % of the teams of `year`, for Ridge and for the baseline
# (each team gets the same points % as last season).
def predict(year, window, alpha=ALPHA):
    train = training_seasons(year, window)
    test = df[df["year"] == year]
    model = make_model(alpha).fit(train[FEATURES], train["points_%"])
    ridge = pd.Series(model.predict(test[FEATURES]), index=test.index)
    baseline = test["last_season_points_%"].fillna(train["points_%"].mean())
    return test, ridge, baseline


def rank_error(test, pred, level):
    return (predicted_rank(test, level, pred) - real_rank(test, level)).abs().mean()


# MAE in standings points (2 per win): how many points the predicted total of a
# team is wrong on average. Easier to read than the error in points % (0 to 1).
def mae(test, pred):
    return ((pred - test["points_%"]) * 2 * test["games"]).abs().mean()


# The code below only runs with python task_a.py (other scripts can import the
# functions above without running it).
if __name__ == "__main__":
    # ---------- one season: python task_a.py 2012 ----------

    if len(sys.argv) > 1:
        year = int(sys.argv[1])
        if year not in df["year"].values or len([y for y in known if y < year]) < WINDOW:
            sys.exit(f"Season {year} can't be predicted: it must be in the data (up to {df['year'].max()}) "
                     f"and have {WINDOW} seasons with results before it.")
        test, ridge, baseline = predict(year, WINDOW)
        has_results = test["points_%"].notna().all()

        final = pd.DataFrame({
            "ranking_group": groups(test, "given"),
            "team": test["team"],
            "predicted_points_%": ridge.round(3),
            "predicted_rank_0_to_1": predicted_rank(test, "given", ridge).round(3),
        })
        final["predicted_rank"] = final.groupby("ranking_group")["predicted_rank_0_to_1"].rank(method="first")
        final["predicted_rank"] = final["predicted_rank"].astype(int)
        if has_results:
            final["real_rank"] = (real_rank(test, "given").groupby(final["ranking_group"])
                                  .rank(method="first").astype(int))
        final = final.sort_values(["ranking_group", "predicted_rank"])
        cols = ["ranking_group", "predicted_rank"] + (["real_rank"] if has_results else []) + \
               ["team", "predicted_points_%", "predicted_rank_0_to_1"]
        final = final[cols]
        final.to_csv(f"outputs/task_a_prediction_{year}.csv", index=False)

        train_years = training_seasons(year, WINDOW)["year"]
        print(f"Season {year}: Ridge (alpha {ALPHA}) trained on the last {WINDOW} seasons with results "
              f"({train_years.min()}-{train_years.max()})")
        for group, rows in final.groupby("ranking_group"):
            print(f"\n{group}")
            print(rows.drop(columns="ranking_group").to_string(index=False))
        print(f"\nSaved outputs/task_a_prediction_{year}.csv")

        if has_results:
            print("\nRank error in this season (0 to 1, lower is better):")
            print(f"  Ridge    {rank_error(test, ridge, 'given'):.3f}")
            print(f"  Baseline {rank_error(test, baseline, 'given'):.3f}")
            print("\nMAE in this season (standings points per team, lower is better):")
            print(f"  Ridge    {mae(test, ridge):.1f}")
            print(f"  Baseline {mae(test, baseline):.1f}")
        sys.exit()


    # ---------- compare the training windows and alphas on 2005-2009 ----------

    rows = []
    for year in study_years:
        for alpha in ALPHAS:
            for window in WINDOWS:
                test, ridge, baseline = predict(year, window, alpha)
                rows.append({
                    "test_season": year,
                    "alpha": alpha,
                    "training_seasons": window,
                    "ridge_rank_error": rank_error(test, ridge, "given"),
                    "baseline_rank_error": rank_error(test, baseline, "given"),
                    "ridge_mae": mae(test, ridge),
                    "baseline_mae": mae(test, baseline),
                    "ridge_rank_error_by_conference": rank_error(test, ridge, "conference"),
                    "ridge_rank_error_all_teams": rank_error(test, ridge, "all teams"),
                })
    results = pd.DataFrame(rows)
    results.round(3).to_csv("outputs/task_a_windows.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    baseline_per_season = results.groupby("test_season")["baseline_rank_error"].first()
    baseline_mean = baseline_per_season.mean()
    summary = results.groupby(["alpha", "training_seasons"]).agg(
        mean=("ridge_rank_error", "mean"),
        variance=("ridge_rank_error", "var"),
        std=("ridge_rank_error", "std"),
        worst_season=("ridge_rank_error", "max"),
        mae=("ridge_mae", "mean"),
    )
    summary["seasons_better_than_baseline"] = (results["ridge_rank_error"] < results["baseline_rank_error"]) \
        .groupby([results["alpha"], results["training_seasons"]]).sum()

    print(f"Rank error of Ridge per alpha and training window, test seasons {study_years[0]}-{study_years[-1]} "
          f"({len(study_years)} seasons, 2004 is missing)")
    print("Ranking per division (0 to 1, lower is better)\n")
    print(summary.round(4).round({"mae": 1}))
    print(f"\nBaseline (last season's points %): mean {baseline_mean:.4f}, "
          f"variance {baseline_per_season.var():.4f}, worst season {baseline_per_season.max():.4f}, "
          f"mae {results['baseline_mae'].mean():.1f}")
    print("mae = error of the predicted total in standings points per team")

    # Choice (teacher): we want a window that is better than the baseline in every
    # season (or in as many as possible) with a small variance, even if its mean
    # is not the best. So: the (alpha, window) pairs better than the baseline in the
    # most seasons, and among them the one with the smallest variance.
    # Note: with many combinations and only 5 seasons, the best one is partly luck,
    # so these numbers are a bit optimistic. The real test is 2010-2013.
    most = summary["seasons_better_than_baseline"].max()
    candidates = summary[summary["seasons_better_than_baseline"] == most]
    best = candidates["variance"].idxmin()
    print(f"\n(alpha, window) better than the baseline in {most} of {len(study_years)} seasons: {list(candidates.index)}")
    print(f"Best (smallest variance among them): alpha {best[0]}, {best[1]} seasons")
    print(f"Used for the predictions: alpha {ALPHA}, {WINDOW} seasons")

    chosen = results[results["alpha"] == ALPHA]
    print(f"\nRank error of each window (alpha {ALPHA}) if we only knew the conferences, or only the teams:")
    print(chosen.groupby("training_seasons")[["ridge_rank_error_by_conference", "ridge_rank_error_all_teams"]]
          .mean().round(4))

    by_year = chosen.pivot(index="test_season", columns="training_seasons", values="ridge_rank_error")
    by_year.columns = [f"Ridge, {w} seasons" for w in by_year.columns]
    ax = by_year.plot(figsize=(9, 5), marker="o")
    baseline_per_season.plot(ax=ax, color="black", linestyle="--", label="Baseline (last season)")
    ax.legend()
    plt.ylabel("mean rank error (0 to 1)")
    plt.title(f"Task (a): rank error per test season and training window (alpha {ALPHA})")
    plt.tight_layout()
    plt.savefig("outputs/task_a_windows.png")
    plt.close()
    print("\nSaved outputs/task_a_windows.csv and outputs/task_a_windows.png")

    # ---------- how sure are we? bootstrap confidence interval ----------
    # Lab 2 (8.4): resample the test set with replacement many times and recompute the
    # metric. We resample the teams inside each season (so every season keeps its size)
    # and look at model - baseline. If the 95% interval does not include 0, the gain
    # is unlikely to be luck. Limitation: teams of the same season are not independent
    # (Lab 2 suggests resampling whole groups), but 5 seasons are too few to resample.

    def bootstrap(per_team, n=1000, seed=0):
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(n):
            sample = per_team.groupby("season").sample(frac=1, replace=True, random_state=rng.integers(10**9))
            diffs.append(sample["ridge"].mean() - sample["baseline"].mean())
        return np.array(diffs)


    per_team = []
    for year in study_years:
        test, ridge, baseline = predict(year, WINDOW)
        real = real_rank(test, "given")
        per_team.append(pd.DataFrame({"season": year,
                                      "ridge": (predicted_rank(test, "given", ridge) - real).abs(),
                                      "baseline": (predicted_rank(test, "given", baseline) - real).abs()}))
    per_team = pd.concat(per_team)
    diffs = bootstrap(per_team)
    low, high = np.percentile(diffs, [2.5, 97.5])
    print(f"\nRidge - baseline rank error (alpha {ALPHA}, {WINDOW} seasons, {len(per_team)} team-seasons):")
    print(f"  {diffs.mean():+.4f}, 95% interval [{low:+.4f}, {high:+.4f}], Ridge better in {(diffs < 0).mean():.0%}"
          " of the resamples")
    print("  -> " + ("the interval is below 0: Ridge is better than the baseline"
                     if high < 0 else "the interval includes 0: the gain could be luck"))
