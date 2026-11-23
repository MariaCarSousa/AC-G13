# Task (c): predict the penalty minutes (PIM) of each team in the season.
# We work with penalty minutes per game (seasons can have a different number of
# games, e.g. 48 in 1994) and multiply by the games of the season to get the total.
# Penalty minutes go down almost every season (about 20 per game in 1995, 13 in
# 2009), so a model trained on older seasons predicts too many. That is why Ridge
# predicts the CHANGE from last season (this season - last season, per game):
# a prediction of 0 is the baseline, and Ridge only moves away from it when the
# features say so.
# Ridge alpha: a big alpha (strong regularization) pulls the change towards 0.
# Baseline (teacher): each team gets the same penalty minutes per game as last season.
# Metrics (teacher): MAE (in penalty minutes per team) and Spearman correlation
# between the predicted and the real penalty minutes of the teams of the season.
#
# Training window and alpha: same study as task (a). For every test season Y from
# 2004 to 2009 (2004 is missing, lockout) we train on the last 9, 7, 5, 3 and 1
# seasons before Y, with several values of alpha. The teacher tests on 2010-2013,
# so 2005-2009 works as our validation seasons and we tune both on them. We want a
# combination that is better than the baseline in as many seasons as possible,
# with a small variance.
#
# Run prepare.py first, then:
#   python task_c.py        compares the training windows and alphas on 2005-2009
#   python task_c.py 2012   predicts the penalty minutes of one season, trained
#                           only on the seasons we have results for (2009 and before)

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WINDOWS = [9, 7, 5, 3, 1]
ALPHAS = [1, 10, 30, 100, 300, 1000]
# Window and alpha used for the predictions, chosen with python task_c.py: alpha 100
# with 3 seasons was better than the baseline in all 5 test seasons and had the
# smallest variance among those
WINDOW = 3
ALPHA = 100
STUDY_YEARS = range(2004, 2010)
TARGET = "penalty_minutes_per_game"

FEATURES = [
    # team, last seasons
    "last_season_penalty_minutes_per_game", "last_3_seasons_penalty_minutes_per_game",
    "last_season_league_penalty_minutes_per_game", "last_season_penalty_kill_%",
    "last_season_power_play_%", "last_season_points_%", "last_season_goals_against_per_game",
    # players
    "players_last_season_penalty_minutes_per_game", "players_avg_age", "top3_players_points_per_game",
    # coach
    "coach_new_this_season", "coach_seasons_total", "coach_win_%",
]

df = pd.read_csv("outputs/team_seasons.csv")
df = df.sort_values(["year", "team_id"]).reset_index(drop=True)

# seasons with results (we can train on them and test them)
known = sorted(df.loc[df[TARGET].notna(), "year"].unique())
study_years = [y for y in STUDY_YEARS if y in known]


# Training rows for season `year`: the last `window` seasons with results before it.
def training_seasons(year, window):
    before = [y for y in known if y < year][-window:]
    return df[df["year"].isin(before)]


# Imputation and scaling are fitted on the training seasons only.
def make_model(alpha):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=alpha))


# Games of each team in the season. Seasons we still have to predict have no
# games yet, so we use the games of the last season with results.
def games(test):
    last_known = df.loc[df["year"] == max(y for y in known if y < test["year"].iloc[0]), "games"]
    return test["games"].fillna(last_known.median())


# Predicted penalty minutes (season total) of the teams of `year`, for Ridge and
# for the baseline (same penalty minutes per game as last season).
# Teams without a last season (new teams) get the training average.
def predict(year, window, alpha=ALPHA):
    train = training_seasons(year, window).dropna(subset=["last_season_penalty_minutes_per_game"])
    test = df[df["year"] == year]
    change = train[TARGET] - train["last_season_penalty_minutes_per_game"]
    model = make_model(alpha).fit(train[FEATURES], change)
    last = test["last_season_penalty_minutes_per_game"].fillna(train[TARGET].mean())
    ridge = (last + model.predict(test[FEATURES])) * games(test)
    baseline = last * games(test)
    return test, ridge, baseline


def mae(test, pred):
    return (pred - test["penalty_minutes"]).abs().mean()


def spearman(test, pred):
    return spearmanr(pred, test["penalty_minutes"]).statistic


# The code below only runs with python task_c.py (other scripts can import the
# functions above without running it).
if __name__ == "__main__":
    # ---------- one season: python task_c.py 2012 ----------

    if len(sys.argv) > 1:
        year = int(sys.argv[1])
        if year not in df["year"].values or len([y for y in known if y < year]) < WINDOW:
            sys.exit(f"Season {year} can't be predicted: it must be in the data (up to {df['year'].max()}) "
                     f"and have {WINDOW} seasons with results before it.")
        test, ridge, baseline = predict(year, WINDOW)
        has_results = test["penalty_minutes"].notna().all()

        final = pd.DataFrame({
            "team": test["team"],
            "predicted_penalty_minutes": ridge.round().astype(int),
            "baseline_penalty_minutes": baseline.round().astype(int),
        })
        if has_results:
            final["real_penalty_minutes"] = test["penalty_minutes"]
        final = final.sort_values("predicted_penalty_minutes", ascending=False)
        final.to_csv(f"outputs/task_c_prediction_{year}.csv", index=False)

        train_years = training_seasons(year, WINDOW)["year"]
        print(f"Season {year}: Ridge (alpha {ALPHA}) trained on the last {WINDOW} seasons with results "
              f"({train_years.min()}-{train_years.max()})\n")
        print(final.to_string(index=False))
        print(f"\nSaved outputs/task_c_prediction_{year}.csv")

        if has_results:
            print("\nMAE in this season (penalty minutes per team, lower is better):")
            print(f"  Ridge    {mae(test, ridge):.0f}")
            print(f"  Baseline {mae(test, baseline):.0f}")
            print("Spearman in this season (-1 to 1, higher is better):")
            print(f"  Ridge    {spearman(test, ridge):.3f}")
            print(f"  Baseline {spearman(test, baseline):.3f}")
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
                    "ridge_mae": mae(test, ridge),
                    "baseline_mae": mae(test, baseline),
                    "ridge_spearman": spearman(test, ridge),
                    "baseline_spearman": spearman(test, baseline),
                })
    results = pd.DataFrame(rows)
    results.round(3).to_csv("outputs/task_c_windows.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    baseline_per_season = results.groupby("test_season")[["baseline_mae", "baseline_spearman"]].first()
    summary = results.groupby(["alpha", "training_seasons"]).agg(
        mae=("ridge_mae", "mean"),
        variance=("ridge_mae", "var"),
        worst_season=("ridge_mae", "max"),
        spearman=("ridge_spearman", "mean"),
    )
    summary["seasons_better_than_baseline"] = (results["ridge_mae"] < results["baseline_mae"]) \
        .groupby([results["alpha"], results["training_seasons"]]).sum()

    print(f"MAE of Ridge per alpha and training window, test seasons {study_years[0]}-{study_years[-1]} "
          f"({len(study_years)} seasons, 2004 is missing)")
    print("mae = error of the predicted penalty minutes per team (lower is better)\n")
    print(summary.round({"mae": 1, "variance": 1, "worst_season": 1, "spearman": 3}))
    print(f"\nBaseline (last season's penalty minutes per game): mae {baseline_per_season['baseline_mae'].mean():.1f}, "
          f"variance {baseline_per_season['baseline_mae'].var():.1f}, "
          f"worst season {baseline_per_season['baseline_mae'].max():.1f}, "
          f"spearman {baseline_per_season['baseline_spearman'].mean():.3f}")

    # Choice (teacher): the (alpha, window) pairs better than the baseline in the most
    # seasons, and among them the one with the smallest variance.
    # Note: with many combinations and only 5 seasons, the best one is partly luck,
    # so these numbers are a bit optimistic. The real test is 2010-2013.
    most = summary["seasons_better_than_baseline"].max()
    candidates = summary[summary["seasons_better_than_baseline"] == most]
    best = candidates["variance"].idxmin()
    print(f"\n(alpha, window) better than the baseline in {most} of {len(study_years)} seasons: {list(candidates.index)}")
    print(f"Best (smallest variance among them): alpha {best[0]}, {best[1]} seasons")
    print(f"Used for the predictions: alpha {ALPHA}, {WINDOW} seasons")

    chosen = results[results["alpha"] == ALPHA]
    by_year = chosen.pivot(index="test_season", columns="training_seasons", values="ridge_mae")
    by_year.columns = [f"Ridge, {w} seasons" for w in by_year.columns]
    ax = by_year.plot(figsize=(9, 5), marker="o")
    baseline_per_season["baseline_mae"].plot(ax=ax, color="black", linestyle="--", label="Baseline (last season)")
    ax.legend()
    plt.ylabel("MAE (penalty minutes per team)")
    plt.title(f"Task (c): MAE per test season and training window (alpha {ALPHA})")
    plt.tight_layout()
    plt.savefig("outputs/task_c_windows.png")
    plt.close()
    print("\nSaved outputs/task_c_windows.csv and outputs/task_c_windows.png")

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
        per_team.append(pd.DataFrame({"season": year,
                                      "ridge": (ridge - test["penalty_minutes"]).abs(),
                                      "baseline": (baseline - test["penalty_minutes"]).abs()}))
    per_team = pd.concat(per_team)
    diffs = bootstrap(per_team)
    low, high = np.percentile(diffs, [2.5, 97.5])
    print(f"\nRidge - baseline MAE (alpha {ALPHA}, {WINDOW} seasons, {len(per_team)} team-seasons):")
    print(f"  {diffs.mean():+.1f} penalty minutes, 95% interval [{low:+.1f}, {high:+.1f}],"
          f" Ridge better in {(diffs < 0).mean():.0%} of the resamples")
    print("  -> " + ("the interval is below 0: Ridge is better than the baseline"
                     if high < 0 else "the interval includes 0: the gain could be luck"))
