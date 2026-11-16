# Task (b): predict which teams will change coach during the season.
# The target is coach_changed_in_season (the team had more than one coach that season).
# Only ~17% of the team-seasons have a change, so the classes are imbalanced and
# the models use class weights (computed on the training seasons only).
# Each model gives a probability of a change, and we predict a change for the k teams
# with the highest probability, where k = usual number of changes per season.
# As a comparison we also try a threshold on the probability (chosen on the training
# seasons), but it ends up predicting a change for almost every team.
# Sliding window: to test season Y we train on the 9 seasons before it (like task a).
#
# Run prepare.py first, then:
#   python task_b.py 2008   predicts one season (any season with 9 seasons before it,
#                           also one added by the teacher)
#   python task_b.py        tests all the models on every season with results

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

WINDOW = 9
# Model used for the predictions (chosen after running python task_b.py).
FINAL_MODEL = "logistic"
RECENT = range(1995, 2010)
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

MODEL_NAMES = {
    "baseline": "Baseline (no changes)", "all_change": "Baseline (all teams change)",
    "random": "Baseline (random k teams)",
    "logistic": "Logistic regression", "knn": "kNN", "tree": "Decision tree", "forest": "Random forest", "boosting": "Gradient boosting",
}

df = pd.read_csv("outputs/team_seasons.csv")
df = df.sort_values(["year", "team_id"]).reset_index(drop=True)

seasons = sorted(df["year"].unique())
PREDICTABLE = seasons[WINDOW:]
with_results = set(df.loc[df[TARGET].notna(), "year"])
TEST_YEARS = [y for y in PREDICTABLE if y in with_results]


def imputer():
    return SimpleImputer(strategy="median", keep_empty_features=True)


# kNN has no class weights, so it gets a smaller k and relies on the threshold.
def make_models():
    return {
        "logistic": make_pipeline(imputer(), StandardScaler(),
                                  LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)),
        "knn": make_pipeline(imputer(), StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
        "tree": make_pipeline(imputer(), DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                                                random_state=0)),
        "forest": make_pipeline(imputer(),
                                RandomForestClassifier(n_estimators=200, min_samples_leaf=10,
                                                       class_weight="balanced", random_state=0, n_jobs=-1)),
        "boosting": make_pipeline(imputer(),
                                  GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                                                             random_state=0)),
    }


def fit(name, model, X, y):
    if name == "boosting":  # no class_weight option, so we give weights to the rows
        model.fit(X, y, gradientboostingclassifier__sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def training_seasons(year):
    before = [y for y in seasons if y < year][-WINDOW:]
    train = df[df["year"].isin(before)]
    return train[train[TARGET].notna()]


def f1(y, pred):
    tp = ((pred == 1) & (y == 1)).sum()
    return 2 * tp / (pred.sum() + y.sum()) if pred.sum() + y.sum() > 0 else 0


# The threshold is chosen only with the training seasons: we split them in 3 groups
# of seasons, predict each group with a model trained on the other two, and keep
# the threshold with the best F1 on those predictions.
def choose_threshold(name, train):
    X, y = train[FEATURES], train[TARGET].astype(int)
    probs = np.zeros(len(train))
    for fit_idx, val_idx in GroupKFold(n_splits=3).split(X, y, groups=train["year"]):
        if y.iloc[fit_idx].nunique() < 2:
            probs[val_idx] = y.iloc[fit_idx].mean()
            continue
        model = fit(name, make_models()[name], X.iloc[fit_idx], y.iloc[fit_idx])
        probs[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    candidates = np.arange(0.05, 0.96, 0.05)
    return max(candidates, key=lambda t: f1(y.values, (probs >= t).astype(int)))


# Trains every model on the seasons before `year` and predicts the teams of `year`.
# Each model gives a probability of a coach change. The prediction is a change for the
# k teams with the highest probability, where k is the share of teams that changed coach
# in the training seasons times the number of teams this season. The yes/no with the
# threshold (pred_threshold) is only kept as a comparison.
def top_k(prob, k):
    pred = np.zeros(len(prob), dtype=int)
    pred[np.argsort(-prob, kind="stable")[:k]] = 1
    return pred


def predict_season(year):
    train = training_seasons(year)
    test = df[df["year"] == year]
    y = train[TARGET].astype(int)
    k = round(y.mean() * len(test))
    random_prob = np.random.default_rng(year).random(len(test))
    out = {
        "baseline": test.assign(prob=0.0, pred=0, pred_threshold=0, k=k, threshold=np.nan),
        "all_change": test.assign(prob=0.0, pred=1, pred_threshold=1, k=k, threshold=np.nan),
        "random": test.assign(prob=random_prob, pred=top_k(random_prob, k), pred_threshold=0, k=k,
                              threshold=np.nan),
    }
    for name, model in make_models().items():
        if y.nunique() < 2:  # no coach changes at all in the training seasons
            prob = np.zeros(len(test))
            threshold = np.nan
        else:
            prob = fit(name, model, train[FEATURES], y).predict_proba(test[FEATURES])[:, 1]
            threshold = choose_threshold(name, train)
        out[name] = test.assign(prob=prob, pred=top_k(prob, k), pred_threshold=(prob >= threshold).astype(int),
                                k=k, threshold=threshold)
    return out


# Counts per season. Many seasons have zero or one change, so F1 per season is very
# noisy; we add up the counts over all the seasons of an era and compute F1 from that.
def counts(s):
    s = s[s[TARGET].notna()]
    y = s[TARGET].astype(int)
    row = {"teams": len(s), "changes": y.sum()}
    for col, tag in [("pred", ""), ("pred_threshold", "threshold_")]:
        row[tag + "tp"] = ((s[col] == 1) & (y == 1)).sum()
        row[tag + "fp"] = ((s[col] == 1) & (y == 0)).sum()
        row[tag + "fn"] = ((s[col] == 0) & (y == 1)).sum()
    row["correct"] = row["teams"] - row["fp"] - row["fn"]
    row["auc"] = roc_auc_score(y, s["prob"]) if y.nunique() == 2 else np.nan
    return row


def metrics(c, tag=""):
    tp, fp, fn = c[tag + "tp"], c[tag + "fp"], c[tag + "fn"]
    return {
        "precision": tp / (tp + fp) if tp + fp > 0 else 0,
        "recall": tp / (tp + fn) if tp + fn > 0 else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0,
    }


# ---------- one season: python task_b.py 2008 ----------

if len(sys.argv) > 1:
    year = int(sys.argv[1])
    if year not in PREDICTABLE:
        sys.exit(f"Season {year} can't be predicted: it needs {WINDOW} seasons before it "
                 f"(possible: {PREDICTABLE[0]}-{PREDICTABLE[-1]}, no 2004).")
    predicted = predict_season(year)
    has_results = year in with_results

    final = predicted[FINAL_MODEL]
    k = final["k"].iloc[0]
    final = pd.DataFrame({
        "team": final["team"],
        "ranking_group": final["ranking_group"],
        "coach_at_start": final["coach_id"],
        "change_probability": final["prob"].round(3),
        "predicted_change": final["pred"],
        "real_change": final[TARGET].astype("Int64"),
    }).sort_values("change_probability", ascending=False)
    if not has_results:
        final = final.drop(columns="real_change")
    final.to_csv(f"outputs/task_b_prediction_{year}.csv", index=False)

    train_years = training_seasons(year)["year"]
    print(f"Season {year}: prediction with {MODEL_NAMES[FINAL_MODEL]} "
          f"(trained on {train_years.min()}-{train_years.max()}, top {k} teams)\n")
    print(final.to_string(index=False))
    print(f"\n{final['predicted_change'].sum()} teams predicted to change coach")
    print(f"Saved outputs/task_b_prediction_{year}.csv")

    if has_results:
        print("\nResults of every model in this season:")
        for name, s in predicted.items():
            c = counts(s)
            m = metrics(c)
            print(f"  {MODEL_NAMES[name]:<28} predicted {c['tp'] + c['fp']:>2}, right {c['tp']} of "
                  f"{c['changes']}, F1 {m['f1']:.3f}, AUC {c['auc']:.3f}")
    sys.exit()


# ---------- sliding window over all the test seasons ----------

rows = []
for year in TEST_YEARS:
    for name, s in predict_season(year).items():
        rows.append({"year": year, "model": name, **counts(s)})
results = pd.DataFrame(rows)

per_season = results.copy()
for i, r in per_season.iterrows():
    for key, value in metrics(r).items():
        per_season.loc[i, key] = value
per_season["accuracy"] = per_season["correct"] / per_season["teams"]
per_season["model"] = per_season["model"].map(MODEL_NAMES)
per_season = per_season.rename(columns={"year": "test_season", "changes": "real_changes",
                                        "tp": "right_changes", "fp": "wrong_changes",
                                        "fn": "missed_changes"})
per_season[["test_season", "model", "teams", "real_changes", "right_changes", "wrong_changes",
            "missed_changes", "accuracy", "precision", "recall", "f1", "auc"]] \
    .round(3).to_csv("outputs/task_b_results.csv", index=False)


def summary_of(years):
    part = results[results["year"].isin(years)]
    total = part.groupby("model")[["teams", "correct", "tp", "fp", "fn",
                                   "threshold_tp", "threshold_fp", "threshold_fn"]].sum()
    rows = {}
    for name, c in total.iterrows():
        m = metrics(c)
        rows[name] = {
            "accuracy": c["correct"] / c["teams"], **m,
            "auc": part.loc[part["model"] == name, "auc"].mean(),
            "threshold_f1": metrics(c, "threshold_")["f1"],
        }
    return pd.DataFrame(rows).T.sort_values("f1", ascending=False)


eras = {
    f"all seasons ({TEST_YEARS[0]}-{TEST_YEARS[-1]})": TEST_YEARS,
    f"{TEST_YEARS[0]}-1966 (before the 1967 expansion)": range(TEST_YEARS[0], 1967),
    "1967-1994": range(1967, 1995),
    f"{RECENT[0]}-{RECENT[-1]} (recent)": RECENT,
}
print("Coach changes during the season, predicting a change for the top k teams.")
print("F1, precision, recall come from the counts of all the seasons together; AUC is the")
print("mean over the seasons with at least one change; threshold_f1 is the F1 with the threshold.")
for era, years in eras.items():
    n = sum(y in years for y in TEST_YEARS)
    print(f"\n{era}, {n} test seasons:")
    print(summary_of(years).round(3))

f1_by_year = per_season.pivot(index="test_season", columns="model", values="f1")
f1_by_year.drop(columns=[MODEL_NAMES["baseline"], MODEL_NAMES["all_change"]]).rolling(5, min_periods=1).mean().plot(figsize=(10, 5))
plt.ylabel("F1 (mean of 5 seasons)")
plt.title("Task (b): F1 per test season")
plt.tight_layout()
plt.savefig("outputs/task_b_f1_per_season.png")
plt.close()

best = [m for m in summary_of(RECENT).index if m not in ("baseline", "all_change", "random")][0]
print(f"\nBest model on {RECENT[0]}-{RECENT[-1]} (F1): {MODEL_NAMES[best]} "
      f"(FINAL_MODEL is {MODEL_NAMES[FINAL_MODEL]})")
print("To predict a season: python task_b.py <year>")
