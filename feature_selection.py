# Feature selection for the three tasks (ch04 slides: filter and wrapper methods).
# To not choose the features on the seasons we report, we select them on the seasons
# 1995-2003 ("selection seasons", each one predicted with the window before it) and
# then compare on 2005-2009 (validation seasons, as in the task scripts):
#   - all features
#   - filter: drop features almost unrelated to the target (|Spearman| < 0.05) and,
#     for each very correlated pair (|Spearman| > 0.8), the one less related to the target
#   - wrapper (forward selection): start with no features and add, one at a time, the
#     feature that improves the model most; stop when no feature improves it
# The candidates are all the features of team_seasons.csv (known before the season).
# Each task keeps the parameters chosen in its script (window, alpha / min_samples_leaf).
# Run prepare.py first, then: python feature_selection.py   (takes a few minutes)

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import task_a
import task_b
import task_c

OUT = "outputs/feature_selection"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")
SELECTION_YEARS = range(1995, 2004)

df = pd.read_csv("outputs/team_seasons.csv")
CANDIDATES = [c for c in df.columns if c.startswith(("last_", "players_", "goalies_", "coach_"))
              and c not in ("coach_id", "coach_changed_in_season")]


# For each task: how to score one season, which direction is better, the target,
# and how the baseline does in that season.
def score_a(year):
    test, ridge, baseline = task_a.predict(year, task_a.WINDOW)
    return task_a.rank_error(test, ridge, "given"), task_a.rank_error(test, baseline, "given")


def score_b(year):
    # AUC to choose the features (F1 of the top k teams is too noisy with ~5 changes a season)
    test, prob, pred, k = task_b.predict(year, task_b.WINDOW)
    s = task_b.scores(test, prob, pred, k)
    return s["forest_auc"], 0.5


def score_b_f1(year):
    s = task_b.scores(*task_b.predict(year, task_b.WINDOW))
    return s["forest_f1"], s["random_k_f1"]


def score_c(year):
    test, ridge, baseline = task_c.predict(year, task_c.WINDOW)
    return task_c.mae(test, ridge), task_c.mae(test, baseline)


TASKS = {
    "a": {"module": task_a, "score": score_a, "lower_is_better": True, "target": "points_%",
          "metric": "rank error"},
    "b": {"module": task_b, "score": score_b, "lower_is_better": False, "target": "coach_changed_in_season",
          "metric": "AUC"},
    "c": {"module": task_c, "score": score_c, "lower_is_better": True, "target": "penalty_minutes_per_game",
          "metric": "MAE"},
}


def mean_score(task, features, years):
    task["module"].FEATURES = list(features)
    return np.mean([task["score"](y)[0] for y in years])


def filter_features(task):
    rows = df[df["year"].isin(SELECTION_YEARS) & df[task["target"]].notna()]
    target = rows[task["target"]].astype(float)
    if task["target"] == "penalty_minutes_per_game":  # task c predicts the change from last season
        target = target - rows["last_season_penalty_minutes_per_game"]
    rho = rows[CANDIDATES].corrwith(target, method="spearman").abs().dropna()
    keep = rho[rho >= 0.05].sort_values(ascending=False)
    corr = rows[keep.index].corr(method="spearman").abs()
    chosen = []
    for f in keep.index:  # from the most related to the target: skip it if it repeats one we kept
        if all(corr.loc[f, g] <= 0.8 for g in chosen):
            chosen.append(f)
    return chosen, rho


def forward_selection(task, name):
    chosen, history = [], []
    best = np.inf if task["lower_is_better"] else -np.inf
    while True:
        tries = {f: mean_score(task, chosen + [f], SELECTION_YEARS) for f in CANDIDATES if f not in chosen}
        f, s = (min if task["lower_is_better"] else max)(tries.items(), key=lambda kv: kv[1])
        improved = s < best if task["lower_is_better"] else s > best
        if not improved:
            break
        chosen.append(f)
        best = s
        history.append((len(chosen), f, s))
        print(f"  task {name}: + {f:45s} {task['metric']} {s:.4f}")
    return chosen, pd.DataFrame(history, columns=["features", "added", "score"])


summary_rows = []
chosen_sets = {}
for name, task in TASKS.items():
    print(f"\n===== task ({name}) =====")
    original = list(task["module"].FEATURES)

    filt, rho = filter_features(task)
    print(f"Filter: {len(filt)} of {len(CANDIDATES)} features kept")

    fwd, history = forward_selection(task, name)
    print(f"Forward selection: {len(fwd)} features")

    # plots: filter ranking and forward selection curve
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    rho.sort_values().plot.barh(ax=axes[0], color=["#C44E52" if f in filt else "#8C8C8C" for f in
                                                   rho.sort_values().index])
    axes[0].axvline(0.05, color="k", ls="--", lw=1)
    axes[0].set(title=f"({name}) filter: |Spearman| with the target, 1995-2003 (red = kept)",
                xlabel="|Spearman|")
    axes[1].plot(history["features"], history["score"], marker="o")
    for _, r in history.iterrows():
        axes[1].annotate(r["added"], (r["features"], r["score"]), fontsize=7, rotation=20)
    axes[1].set(title=f"({name}) forward selection on 1995-2003", xlabel="number of features",
                ylabel=task["metric"])
    plt.tight_layout()
    plt.savefig(f"{OUT}/task_{name}.png", dpi=110)
    plt.close()

    # compare the three sets on the validation seasons, with the metric of the task script
    score = score_b_f1 if name == "b" else task["score"]
    for set_name, feats in [("all (current)", original), ("filter", filt), ("forward", fwd)]:
        task["module"].FEATURES = list(feats)
        per_season = [score(y) for y in task_a.study_years]
        m = np.array([s[0] for s in per_season])
        base = np.array([s[1] for s in per_season])
        better = (m < base).sum() if task["lower_is_better"] else (m > base).sum()
        summary_rows.append({"task": name, "features": set_name, "n": len(feats), "mean": m.mean(),
                             "variance": m.var(ddof=1), "seasons_better_than_baseline": better})
    task["module"].FEATURES = original
    chosen_sets[name] = {"filter": filt, "forward": fwd}

summary = pd.DataFrame(summary_rows)
pd.set_option("display.width", 200)
print("\nValidation seasons 2005-2009 (a: rank error, lower is better; b: F1, higher is better;"
      " c: MAE, lower is better)")
print(summary.round(4).to_string(index=False))
summary.round(4).to_csv(f"{OUT}/summary.csv", index=False)
pd.DataFrame([(t, s, ", ".join(f)) for t, sets in chosen_sets.items() for s, f in sets.items()],
             columns=["task", "method", "features"]).to_csv(f"{OUT}/chosen_features.csv", index=False)
print(f"\nSaved {OUT}/summary.csv, {OUT}/chosen_features.csv and the plots")
