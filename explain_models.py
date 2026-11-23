# Explaining the models and checking overfitting.
#   1. White box: what did the final models learn? (the models trained on the seasons
#      before 2010, the ones that predict 2010-2013)
#      - (a), (c) Ridge coefficients: the features are standardized, so a coefficient is
#        how much the prediction moves when the feature goes up by one standard deviation
#      - (b) random forest feature importance (embedded method, ch04 slides)
#      - a decision tree of depth 3 for each task, easy to read (Lab 1, Chap10 slides)
#   2. Overfitting: error on the training seasons vs error on the validation seasons
#      (2005-2009), for each training window and each value of the parameter.
#      A big gap (very good on training, bad on validation) = overfitting.
# Run prepare.py first, then: python explain_models.py

import os

import matplotlib
matplotlib.use("Agg")  # only save files (no windows); avoids a crash with the forest's threads
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

import task_a
import task_b
import task_c

OUT = "outputs/explain"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")
FIRST_TEST = 2010  # the final models are trained on the seasons before it


def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=120)
    plt.close()
    print(f"  saved {OUT}/{name}.png")


# targets as each task trains them
def target_a(rows):
    return rows["points_%"]


def target_b(rows):
    return rows["coach_changed_in_season"].astype(int)


def target_c(rows):
    return rows["penalty_minutes_per_game"] - rows["last_season_penalty_minutes_per_game"]


def train_rows(task, target, year, window):
    rows = task.training_seasons(year, window)
    if task is task_c:
        rows = rows.dropna(subset=["last_season_penalty_minutes_per_game"])
    return rows, target(rows)


# ---------- 1. white box ----------

print("=" * 70 + "\n1. What did the final models learn?\n" + "=" * 70)

for name, task, target, label in [("a", task_a, target_a, "points %"),
                                  ("c", task_c, target_c, "change in penalty minutes per game")]:
    rows, y = train_rows(task, target, FIRST_TEST, task.WINDOW)
    model = task.make_model(task.ALPHA).fit(rows[task.FEATURES], y)
    coef = pd.Series(model[-1].coef_, index=task.FEATURES).sort_values()
    plt.figure(figsize=(9, 0.35 * len(coef) + 1.5))
    coef.plot.barh(color=["#C44E52" if v < 0 else "#4C72B0" for v in coef])
    plt.axvline(0, color="k", lw=1)
    plt.xlabel(f"effect on {label} of +1 standard deviation of the feature")
    plt.title(f"({name}) Ridge coefficients, alpha {task.ALPHA}, trained on "
              f"{rows['year'].min()}-{rows['year'].max()}")
    save(f"task_{name}_ridge_coefficients")
    top = coef.abs().sort_values(ascending=False).head(5).index
    print(f"\n({name}) biggest Ridge coefficients (per +1 std of the feature):")
    print(coef[top].round(4).to_string())

rows, y = train_rows(task_b, target_b, FIRST_TEST, task_b.WINDOW)
forest = task_b.make_model(task_b.LEAF_SIZE).fit(rows[task_b.FEATURES], y)
imp = pd.Series(forest[-1].feature_importances_, index=task_b.FEATURES).sort_values()
plt.figure(figsize=(9, 0.35 * len(imp) + 1.5))
imp.plot.barh(color="#C44E52")
plt.xlabel("importance (mean decrease in impurity, sums to 1)")
plt.title(f"(b) random forest feature importance, trained on {rows['year'].min()}-{rows['year'].max()}")
save("task_b_forest_importance")
print("\n(b) most important features of the random forest:")
print(imp.sort_values(ascending=False).head(5).round(3).to_string())

# small decision trees: not used to predict, only to read the main rules of each task
for name, task, target, tree, label in [
        ("a", task_a, target_a, DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=0), "points %"),
        ("b", task_b, target_b, DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, class_weight="balanced",
                                                       random_state=0), "coach change"),
        ("c", task_c, target_c, DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=0),
         "change in penalty minutes per game")]:
    rows, y = train_rows(task, target, FIRST_TEST, task.WINDOW)
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), tree)
    model.fit(rows[task.FEATURES], y)
    plt.figure(figsize=(20, 8))
    plot_tree(model[-1], feature_names=task.FEATURES, filled=True, rounded=True, fontsize=8, precision=2,
              class_names=["no change", "change"] if name == "b" else None)
    plt.title(f"({name}) decision tree of depth 3 for {label}, trained on "
              f"{rows['year'].min()}-{rows['year'].max()} (only to read the rules)")
    save(f"task_{name}_tree")
    root = task.FEATURES[model[-1].tree_.feature[0]]
    print(f"({name}) first split of the tree: {root} <= {model[-1].tree_.threshold[0]:.3f}")


# ---------- 2. overfitting ----------

print("\n" + "=" * 70 + "\n2. Overfitting: training vs validation error\n" + "=" * 70)


def errors(name, task, param, window):
    # mean error on the training seasons and on the validation season, over 2005-2009
    train_err, val_err = [], []
    for year in task.study_years:
        if name == "b":
            rows, y = train_rows(task, target_b, year, window)
            model = task.make_model(param).fit(rows[task.FEATURES], y)
            train_err.append(roc_auc_score(y, model.predict_proba(rows[task.FEATURES])[:, 1]))
            test, prob, pred, k = task.predict(year, window, param)
            val_err.append(task.scores(test, prob, pred, k)["forest_auc"])
            continue
        target = target_a if name == "a" else target_c
        rows, y = train_rows(task, target, year, window)
        model = task.make_model(param).fit(rows[task.FEATURES], y)
        fitted = pd.Series(model.predict(rows[task.FEATURES]), index=rows.index)
        if name == "a":
            train_err.append(np.mean([task.rank_error(s, fitted[s.index], "given") for _, s in rows.groupby("year")]))
        else:
            train_err.append(task.mae(rows, (rows["last_season_penalty_minutes_per_game"] + fitted) * rows["games"]))
        test, model_pred, baseline = task.predict(year, window, param)
        val_err.append(task.rank_error(test, model_pred, "given") if name == "a" else task.mae(test, model_pred))
    return np.mean(train_err), np.mean(val_err)


setups = [("a", task_a, "alpha", task_a.ALPHAS, task_a.ALPHA, "rank error (lower is better)"),
          ("b", task_b, "min_samples_leaf", task_b.LEAF_SIZES, task_b.LEAF_SIZE, "AUC (higher is better)"),
          ("c", task_c, "alpha", task_c.ALPHAS, task_c.ALPHA, "MAE (lower is better)")]
rows_out = []
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for col, (name, task, pname, params, chosen, metric) in enumerate(setups):
    by_param = pd.DataFrame([(p, *errors(name, task, p, task.WINDOW)) for p in params],
                            columns=[pname, "training", "validation"]).set_index(pname)
    by_window = pd.DataFrame([(w, *errors(name, task, chosen, w)) for w in sorted(task.WINDOWS)],
                             columns=["window", "training", "validation"]).set_index("window")
    by_param.plot(ax=axes[0, col], marker="o", logx=(name != "b"))
    axes[0, col].axvline(chosen, color="k", ls=":", lw=1)
    axes[0, col].set(title=f"({name}) by {pname}, window {task.WINDOW}", ylabel=metric)
    by_window.plot(ax=axes[1, col], marker="o")
    axes[1, col].axvline(task.WINDOW, color="k", ls=":", lw=1)
    axes[1, col].set(title=f"({name}) by training window, {pname} {chosen}", xlabel="training seasons",
                     ylabel=metric)
    print(f"\n({name}) by {pname} (window {task.WINDOW}):")
    print(by_param.round(4).to_string())
    print(f"({name}) by training window ({pname} {chosen}):")
    print(by_window.round(4).to_string())
    for p, r in by_param.iterrows():
        rows_out.append({"task": name, "parameter": f"{pname}={p}", "window": task.WINDOW, **r.round(4).to_dict()})
    for w, r in by_window.iterrows():
        rows_out.append({"task": name, "parameter": f"{pname}={chosen}", "window": w, **r.round(4).to_dict()})
fig.suptitle("Training vs validation (2005-2009). Dotted line = chosen value. A big gap = overfitting")
save("overfitting")
pd.DataFrame(rows_out).to_csv(f"{OUT}/overfitting.csv", index=False)
print(f"\nSaved {OUT}/overfitting.csv")
