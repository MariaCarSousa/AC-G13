# Exploratory data analysis (EDA) for the three tasks, in the style of the lab
# notebooks (teacher feedback): one question per plot, from one variable, to two,
# to many. Data quality (missing values, outliers, ...) is in data_quality.py.
# We only look at seasons with results (up to 2009): the test seasons stay unseen.
# Run make_db.py and prepare.py first, then: python eda.py
# Plots are saved in outputs/eda/ and the conclusions are printed.

import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates

OUT = "outputs/eda"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
CHANGE_PALETTE = {0: "#4C72B0", 1: "#C44E52"}

conn = sqlite3.connect("outputs/hockey.db")
teams = pd.read_sql("SELECT * FROM Teams", conn)
players = pd.read_sql("SELECT * FROM Master WHERE playerID IS NOT NULL", conn)
scoring = pd.read_sql("SELECT * FROM Scoring", conn)
conn.close()

df = pd.read_csv("outputs/team_seasons.csv")
df = df[df["points_%"].notna()].copy()  # only seasons with results
df["decade"] = df["year"] // 10 * 10
df["era"] = pd.cut(df["year"], [1900, 1942, 1967, 1979, 1994, 2009],
                   labels=["1917-42", "1943-67 (Original Six)", "1968-79 (expansion)", "1980-94", "1995-2009"])
df["coach_changed_in_season"] = df["coach_changed_in_season"].astype("Int64")
# the seasons we train and validate on
recent = df[(df["league"] == "NHL") & (df["year"] >= 1995)].copy()


def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=120)
    plt.close()
    print(f"  saved {OUT}/{name}.png")


def title(text):
    print("\n" + "=" * 70 + "\n" + text + "\n" + "=" * 70)


# ---------- 1. meet the data ----------

title("1. Meet the data")
print(pd.DataFrame([
    ("Teams", "one team in one season", len(teams)),
    ("Master", "one person (player or coach)", len(players)),
    ("Scoring", "one player in one team in one season (stint)", len(scoring)),
    ("team_seasons.csv", "one team in one season, with targets and features", len(df)),
], columns=["table", "one row is...", "rows"]).to_string(index=False))
print("\nSeasons:", df["year"].min(), "to", df["year"].max(), "| missing seasons after 1917:",
      sorted(set(range(1917, 2010)) - set(df["year"])))

# how many teams each league had per decade
counts = pd.crosstab(df["decade"], df["league"])
plt.figure(figsize=(8, 5))
sns.heatmap(counts, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Team-seasons per league and decade")
save("01_teams_per_league_decade")
print("-> The NHL is almost all the data; other leagues (WHA, PCHA, ...) are short and old.")

print("\nSummary of the targets (all seasons):")
print(df[["points_%", "penalty_minutes_per_game", "coach_changed_in_season"]].astype(float).describe().round(3))


# ---------- 2. one variable ----------

title("2. One variable: what do the targets look like?")

# targets of the three tasks, in the recent seasons
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(recent["points_%"], bins=25, ax=axes[0])
axes[0].axvline(0.5, color="k", ls="--")
axes[0].set_title("(a) points % (1995-2009)")
sns.histplot(recent["penalty_minutes"], bins=25, ax=axes[1], color="#55A868")
axes[1].set_title("(c) penalty minutes per season (1995-2009)")
changes = recent["coach_changed_in_season"].value_counts().sort_index()
axes[2].bar(["no change", "changed coach"], changes.values, color=[CHANGE_PALETTE[0], CHANGE_PALETTE[1]])
for i, v in enumerate(changes.values):
    axes[2].annotate(f"{v} ({v / changes.sum():.0%})", (i, v), ha="center", va="bottom")
axes[2].set_title("(b) coach changed during the season (1995-2009)")
save("02_targets")
print(f"points %: mean {recent['points_%'].mean():.3f}, median {recent['points_%'].median():.3f}"
      " -> above .500 because since 1999 a team losing in overtime (or shootout) also gets 1 point")
print(f"coach changes: {changes[1]} of {changes.sum()} ({changes[1] / changes.sum():.1%}),"
      f" imbalance ratio {changes[0] / changes[1]:.1f} to 1 -> F1/AUC, not accuracy")
print(f"always predicting 'no change' would have accuracy {changes[0] / changes.sum():.1%} and find no change at all")

# penalty minutes per game: three views of the same column (like the delay plots of Lab 2)
pim = df["penalty_minutes_per_game"].dropna()
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].hist(pim, bins=50)
axes[0].set(title="all seasons", xlabel="penalty minutes per game")
axes[1].hist(pim, bins=50, log=True)
axes[1].set(title="all seasons, log-scaled counts", xlabel="penalty minutes per game")
axes[2].hist(recent["penalty_minutes_per_game"], bins=30, color="#55A868")
axes[2].axvline(recent["penalty_minutes_per_game"].median(), color="k", ls="--", label="median")
axes[2].axvline(recent["penalty_minutes_per_game"].mean(), color="#C44E52", ls="--", label="mean")
axes[2].set(title="1995-2009 only", xlabel="penalty minutes per game")
axes[2].legend()
save("03_penalty_minutes_three_views")
print(f"-> Right-skewed with a long tail of high values (mostly the 1980s); std all seasons"
      f" {pim.std():.1f}, 1995-2009 only {recent['penalty_minutes_per_game'].std():.1f}: inside one era it is narrower.")

# ECDF by era: no bins to choose, easy to compare groups (Lab 2)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.ecdfplot(data=df[df["league"] == "NHL"], x="points_%", hue="era", ax=axes[0])
axes[0].set_title("ECDF of points % by era (NHL)")
sns.ecdfplot(data=df[df["league"] == "NHL"], x="penalty_minutes_per_game", hue="era", ax=axes[1])
axes[1].set_title("ECDF of penalty minutes per game by era (NHL)")
save("04_ecdf_by_era")
print("\nSpread (std) of points % per era (NHL):")
print(df[df["league"] == "NHL"].groupby("era", observed=True)["points_%"].std().round(3).to_string())
print("-> The league became more balanced: modern seasons have a smaller spread of points %.")

# players
pl = players.copy()
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
sns.histplot(pl["height"].dropna(), discrete=True, ax=axes[0])
axes[0].set(title="height (inches)")
sns.histplot(pl["weight"].dropna(), bins=30, ax=axes[1])
axes[1].set(title="weight (pounds)")
first = scoring.groupby("playerID")["year"].min().rename("first_season")
pl = pl.merge(first, left_on="playerID", right_index=True, how="left")
pl["age_at_first_season"] = pl["first_season"] - pl["birthYear"]
sns.histplot(pl["age_at_first_season"].dropna(), discrete=True, ax=axes[2])
axes[2].set(title="age in the first season", xlim=(14, 40))
sns.countplot(data=pl, x="pos", order=pl["pos"].value_counts().index, ax=axes[3], color="#8C8C8C")
axes[3].set(title="position")
save("05_players")
print(f"\nPlayers: median height {pl['height'].median():.0f} in, weight {pl['weight'].median():.0f} lb,"
      f" debut at {pl['age_at_first_season'].median():.0f} years")


# ---------- 3. two variables ----------

title("3. Two variables: how does each target change with the features?")

# (a) points % this season vs last season: the baseline idea
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(recent["last_season_points_%"], recent["points_%"], s=10, alpha=0.4)
axes[0].plot([0.2, 0.8], [0.2, 0.8], "k--", lw=1, label="same as last season")
axes[0].set(xlabel="points % last season", ylabel="points % this season",
            title="(a) this season vs last season (1995-2009)")
axes[0].legend()
nhl_all = df[(df["league"] == "NHL")]
hb = axes[1].hexbin(nhl_all["last_season_points_%"], nhl_all["points_%"], gridsize=30, mincnt=1, cmap="Blues")
plt.colorbar(hb, ax=axes[1], label="team-seasons")
axes[1].set(xlabel="points % last season", ylabel="points % this season", title="all NHL seasons (hexbin)")
save("06_points_last_vs_this")
r = recent[["last_season_points_%", "points_%"]].corr().iloc[0, 1]
slope = np.polyfit(recent["last_season_points_%"].dropna(),
                   recent.loc[recent["last_season_points_%"].notna(), "points_%"], 1)[0]
print(f"(a) correlation last vs this season: {r:.2f}, slope {slope:.2f}")
print("-> Strong link (the baseline is good), but the slope is below 1: good teams get a bit worse and")
print("   bad teams a bit better (regression to the mean). Ridge can learn that, the baseline cannot.")

# (a) the most related features, one panel each (small multiples)
a_feats = ["last_3_seasons_points_%", "last_season_goals_against_per_game", "top3_players_points_per_game",
           "players_last_season_points", "goalies_last_season_save_%", "coach_win_%"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, f in zip(axes.flat, a_feats):
    sns.regplot(data=recent, x=f, y="points_%", ax=ax, scatter_kws={"s": 8, "alpha": 0.4},
                line_kws={"color": "#C44E52"})
    ax.set_title(f"rho = {recent[[f, 'points_%']].corr('spearman').iloc[0, 1]:.2f}")
fig.suptitle("(a) points % against some features (1995-2009), rho = Spearman")
save("07_points_vs_features")

# (a) box plots against categories
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
sns.boxplot(data=recent, x="last_season_made_playoffs", y="points_%", ax=axes[0])
axes[0].set(xlabel="made the playoffs last season (0/1)")
sns.boxplot(data=recent, x="coach_new_this_season", y="points_%", ax=axes[1])
axes[1].set(xlabel="new coach before the season (0/1)")
sns.boxplot(data=recent, x="players_with_20_goals", y="points_%", ax=axes[2])
axes[2].set(xlabel="players who scored 20+ goals last season")
fig.suptitle("(a) points % by category (1995-2009)")
save("08_points_by_category")
new = recent.groupby("coach_new_this_season")[["last_season_points_%", "points_%"]].median().round(3)
new.columns = ["median points % last season", "median points % this season"]
print(new.to_string())
print("-> Teams that start with a new coach were worse last season, but this season they are about")
print("   as good as the others: a new coach alone says little about the coming season.")

# (b) coach change rate per feature (like the survival rate per feature in the Titanic lab)
b = recent.copy()
b["changed"] = b["coach_changed_in_season"].astype(float)
b["points_%_last_season_5_groups"] = pd.qcut(b["last_season_points_%"], 5,
                                             labels=["worst", "2", "3", "4", "best"])
b["coach_seasons_with_team_groups"] = pd.cut(b["coach_seasons_with_team"], [-1, 0, 1, 3, 100],
                                             labels=["0", "1", "2-3", "4+"])
fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
for ax, (x, label) in zip(axes, [("points_%_last_season_5_groups", "points % last season (5 groups)"),
                                 ("last_season_made_playoffs", "made playoffs last season"),
                                 ("coach_new_this_season", "new coach before the season"),
                                 ("coach_seasons_with_team_groups", "coach's seasons with the team")]):
    sns.barplot(data=b, x=x, y="changed", ax=ax, color="#C44E52", errorbar=("ci", 95))
    ax.axhline(b["changed"].mean(), color="k", ls="--", lw=1)
    ax.set(xlabel=label, ylabel="share of teams changing coach")
fig.suptitle("(b) coach change rate by feature, with 95% confidence intervals (dashed = overall rate)")
save("09_coach_change_rate_by_feature")
rate = b.groupby("points_%_last_season_5_groups", observed=True)["changed"].mean()
print("\n(b) share of teams changing coach, by points % last season (5 groups):")
print(rate.round(3).to_string())
print("-> Weaker teams change coach a bit more often, but the confidence intervals overlap: no single")
print("   feature separates the classes well, which is why task (b) is hard (F1 around 0.25).")

# (b) two features at once as a heatmap (like sex x class in the Titanic lab)
pivot = b.pivot_table(index="points_%_last_season_5_groups", columns="coach_new_this_season",
                      values="changed", aggfunc="mean", observed=True)
cnt = b.pivot_table(index="points_%_last_season_5_groups", columns="coach_new_this_season",
                    values="changed", aggfunc="count", observed=True)
plt.figure(figsize=(6, 5))
sns.heatmap(pivot, annot=pivot.map(lambda v: f"{v:.0%}") + "\n(n=" + cnt.astype(str) + ")", fmt="",
            cmap="Reds", cbar_kws={"label": "share changing coach"})
plt.xlabel("new coach before the season (0/1)")
plt.ylabel("points % last season (5 groups)")
plt.title("(b) coach change rate by two features")
save("10_coach_change_heatmap")
print("\n(b) coach change rate by points % last season (rows) and new coach before the season (columns):")
print(pivot.round(2).to_string())
print("-> Together the two features say more than each alone: a weak team that KEPT its coach changes")
print("   him much more often than a weak team that just hired one (the new coach gets time).")
print("   Some cells have few teams (n in the plot), so this is a hint, not a proof. Tree-based")
print("   models (the random forest of task b) can learn this kind of interaction; Ridge cannot.")

# (b) 100% stacked bars per decade: the balance of the target over time
nhl_b = df[(df["league"] == "NHL") & df["coach_changed_in_season"].notna()]
stack = pd.crosstab(nhl_b["decade"], nhl_b["coach_changed_in_season"], normalize="index")
stack.plot.bar(stacked=True, color=[CHANGE_PALETTE[0], CHANGE_PALETTE[1]], figsize=(9, 4), rot=0)
plt.legend(["no change", "changed coach"], loc="lower right")
plt.ylabel("share of teams")
plt.title("(b) coach changes during the season per decade (NHL)")
save("11_coach_change_per_decade")
print("\n(b) share of teams changing coach per decade:")
print(stack[1].round(3).to_string())

# (c) penalty minutes this season vs last season and vs the new players
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(recent["last_season_penalty_minutes_per_game"], recent["penalty_minutes_per_game"], s=10, alpha=0.4)
axes[0].plot([8, 30], [8, 30], "k--", lw=1, label="same as last season")
axes[0].set(xlabel="last season", ylabel="this season", title="(c) penalty minutes per game")
axes[0].legend()
axes[1].scatter(recent["players_last_season_penalty_minutes_per_game"], recent["penalty_minutes_per_game"],
                s=10, alpha=0.4, color="#55A868")
axes[1].set(xlabel="this season's players, their penalty minutes per game last season",
            ylabel="team penalty minutes per game", title="(c) the roster's penalty history")
change = recent["penalty_minutes_per_game"] - recent["last_season_penalty_minutes_per_game"]
sns.boxplot(x=recent["year"], y=change, ax=axes[2], color="#8C8C8C")
axes[2].axhline(0, color="k", lw=1)
axes[2].set(xlabel="season", ylabel="change from last season", title="(c) change per game, per season")
axes[2].tick_params(axis="x", rotation=90)
save("12_penalty_minutes")
print(f"\n(c) correlation last vs this season: "
      f"{recent[['last_season_penalty_minutes_per_game', 'penalty_minutes_per_game']].corr().iloc[0, 1]:.2f}")
print("(c) median change from last season, per season:")
print(change.groupby(recent["year"]).median().round(2).to_string())
med = change.groupby(recent["year"]).median()
print(f"-> The median change is negative in {(med < 0).sum()} of {len(med)} seasons (overall {change.median():.2f}"
      " per game): penalty minutes keep going down, which is why task_c.py predicts the change.")


# ---------- 4. time ----------

title("4. Time: how does the league change?")
nhl_t = teams[teams["lgID"].isin(["NHL", "WHA"]) & teams["G"].notna()].copy()
nhl_t["goals_per_game"] = nhl_t["GF"] / nhl_t["G"]
nhl_t["penalty_minutes_per_game"] = nhl_t["PIM"] / nhl_t["G"]
nhl_t["power_play_%"] = nhl_t["PPG"] / nhl_t["PPC"]
yearly = nhl_t.groupby(["lgID", "year"])[["goals_per_game", "penalty_minutes_per_game", "power_play_%"]] \
    .mean().reset_index().melt(id_vars=["lgID", "year"], var_name="stat")
g = sns.relplot(data=yearly, x="year", y="value", hue="lgID", row="stat", kind="line",
                height=2.8, aspect=4, facet_kws={"sharey": False})
g.set_titles("{row_name}")
g.fig.suptitle("League averages per season (small multiples)", y=1.01)
plt.savefig(f"{OUT}/13_league_over_time.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"  saved {OUT}/13_league_over_time.png")
print("-> Goals per game peak in the early 1980s and fall after; penalty minutes peak in the late 1980s.")
print("   The 'normal' values change with the era, so we compare teams inside their season.")


# ---------- 5. many variables ----------

title("5. Many variables")

# which features are related to each target? (filter view for feature selection)
feats = [c for c in recent.columns if c.startswith(("last_", "players_", "goalies_", "coach_"))
         and c not in ("coach_id", "coach_changed_in_season")]
rel = pd.DataFrame({
    "(a) points %": recent[feats].corrwith(recent["points_%"], method="spearman"),
    "(b) coach change": recent[feats].corrwith(recent["coach_changed_in_season"].astype(float), method="spearman"),
    "(c) penalty minutes": recent[feats].corrwith(recent["penalty_minutes_per_game"], method="spearman"),
})
plt.figure(figsize=(9, 11))
sns.heatmap(rel, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-0.8, vmax=0.8, cbar=False)
plt.title("Spearman correlation of each feature with each target (1995-2009)")
save("14_feature_target_correlation")
for c in rel.columns:
    weak = rel.index[rel[c].abs() < 0.05].tolist()
    print(f"{c}: strongest {rel[c].abs().idxmax()} ({rel[c][rel[c].abs().idxmax()]:.2f});"
          f" almost unrelated (|rho| < 0.05): {weak}")

# scatter plot matrix of a few features, coloured by the task (b) target
pair_cols = ["last_season_points_%", "last_season_goals_against_per_game", "coach_win_%",
             "players_avg_age", "coach_changed_in_season"]
pp = recent[pair_cols].dropna().astype({"coach_changed_in_season": int})
g = sns.pairplot(pp, hue="coach_changed_in_season", palette=CHANGE_PALETTE, corner=True,
                 plot_kws={"s": 12, "alpha": 0.5}, diag_kind="kde")
g.fig.suptitle("(b) scatter plot matrix coloured by coach change (1995-2009)", y=1.02)
plt.savefig(f"{OUT}/15_pairplot_coach_change.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"  saved {OUT}/15_pairplot_coach_change.png")

# parallel coordinates (ch03): each line is a team-season, features standardized
pc_cols = ["last_season_points_%", "last_season_goals_for_per_game", "last_season_goals_against_per_game",
           "top3_players_points_per_game", "goalies_last_season_save_%", "coach_win_%"]
pc = recent[pc_cols + ["coach_changed_in_season"]].dropna()
pc[pc_cols] = (pc[pc_cols] - pc[pc_cols].mean()) / pc[pc_cols].std()
pc["coach_changed_in_season"] = pc["coach_changed_in_season"].map({0: "no change", 1: "changed coach"})
plt.figure(figsize=(14, 5))
parallel_coordinates(pc.sort_values("coach_changed_in_season", ascending=False), "coach_changed_in_season",
                     color=["#4C72B0", "#C44E52"], alpha=0.25)
plt.xticks(rotation=15)
plt.ylabel("standardized value")
plt.title("(b) parallel coordinates: teams that changed coach (red) vs not (blue), 1995-2009")
save("16_parallel_coordinates")

# three variables in one plot: penalty minutes vs points %, coloured by era (task c)
nhl_c = df[df["league"] == "NHL"]
plt.figure(figsize=(10, 6))
sns.scatterplot(data=nhl_c, x="points_%", y="penalty_minutes_per_game", hue="era", s=15, alpha=0.6,
                palette="viridis")
plt.title("(c) penalty minutes vs points %, coloured by era (NHL)")
save("17_penalty_vs_points_by_era")
by_era = nhl_c.groupby("era", observed=True)[["points_%", "penalty_minutes_per_game"]] \
    .corr(method="spearman").xs("points_%", level=1)["penalty_minutes_per_game"]
print("\n(c) Spearman points % vs penalty minutes, per era:")
print(by_era.round(2).to_string())
print("-> The era explains penalty minutes much more than team quality. In old eras good teams had")
print("   slightly more penalty minutes; since 1995 good teams have fewer.")

print(f"\nAll plots saved in {OUT}/")
