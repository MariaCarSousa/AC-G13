# Data quality of the hockey data (teacher feedback: plots to find missing values,
# outliers and irrelevant or badly filled columns). Follows the data quality part of
# the course (ch04 slides, Lab 2): keys, missing, redundant, inconsistent, noisy data
# and outliers. Each part saves its plots in outputs/data_quality/ and prints what
# we learn from them.
# Run make_db.py and prepare.py first, then: python data_quality.py

import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest

OUT = "outputs/data_quality"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

conn = sqlite3.connect("outputs/hockey.db")
teams = pd.read_sql("SELECT * FROM Teams", conn)
master = pd.read_sql("SELECT * FROM Master", conn)
scoring = pd.read_sql("SELECT * FROM Scoring", conn)
goalies = pd.read_sql("SELECT * FROM Goalies", conn)
coaches = pd.read_sql("SELECT * FROM Coaches", conn)
conn.close()
seasons = pd.read_csv("outputs/team_seasons.csv")
seasons = seasons[seasons["points_%"].notna()]  # only seasons with results

teams["decade"] = teams["year"] // 10 * 10
teams["pts_pct"] = teams["Pts"] / (2 * teams["G"])
teams["gf_pg"] = teams["GF"] / teams["G"]
teams["ga_pg"] = teams["GA"] / teams["G"]
teams["pim_pg"] = teams["PIM"] / teams["G"]


def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=120)
    plt.close()
    print(f"  saved {OUT}/{name}.png")


def title(text):
    print("\n" + "=" * 70 + "\n" + text + "\n" + "=" * 70)


# ---------- 1. keys and orphans ----------
# A key must identify each row (no duplicates, no missing values). An orphan is a
# value that points to a row that does not exist in the other table: joins drop
# those rows without any warning.

title("1. Keys and orphans")


def is_key(df, cols):
    return bool(df[cols].notna().all().all() and not df.duplicated(cols).any())


keys = pd.DataFrame([
    ("Teams", "year, tmID", is_key(teams, ["year", "tmID"])),
    ("Master", "playerID (players only)", is_key(master[master["playerID"].notna()], ["playerID"])),
    ("Scoring", "playerID, year, stint", is_key(scoring, ["playerID", "year", "stint"])),
    ("Goalies", "playerID, year, stint", is_key(goalies, ["playerID", "year", "stint"])),
    ("Coaches", "coachID, year, tmID, stint", is_key(coaches, ["coachID", "year", "tmID", "stint"])),
], columns=["table", "columns", "is a key"])
print(keys.to_string(index=False))


def orphans(df, cols, other, other_cols):
    m = df[cols].merge(other[other_cols].drop_duplicates(), left_on=cols, right_on=other_cols,
                       how="left", indicator=True)
    return (m["_merge"] == "left_only").sum()


orph = pd.DataFrame([
    ("Scoring.playerID -> Master", orphans(scoring, ["playerID"], master, ["playerID"]), len(scoring)),
    ("Scoring.(year, tmID) -> Teams", orphans(scoring, ["year", "tmID"], teams, ["year", "tmID"]), len(scoring)),
    ("Goalies.(year, tmID) -> Teams", orphans(goalies, ["year", "tmID"], teams, ["year", "tmID"]), len(goalies)),
    ("Coaches.(year, tmID) -> Teams", orphans(coaches, ["year", "tmID"], teams, ["year", "tmID"]), len(coaches)),
], columns=["foreign key", "orphan rows", "rows"])
print("\n" + orph.to_string(index=False))
print("-> If all counts are 0, our joins in prepare.py do not lose rows.")


# ---------- 2. missing values ----------

title("2. Missing values")

# 2a. how much is missing in each column
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, df) in zip(axes, [("Teams", teams), ("Master (players and coaches)", master)]):
    miss = (100 * df.isna().mean()).loc[lambda s: s > 0].sort_values()
    miss.plot.barh(ax=ax, color="#4C72B0")
    ax.set(xlabel="% missing", title=f"{name}: missing values per column")
save("01_missing_per_column")

# 2b. where: missing share per decade (Teams). A column can be fine today and empty in old seasons.
cols = ["confID", "divID", "PIM", "PPG", "PPC", "PKG", "PKC", "SHF", "BenchMinor", "T", "OTL", "SoW"]
by_decade = teams.groupby("decade")[cols].apply(lambda d: d.isna().mean())
plt.figure(figsize=(11, 5))
sns.heatmap(by_decade.T, cmap="Blues", vmin=0, vmax=1, annot=True, fmt=".0%", cbar=False)
plt.title("Teams: share of missing values per decade")
plt.xlabel("decade")
save("02_missing_by_decade_teams")
first_full = {c: int(by_decade.index[by_decade[c] == 0].min()) if (by_decade[c] == 0).any() else None
              for c in ["PIM", "PPG", "BenchMinor", "confID", "divID"]}
print("First decade with no missing values:", first_full)
print("-> Power play / penalty kill stats start in the 1960s and conferences in the 1970s; both are")
print("   complete only from 1980. Ties (T) are missing after 2005 because ties were abolished.")
print("   These gaps are structural (the stat did not exist or was not recorded), not random.")
print("   Our training seasons (1995-2009) are complete, so these columns are usable for us.")

# 2c. patterns of missingness in Master (players only): columns go missing together
players = master[master["playerID"].notna()].copy()
pattern = (players[["height", "weight", "birthYear", "pos"]].isna()
           .value_counts().rename("players").reset_index()
           .replace({True: "missing", False: "."}))
print("\nPatterns of missing values (players):")
print(pattern.to_string(index=False))
print("-> height and weight are almost always missing together, so one cannot be")
print("   predicted from the other: the regression needs other columns (position, birth year).")

# 2d. why: are old players missing more often? (if yes, not missing completely at random)
players["first_season"] = scoring.groupby("playerID")["year"].min().reindex(players["playerID"]).values
players["first_decade"] = players["first_season"] // 10 * 10
why = players.groupby("first_decade")[["height", "weight", "birthYear"]].apply(lambda d: 100 * d.isna().mean())
why.plot(marker="o", figsize=(9, 4))
plt.ylabel("% missing")
plt.xlabel("decade of the player's first season")
plt.title("Players: missing values by era")
save("03_missing_by_era_players")
print("-> Missing height/weight depends on the era (old players), so it is not missing completely")
print("   at random. Filling with the global median would make old players look like modern ones;")
print("   a median per era and position, or a regression on them, is better.")

# 2e. how to fill height and weight? (Lab 2: hide values we know, fill them, measure the error)
# Height and weight are missing together, so the teacher's idea (predict one from the other)
# only works for the few players missing just one. For the others we use what we know:
# the era (first season, known for all of them) and the position (known for half).
players["pos_group"] = players["pos"].str[0].map({"D": "defense", "G": "goalie"}).fillna("forward")
players.loc[players["pos"].isna(), "pos_group"] = "unknown"
players["first_decade"] = players["first_decade"].fillna(players["first_decade"].median())


def fill_size(known, missing, col, method):
    # fills `col` of the `missing` players using only the `known` players
    if method == "global median":
        return pd.Series(known[col].median(), index=missing.index)
    if method == "median per decade":
        by = known.groupby("first_decade")[col].median()
        return missing["first_decade"].map(by).fillna(known[col].median())
    if method == "median per decade and position":
        by = known.groupby(["first_decade", "pos_group"])[col].median()
        keys = list(zip(missing["first_decade"], missing["pos_group"]))
        return pd.Series([by.get(k, np.nan) for k in keys], index=missing.index) \
            .fillna(fill_size(known, missing, col, "median per decade"))
    # linear regression on the first season and the position (one-hot)
    from sklearn.linear_model import LinearRegression
    x_known = pd.get_dummies(known[["first_season", "pos_group"]], columns=["pos_group"], dtype=float)
    x_miss = pd.get_dummies(missing[["first_season", "pos_group"]], columns=["pos_group"], dtype=float) \
        .reindex(columns=x_known.columns, fill_value=0)
    model = LinearRegression().fit(x_known.fillna(x_known.median()), known[col])
    return pd.Series(model.predict(x_miss.fillna(x_known.median())), index=missing.index)


methods = ["global median", "median per decade", "median per decade and position", "linear regression"]
full = players.dropna(subset=["height", "weight", "first_season"])
rng = np.random.default_rng(0)
errors = []
for rep in range(20):
    # hide the size of 20% of the players, mostly old ones like the real missing values
    p_hide = np.where(full["first_season"] < 1980, 0.4, 0.05)
    hide = rng.random(len(full)) < p_hide
    known, missing = full[~hide], full[hide]
    for col in ["height", "weight"]:
        for m in methods:
            err = fill_size(known, missing, col, m) - missing[col]
            errors.append({"column": col, "method": m, "MAE": err.abs().mean()})
imp = pd.DataFrame(errors).groupby(["column", "method"])["MAE"].agg(["mean", "std"]).round(2)
print("\nFilling hidden height (inches) and weight (pounds), 20 repetitions:")
print(imp.to_string())
imp_plot = pd.DataFrame(errors).pivot_table(index="method", columns="column", values="MAE").loc[methods]
imp_plot.plot.barh(subplots=True, layout=(1, 2), figsize=(13, 3.5), legend=False, sharex=False, color="#4C72B0")
plt.suptitle("Error when filling hidden heights and weights (lower is better)")
save("03b_fill_height_weight")
best = {col: imp.loc[col]["mean"].idxmin() for col in ["height", "weight"]}
print(f"-> Best method: height '{best['height']}', weight '{best['weight']}'. prepare.py uses it.")
only_one = (players["height"].isna() != players["weight"].isna()).sum()
print(f"   Players missing only one of the two: {only_one}, so predicting one from the other (corr 0.76)")
print("   cannot fill them. The group median is a simple regression on era and position.")

# 2f. missing values of the features we built (team_seasons), per season
feat = [c for c in seasons.columns if c.startswith(("last_", "players_", "goalies_", "coach_")) and c != "coach_id"]
by_year = seasons.groupby("year")[feat].apply(lambda d: d.isna().mean())
plt.figure(figsize=(14, 7))
sns.heatmap(by_year.T, cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "share missing"})
plt.title("Our features: share of teams with a missing value, per season")
save("04_missing_features_by_season")
recent = by_year.loc[by_year.index >= 1995].mean().sort_values(ascending=False)
print("\nFeatures most missing since 1995 (the seasons we train on):")
print((100 * recent.head(6)).round(1).to_string())


# ---------- 3. redundant data ----------

title("3. Redundant data")

t0 = teams.fillna({"T": 0, "OTL": 0})
print("Pts == 2*W + T + OTL in", (t0["Pts"] == 2 * t0["W"] + t0["T"] + t0["OTL"]).mean().round(3) * 100, "% of rows")
print("G   == W + L + T + OTL in", (t0["G"] == t0["W"] + t0["L"] + t0["T"] + t0["OTL"]).mean().round(3) * 100, "% of rows")
print("Scoring Pts == G + A in", (scoring["Pts"] == scoring["G"] + scoring["A"]).mean().round(3) * 100, "% of rows")
print("-> Points and games are derived columns. We keep points % (one number) and not W, L, T, Pts.")
print("Exact duplicated rows: Teams", teams.duplicated().sum(), "| Scoring", scoring.duplicated().sum(),
      "| Coaches", coaches.duplicated().sum())

# correlation between our features: Pearson (linear) and Spearman (ranks, robust to outliers)
num = seasons[seasons["year"] >= 1995][feat].select_dtypes("number")
fig, axes = plt.subplots(1, 2, figsize=(22, 10))
for ax, method in zip(axes, ["pearson", "spearman"]):
    corr = num.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt=".1f",
                annot_kws={"size": 6}, cbar=False, ax=ax)
    ax.set_title(f"{method.capitalize()} correlation between features (1995-2009)")
save("05_correlation_features")

corr = num.corr(method="spearman").abs()
pairs = (corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1)).stack()
         .sort_values(ascending=False).rename("spearman").reset_index())
pairs.columns = ["feature 1", "feature 2", "spearman"]
print("\nMost correlated feature pairs (|Spearman| > 0.8):")
print(pairs[pairs["spearman"] > 0.8].round(2).to_string(index=False))
print("-> These pairs carry almost the same information. Ridge handles it (it shares the weight),")
print("   but for the white-box explanation and feature selection we keep only one of each pair.")


# ---------- 4. inconsistent data ----------
# Rules that every row must follow. A value can be a valid number and still be wrong.

title("4. Inconsistent data (rules)")

player_team = scoring.merge(teams[["year", "tmID", "G"]].rename(columns={"G": "team_G"}), on=["year", "tmID"])
born = scoring.merge(players[["playerID", "birthYear"]], on="playerID")
rules = pd.DataFrame([
    ("Teams: W > G", (teams["W"] > teams["G"]).sum()),
    ("Teams: GF or GA < 0", ((teams["GF"] < 0) | (teams["GA"] < 0)).sum()),
    ("Teams: power play goals > power play chances", (teams["PPG"] > teams["PPC"]).sum()),
    ("Teams: penalty kill goals against > times short-handed", (teams["PKG"] > teams["PKC"]).sum()),
    ("Scoring: player games > team games", (player_team["GP"] > player_team["team_G"]).sum()),
    ("Scoring: player younger than 16 or older than 50", ((born["year"] - born["birthYear"] < 16)
                                                          | (born["year"] - born["birthYear"] > 50)).sum()),
    ("Goalies: goals against > shots against", ((goalies["SA"] > 0) & (goalies["GA"] > goalies["SA"])).sum()),
    ("Master: height outside 60-84 inches", (~players["height"].dropna().between(60, 84)).sum()),
    ("Master: weight outside 120-300 pounds", (~players["weight"].dropna().between(120, 300)).sum()),
    ("Coaches: w + l + t > g", (coaches[["w", "l", "t"]].fillna(0).sum(axis=1) > coaches["g"]).sum()),
], columns=["rule", "rows breaking it"])
print(rules.to_string(index=False))
rules.to_csv(f"{OUT}/inconsistency_rules.csv", index=False)
print("-> Rows breaking a rule are treated as missing (ch04: inconsistent data -> missing).")

# players traded during a season have one row per team (stint): games add up across teams
traded = scoring.groupby(["playerID", "year"])["tmID"].nunique()
print(f"Player-seasons with more than one team (trades): {(traded > 1).mean():.1%}")


# ---------- 5. noisy data ----------
# Points % of a short season is noisier: a few lucky games move it a lot.

title("5. Noisy data")

nhl = teams[(teams["lgID"] == "NHL") & teams["Pts"].notna()]
spread = nhl.groupby("year").agg(games=("G", "median"), spread=("pts_pct", "std"))
plt.figure(figsize=(8, 5))
sc = plt.scatter(spread["games"], spread["spread"], c=spread.index, cmap="viridis")
plt.colorbar(sc, label="season")
for y in [1994, 2005]:
    if y in spread.index:
        plt.annotate(str(y), (spread.loc[y, "games"], spread.loc[y, "spread"]))
plt.xlabel("games per team in the season")
plt.ylabel("std of points % between teams")
plt.title("NHL: spread of points % vs length of the season (colour = season)")
save("06_noise_points_vs_games")
print(f"Correlation games vs spread: {spread['games'].corr(spread['spread']):.2f}")
print("-> Short old seasons have a bigger spread (fewer games = noisier standings), but 1994")
print("   (48 games) is close to the modern seasons: the league also became more balanced over time.")
print("   Points % from different eras are not directly comparable, another reason for recent windows.")


# ---------- 6. outliers ----------

title("6. Outliers")

stats = ["pts_pct", "gf_pg", "ga_pg", "pim_pg"]
labels = {"pts_pct": "points %", "gf_pg": "goals for per game", "ga_pg": "goals against per game",
          "pim_pg": "penalty minutes per game"}

# 6a. box plots per decade (small multiples): the "normal" range changes with the era
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
for ax, s in zip(axes.flat, stats):
    sns.boxplot(data=teams, x="decade", y=s, ax=ax, color="#8C8C8C", fliersize=3)
    ax.set(title=labels[s], xlabel="", ylabel="")
    ax.tick_params(axis="x", rotation=45)
fig.suptitle("Team stats per decade (dots = outliers of the box plot, 1.5 IQR)")
save("07_boxplots_by_decade")


# 6b. three rules, applied to all seasons together (global) and inside each season (contextual)
def z_rule(x):
    return (x - x.mean()).abs() / x.std() > 3


def iqr_rule(x):
    q1, q3 = x.quantile([0.25, 0.75])
    return (x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))


def robust_z_rule(x):
    mad = (x - x.median()).abs().median()
    return (0.6745 * (x - x.median()) / mad).abs() > 3.5 if mad > 0 else x * False


rows = []
for s in stats:
    x = teams[s].dropna()
    for name, rule in [("z-score", z_rule), ("IQR", iqr_rule), ("robust z (MAD)", robust_z_rule)]:
        within = teams.dropna(subset=[s]).groupby("year")[s].transform(lambda v: rule(v).astype(int))
        rows.append({"stat": labels[s], "rule": name, "global": int(rule(x).sum()),
                     "within season": int(within.sum())})
flags = pd.DataFrame(rows)
print(flags.pivot(index="stat", columns="rule", values=["global", "within season"]).to_string())
print("-> Globally, many outliers are just a different era (e.g. very high penalty minutes in the late 1980s).")
print("   Inside each season the rules flag the teams that are really unusual for their time.")

teams["pim_robust_z"] = teams.groupby("year")["pim_pg"].transform(
    lambda v: 0.6745 * (v - v.median()) / (v - v.median()).abs().median())
plt.figure(figsize=(11, 5))
ok = teams["pim_robust_z"].abs() <= 3.5
plt.scatter(teams.loc[ok, "year"], teams.loc[ok, "pim_pg"], s=8, color="#8C8C8C", label="normal for its season")
plt.scatter(teams.loc[~ok, "year"], teams.loc[~ok, "pim_pg"], s=25, color="#C44E52",
            label="outlier inside its season (|robust z| > 3.5)")
plt.ylabel("penalty minutes per game")
plt.title("Contextual outliers: penalty minutes compared with the same season")
plt.legend()
save("08_contextual_outliers_pim")
print("\nMost unusual team-seasons in penalty minutes (inside their season):")
print(teams.loc[teams["pim_robust_z"].abs().sort_values(ascending=False).index[:8],
                ["year", "lgID", "name", "G", "pim_pg", "pim_robust_z"]].round(2).to_string(index=False))

# 6c. very short seasons: a team with a few games is not comparable with the others
short = teams[teams["G"] < 0.5 * teams.groupby(["year", "lgID"])["G"].transform("median")]
print("\nTeams that played less than half of the usual games of their league-season:")
print(short[["year", "lgID", "name", "G"]].to_string(index=False))
print("-> Candidates to drop from training (e.g. national teams in the 1978 WHA).")

# 6d. multivariate outliers: isolation forest on the 4 stats, standardized inside each season
z = teams.dropna(subset=stats).copy()
for s in stats:
    z[s + "_z"] = z.groupby("year")[s].transform(lambda v: (v - v.mean()) / v.std())
zc = [s + "_z" for s in stats]
z = z.dropna(subset=zc)
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=0).fit(z[zc])
z["anomaly_score"] = -iso.score_samples(z[zc])
z["flag"] = iso.predict(z[zc]) == -1
plt.figure(figsize=(8, 6))
plt.scatter(z.loc[~z["flag"], "gf_pg_z"], z.loc[~z["flag"], "ga_pg_z"], s=8, alpha=0.4, color="#8C8C8C",
            label="normal")
plt.scatter(z.loc[z["flag"], "gf_pg_z"], z.loc[z["flag"], "ga_pg_z"], s=30, color="#C44E52",
            label="isolation forest outlier (1%)")
plt.xlabel("goals for per game (standardized in the season)")
plt.ylabel("goals against per game (standardized in the season)")
plt.title("Multivariate outliers (points %, goals for/against, penalty minutes)")
plt.legend()
save("09_isolation_forest")
print("\nTeam-seasons the isolation forest finds most unusual:")
print(z.nlargest(8, "anomaly_score")[["year", "lgID", "name", "G"] + stats].round(2).to_string(index=False))

# 6e. players: height and weight together
plt.figure(figsize=(8, 6))
hb = plt.hexbin(players["height"], players["weight"], gridsize=25, mincnt=1, bins="log", cmap="Blues")
plt.colorbar(hb, label="players (log)")
plt.xlabel("height (inches)")
plt.ylabel("weight (pounds)")
plt.title("Players: height vs weight")
save("10_height_weight")
print(f"\nHeight vs weight correlation: {players['height'].corr(players['weight']):.2f}"
      " -> strongly related, but they are missing together (2c).")


# ---------- 7. targets over time ----------
# If the targets change over the years, old seasons teach the model a different
# "normal" (Lab 2: stationarity). This is why we train on a window of recent seasons.

title("7. Targets over time")

nhl_s = seasons[seasons["league"] == "NHL"]
over_time = nhl_s.groupby("year").agg(
    teams=("team_id", "count"),
    points_pct_spread=("points_%", "std"),
    coach_change_rate=("coach_changed_in_season", "mean"),
    penalty_minutes_per_game=("penalty_minutes_per_game", "mean"),
)
fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
for ax, (c, label) in zip(axes, [("teams", "teams in the NHL"),
                                 ("points_pct_spread", "std of points % (task a)"),
                                 ("coach_change_rate", "share of teams changing coach (task b)"),
                                 ("penalty_minutes_per_game", "penalty minutes per game (task c)")]):
    ax.plot(over_time.index, over_time[c], marker=".")
    ax.plot(over_time.index, over_time[c].rolling(5, center=True, min_periods=1).mean(), color="#C44E52",
            label="5-season mean")
    ax.set_ylabel(label, fontsize=9)
    ax.axvspan(1995, 2009, color="#4C72B0", alpha=0.08)
axes[0].legend()
axes[-1].set_xlabel("season (blue band = seasons we train and validate on)")
fig.suptitle("NHL targets over time")
save("11_targets_over_time")
print(over_time.loc[over_time.index >= 1990].round(3).to_string())
print("-> The targets drift (penalty minutes go down, the number of teams goes up), so the data is")
print("   not stationary: recent seasons are the best training data for 2010-2013.")

print(f"\nAll plots saved in {OUT}/")
