# AC-G13: Ice hockey data mining

Predicting three things about each team of an ice hockey season, using only what is
known before the season starts (data: hockey seasons 1909-2009, folder `csv_datas/`):

- **(a)** the final ranking of the regular season (inside each division, conference, or league)
- **(b)** which teams will change coach during the season
- **(c)** the penalty minutes of each team

## Setup

```
pip install -r requirements.txt
```

## Order to run the scripts

Run everything from the project folder. Results go to `outputs/`.

| Step | Command | What it does |
|---|---|---|
| 1 | `python make_db.py` | puts all the CSV files in one SQLite database (`outputs/hockey.db`) |
| 2 | `python prepare.py` | builds one row per team and season, with the targets and the features (`outputs/team_seasons.csv`) |
| 3 | `python data_quality.py` | keys, missing values, redundancy, rules, noise, outliers (`outputs/data_quality/`) |
| 4 | `python eda.py` | exploratory plots for the three tasks (`outputs/eda/`) |
| 5 | `python task_a.py` | task (a): chooses the training window and alpha on 2005-2009 |
| 6 | `python task_b.py` | task (b): chooses the training window and min_samples_leaf on 2005-2009 |
| 7 | `python task_c.py` | task (c): chooses the training window and alpha on 2005-2009 |
| 8 | `python feature_selection.py` | filter and forward selection (takes about 15 minutes) (`outputs/feature_selection/`) |
| 9 | `python explain_models.py` | model coefficients, importances, small trees and overfitting check (`outputs/explain/`) |

Steps 3, 4, 8 and 9 are analysis only: the task scripts do not need them.

## Predicting a season

Each task script predicts one season when it gets the year:

```
python task_a.py 2012
python task_b.py 2012
python task_c.py 2012
```

The models are trained only on the seasons that have results (2009 and before), on the
window of seasons chosen in step 5-7. To predict a new season (e.g. 2010-2013), add its
teams, players and coaches to the CSV files in `csv_datas/` and run steps 1 and 2 again.
`prepare.py` also adds the 4 seasons after the last one in the data, with the NHL teams
of that season, so the scripts can run before the new data arrives.

## Files

- `src/data.py`: loads the CSV files with the right column types
- `TODO.md`: what is done and what is missing (from the teacher's feedback and the rubric)
- `readme/`: project description, instructions and grading rubric
