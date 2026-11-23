# To do

Everything still missing, from the teacher's feedback (T1, T2, T3 below) and the
grading rubric (readme/Grading_Rubric.md). Tick an item when it is done.

- T1 = feedback on the business understanding (years, rankings, one model, training window)
- T2 = feedback on the rubric (levels given halfway through the project, and task c)
- T3 = feedback on the plots, columns and missing values

## Already done

- [x] (a) ranking per division / per conference / all teams, depending on what we are given (T1)
- [x] (b) only coach changes in the middle of the season (T1)
- [x] Predict 2010-2013 training only on seasons we have results for (2009 and before) (T1)
- [x] Training window study (9, 7, 5, 3, 1 seasons) on 2005-2009, chosen by consistency vs baseline and variance (T1)
- [x] Only one model studied per task: Ridge (a), random forest (b) (T1)
- [x] "mae" instead of "points_%_mae" (T3)

## Business understanding

- [ ] Write down the business goals and how each one becomes a data mining goal (rubric: BU, all 3 criteria)
- [ ] Write down the questions we asked the teacher and the answers (T1, T2): evaluation years, ranking levels, mid-season changes, train only on known seasons
- [ ] Ask the teacher the exact format of the evaluation data (2010-2013)

## Data understanding

- [x] Analysis plots saved by a script (no notebooks) (T3): data_quality.py, outputs/data_quality/
  - [x] missing values per column and per season (which columns are badly filled)
  - [x] outliers (box plots of the main stats, contextual outliers, isolation forest)
  - [x] correlations between features, to find irrelevant or redundant ones
  - [x] eda.py: many more descriptive plots in the style of the lab notebooks (teacher: take inspiration from them), outputs/eda/
- [ ] Each plot and statistic with a reason for being there (rubric: diversity, "unjustified" now)
- [x] 2-3 plots with 3+ variables and a clear conclusion (rubric: complexity, "1D/2D" now): eda 10, 15, 16, 17
- [ ] Present the data understanding in a systematic way (T2: "done but not presented systematically")
- [ ] Statistics beyond counts and means (e.g. correlation tests), with interpretation (rubric: statistical methods)
- [ ] Write down the knowledge we got from each plot, not just the plot (rubric: knowledge extraction)

## Data preparation

- [ ] Decide which columns are useful and which are too badly filled to use (T3), and justify each one
- [~] data_quality.py done (keys, missing, redundant, inconsistent, noisy, outliers); still map it to the 6 dimensions (completeness, accuracy, consistency, uniqueness, validity, timeliness) (rubric: 6 dimensions)
- [x] Missing values with a regression (rubric: level 5; T3)
  - [x] Height and weight: always missing together (all players before 1980), so filled from era + position.
        data_quality.py hides known values and compares 4 methods: median per decade and position is best
        (MAE 1.54 in / 10.5 lb vs global median 1.89 in / 15.0 lb; linear regression 1.55 / 10.8). Used in prepare.py.
  - [x] Roster height/weight features (players_avg_height/weight) tested: no gain (a 0.267 -> 0.270, b F1 0.257 -> 0.213,
        c MAE 126.6 -> 126.3), so not used in the models. Mention as a model-improvement experiment.
  - [ ] Replace the median imputer in the models by a regression imputer inside the pipeline
- [ ] Outliers: identify, discuss and deal with them (rubric: outliers, not done)
  - [ ] 1978 WHA: Czechoslovakia and Finland national teams (a few games) - drop?
  - [ ] 1994 (48 games, lockout) - we already use points %
- [ ] Remove redundant attributes in a systematic way, e.g. very correlated features (rubric: level 4 now)
- [ ] Explain the feature engineering with business concepts: roster quality, coach experience, last-season lag features (rubric: level 2 now)
- [ ] Imbalanced data in (b): SMOTE inside the training seasons only, compared with class weights (rubric: not done)
- [x] Feature selection: feature_selection.py, filter (Spearman) + wrapper (forward selection), chosen on 1995-2003,
      compared on 2005-2009 (outputs/feature_selection/). Filter set used in (a) and (b); (c) keeps all features.
      (b) F1 0.257 -> 0.361 (min_samples_leaf 1, 5 seasons after re-tuning), bootstrap interval now above 0.
      Forward selection overfits the selection seasons ((a) 0.222 there, 0.310 on 2005-2009): say it in the slides.
- [ ] Explain the sampling: recent NHL seasons only for testing (domain), and the window study as "start small and grow" (rubric: sampling, levels 2 and 3 now)
- [ ] Database: show we use SQL for exploration and preparation (prepare.py), not only as storage (rubric: tools)

## Modelling

- [ ] compare_models.py: the 5 algorithms on 1995-2009 for (a), (b) and (c), to show why we chose each model (rubric: diversity of algorithms was level 5; we removed it from task_a/task_b)
- [x] task_c.py: predict each team's penalty minutes (PIM) for the test season (T2)
  - [x] baseline: last season's PIM
  - [x] metrics: MAE and Spearman
  - [x] same window study as (a) and (b)
  - [ ] explain in the slides why Ridge predicts the change from last season (PIM falls every season)
- [x] Parameter tuning with a systematic search: alpha (a, c) and min_samples_leaf (b) × window, on 2005-2009 (validation; the real test is 2010-2013) (rubric: not done)
  - [ ] Slides: say the 2005-2009 numbers are a bit optimistic (best of many combinations on 5 seasons)
- [ ] Explain how each chosen algorithm works and what its parameters do (rubric: understanding algorithm behavior)
- [ ] Model improvement: list the experiments and what each one improved (rubric: model improvement)

## Evaluation

- [ ] Explain why each metric fits the data: F1/AUC because changes are rare, Spearman/rank error because the goal is a ranking, MAE for PIM (rubric: level 3 now)
- [x] Significance vs baseline: no formal test in the slides/labs, so bootstrap 95% intervals (Lab 2 8.4) at the end of each task script.
      After feature selection: (a) -0.017 [-0.050, +0.015] includes 0; (b) F1 +0.183 [+0.029, +0.323] above 0;
      (c) -14.1 [-22.2, -5.9] below 0. Slides: (b) and (c) are real gains; (a) better on average but could be luck.
- [x] Overfitting: explain_models.py, training vs validation per parameter and window (outputs/explain/overfitting.png).
      (c) clear bias-variance curve (alpha 1: train 104 / val 140; alpha 100: 125 / 127). (b) forest fits training perfectly
      (AUC 1.0) but validation is still the best: say that training AUC of a forest is not informative.
- [ ] Time: training and prediction time of each model (rubric: other factors)
- [x] Feature importance: forest importances for (b) (coach_win_% first), Ridge coefficients for (a) and (c). Explain in hockey terms in the slides.
- [x] White-box models: Ridge coefficients and depth-3 decision trees for each task (outputs/explain/).
- [ ] Final predictions for 2010-2013 with the teacher's data

## Project management and presentation

- [ ] Say which methodology we followed (CRISP-DM) and show its steps in the project (rubric: methodology)
- [ ] Plan with tasks and who does what, kept in a tool (e.g. GitHub issues/projects, Trello) (rubric: plan, PM tools)
- [ ] Show how we collaborate: git, shared data, communication (rubric: collaboration tools)
- [ ] Presentation: 15 min, max 40 slides, simple and clear, to the point (rubric: presentation)
- [ ] Mention all the teacher's feedback (T1, T2, T3) and what we changed because of it

## Code and repository

- [ ] Commit the current changes (prepare.py, task_a.py, task_b.py, TODO.md)
- [x] eda.py still uses NHL 1980+ only and the old ranking: update it
- [x] Short README with the order to run the scripts (README.md)
