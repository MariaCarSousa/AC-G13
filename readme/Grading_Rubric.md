# Grading rubric (directly from Moodle)

Every criterion is graded on 7 levels, from worst to best:

**not done** → level 1 → level 2 → level 3 → level 4 → level 5 → **beyond expectations**

The tables below show levels 1–5 for each criterion ("–" means the level has no description). "Not done" and "beyond expectations" apply to every criterion and are left out.

## Business Understanding (BU)

| Criterion | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Analysis of requirements with the end user | minimal attempts to interact | – | limited to determining the business objectives | – | extensive identification of business objectives and additional relevant business information |
| Definition of business goals | essentially incorrect/very incomplete business goals | – | essential goals were identified | most goals identified after many trial and error iterations | clearly and quickly identified goals |
| Translation of business goals into data mining goals | essentially incorrect/very incomplete translation | not aligned | essential DM goals aligned with business goals | DM goals aligned with business goals after many trial and error interactions | DM goals clearly aligned with business goals |

## Data Understanding (DU)

| Criterion | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Diversity of statistical methods | single or very limited set of methods | – | diverse but unjustified set of methods | – | rich and justified set of methods |
| Complexity of statistical methods | simple 1D and 2D methods | complex 3+D methods with unclear results | – | complex 3+D methods with clear results | integrated, complex 3+D methods with clear results |
| Interpretation of results of statistical methods | incorrect | – | sometimes correct but often incorrect | – | generally correct |
| Knowledge extraction from results of statistical methods | trivial knowledge obtained | – | interesting OR novel and non-trivial knowledge obtained | – | interesting, novel and non-trivial knowledge obtained |
| Diversity of plots | single or very limited set of methods | – | diverse but unjustified set of methods | – | rich and justified set of methods |
| Complexity of plots | simple 1D and 2D methods | complex 3+D methods with unclear results | – | complex 3+D methods with clear results | integrated, complex 3+D methods with clear results |
| Presentation | confusing, unstructured set of visualizations | – | simple, clear information, even if not entirely well-structured | – | well-structured information |
| Interpretation of plots | incorrect | – | sometimes correct but often incorrect | – | generally correct |
| Visual knowledge extraction | trivial knowledge obtained | – | interesting OR novel and non-trivial knowledge obtained | – | interesting, novel and non-trivial knowledge obtained |

## Data Preparation (DP)

| Criterion | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Data integration | – | – | – | conversion of different formats of data values OR matching entities between different sources | conversion of different formats of data values AND matching entities between different sources |
| Assessment of dimensions of data quality | – | 2 dimensions (usually accuracy and completeness) | 3 dimensions (... consistency or uniqueness) | 4 dimensions (... consistency and uniqueness) | 6 dimensions |
| Cleaning: redundancy | – | – | – | removal of some redundant attributes | systematic removal of redundant attributes |
| Cleaning: missing data | remove a significant number of columns and/or rows | replace MVs with constant | replace MVs with statistically simple method (e.g. mean, mode) | replace MVs with complex method (e.g. regression, classification) with incorrect experimental setup | replace MVs with complex method (e.g. regression, classification) with correct experimental setup |
| Cleaning: outliers | – | identify outliers | identify and discuss outliers | identify and discuss outliers, and address them with simple approach | identify and discuss outliers, and address them with complex (technical or domain-dependent) approaches |
| Data transformation for compatibility with algorithms | – | inadequate, simple discretization OR rescaling | inadequate complex or adequate simple discretization OR rescaling | – | adequate complex discretization or rescaling |
| Feature engineering from tabular data | – | simple methods and knowledge (e.g. ratios) | – | complex methods (e.g. aggregation) OR knowledge (e.g. business concepts) | complex methods (e.g. aggregation) AND knowledge (e.g. business concepts) |
| Sampling for domain-specific purposes | inadequate use of the whole sample | focus on inadequate subset of population | – | – | focus on adequate subset of the population |
| Sampling for development | – | use insufficient sample | non-systematic use of sampling | – | start with very small sample and grow to significant one |
| Imbalanced data | – | use basic resampling methods incorrectly (e.g. whole dataset) | use basic resampling methods correctly | used advanced methods (e.g. SMOTE) incorrectly (e.g. whole dataset) | used advanced methods (e.g. SMOTE) correctly |
| Feature selection | use filter-based methods incorrectly (e.g. whole dataset) | use filter-based methods correctly | use wrapper-based methods incorrectly (e.g. whole dataset) | use wrapper-based methods correctly | correct, combined use of filter and wrapper-based methods |

## Predictive modelling

| Criterion | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Diversity of tasks | – | – | – | classification OR regression | classification AND regression |
| Diversity of algorithms | tested 1 algorithm | – | tested 3 algorithm | – | tested 4+ with significantly different language bias OR with a significant number of variants |
| Parameter tuning | – | – | limited, manual approach | – | systematic approach |
| Understanding algorithm behavior | significant confusion about the behavior of all algorithms | basic understanding of <3 algorithms | solid (even if not deep) understanding of <3 algorithms, without understanding the effect of the parameters | – | solid (even if not deep) understanding of the behavior of the majority of the algorithms used OR of <3 algorithms, also understanding the effect of the parameters |
| Performance estimation: training vs test | training and testing on the same data | training and testing on separate data but with leakage | – | training and testing on correctly separated data | training and testing on correctly separated data, with multiple splits |
| Performance estimation: other factors (e.g. time) | additional factors not taken into account | – | additional factors taken into account but not correctly | – | additional factors correctly taken into account |
| Performance estimation: performance measure | focus on performance measure(s) neither aligned with DM goals nor data characteristics | – | focus on performance measure(s) aligned with DM goals but not data characteristics | – | focus on performance measure(s) aligned with DM goals and data characteristics |
| Performance estimation: correct interpretation of performance measures | – | – | basic performance measures (e.g. accuracy, error, MSE, MAD) | medium complexity performance measures (e.g. precision, recall, ...) | advanced performance measures (e.g. AUC) |
| Performance estimation: baseline | – | – | unsuitable baseline | – | suitable baseline |
| Performance estimation: analysis of results | – | incorrect analysis of values for comparison | – | correct analysis of values for comparison, including informal discussion of statistical significance | correct analysis of values for comparison, including tests for statistical significance |
| Performance estimation: analysis of overfitting | incorrect estimation of overfitting | – | correct estimation of overfitting, analyzed incorrectly | – | correct estimation of overfitting, correctly analyzed |
| Model improvement | 1 iteration, involving a limited set of experiments | – | random set of experiments, without concern for pedagogical goals | random set of experiments, with total focus on pedagogical goals | development guided by performance improvement goals, even if pedagogical goals were not ignored |
| Feature importance | – | interpreted incorrectly, without relating to application domain | interpreted incorrectly, relating to application domain | interpreted correctly, without relating to application domain | interpreted correctly, relating to application domain |
| Analysis of "white-box" models | – | interpreted incorrectly, without relating to application domain | interpreted incorrectly, relating to application domain | interpreted correctly, without relating to application domain | interpreted correctly, relating to application domain |

## Project management

| Criterion | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Methodology | – | a methodology was mentioned but the development of the project didn't seem to follow it | the main steps of a methodology were followed loosely | – | a methodology was followed carefully, reflecting on the documentation produced |
| Plan | – | a plan was created but there was no concern to follow it | a rough plan was created and it guided the development | – | a detailed plan was created and it guided the development |
| PM tools | simple, coarse task sharing | – | detailed task sharing, not supported by appropriate management tool | – | detailed task sharing, supported by appropriate management tool |
| Collaboration tools | complex, unreliable data and workflow sharing | easy communication and unreliable data and workflow sharing | reasonable collaboration with simple collaboration tools | easy communication and documentation, and reasonable collaboration with simple tools | easy collaboration (communication, data, workflows and documentation) |

## Tools

| Criterion | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Analytics | – | 1 tool, from the ones presented in the courses | – | – | multiple tools, from the ones presented in the courses |
| Database | only as master | – | used for simple data preparation operations | – | extensive use for efficient data exploration and preparation |
| Other tools (e.g. data cleaning, visualization) | – | unsuitable use of other tools | – | limited but correct use of other tools | extensive and correct use of other tools |

## Presentation

| Criterion | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Quality of layout | hard to understand | – | simple OR clear | – | simple AND clear |
| Quality of content in slides | extremely incomplete or excessive | – | incomplete or too much content | – | to the point |
| Delivery | confusing | – | clear OR with confidence | – | clear AND with confidence |
| Use of time | extremely long or insufficient for evaluation | – | slight too long or significantly too short | – | to the point |
