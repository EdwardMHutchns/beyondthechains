# Beyond the Chains
### A Statistical Investigation of NFL Team and Player Performance

**Python | Pandas | NumPy | SciPy | Matplotlib | Seaborn | Scikit-learn | Jupyter | Git**

## Project Overview

**Beyond the Chains** is an ongoing sports analytics and data science project focused on investigating historical NFL performance through statistical analysis, numerical methods, and eventually machine learning.

The project begins by examining league-wide offensive performance across 26 NFL regular seasons (2000–2025). It progressively expands toward individual team evaluations, opponent matchups, positional performance, and player-level forecasting.

The broader objective is to understand the statistical factors associated with football performance and eventually develop probabilistic models capable of forecasting individual player statistics while quantifying prediction uncertainty.

Rather than beginning immediately with machine learning, the project follows a research-driven approach: exploring the data, evaluating statistical relationships, identifying relevant predictors, and validating models against historical outcomes.

---

## Research Objectives

The investigation is organized around several central questions:

1. How has NFL offensive production changed across seasons?
2. How do rushing and passing performances differ in terms of central tendency, dispersion, and distributional shape?
3. Which NFL teams consistently demonstrate strong offensive production?
4. How do opposing defenses, supporting personnel, and game situations relate to offensive performance?
5. Can historical team and player statistics improve forecasts of future performance?
6. How reliably can probabilistic models estimate the likelihood of specific player-performance outcomes?

---

## Dataset

**Source:** NFL weekly team statistics accessed through the `nflreadpy` Python package.

**Historical coverage:** 2000–2025 (26 NFL seasons)

**Original dataset:**
- Approximately 13,418 team-game observations
- 138 variables
- Regular-season observations used for the primary analysis

### Observational Unit

Each row represents one team's statistical performance during a particular NFL game.

Consequently, an individual matchup generally contributes two observations: one for each participating team.

This structure enables analysis at multiple levels:

- **League-season:** Average offensive production across all team-game observations within a season.
- **Team-season:** Average and distributional statistics for an individual team across its games.
- **Team-game:** Performance against a specific opponent during a particular week.

Future player-level investigations will require additional datasets containing individual player statistics, usage, and availability.

---

## Research Completed

### 01 — Historical NFL Rushing Performance

**Status: Descriptive analysis completed**

Investigated league-wide rushing production across 26 NFL regular seasons.

**Methods applied:**
- Mean, median, variance, and standard deviation
- Quartiles and interquartile range
- Skewness and excess kurtosis
- Histograms and kernel density estimation
- Seasonal trend visualizations
- Violin plots and distributional comparisons
- Exploratory confidence intervals for seasonal means

**Selected findings:**

| Metric | Result |
|---|---:|
| Overall mean rushing yards per team-game | 114.80 |
| Overall median | 108.00 |
| Highest seasonal average | 2022 — 121.58 |
| Lowest seasonal average | 2015 — 108.84 |
| Change from 2000 to 2025 | +3.71% |
| Pooled skewness | 0.789 |
| Pooled excess kurtosis | 0.935 |

Rushing production fluctuated throughout the historical period without a clear sustained linear trend.

The pooled distribution displayed positive skewness and excess kurtosis, consistent with a longer right tail and a fourth standardized moment above the normal-distribution reference.

These findings establish a descriptive baseline for subsequent team-level rushing investigations.

### 02 — Historical NFL Passing Performance

**Status: Descriptive analysis completed**

Examined changes in league-wide passing production, variability, and distributional characteristics.

**Methods applied:**
- Descriptive statistics and seasonal aggregation
- Historical comparisons of central tendency
- Standard deviation and variance analysis
- Skewness and excess kurtosis
- Histograms, KDE, and violin plots
- Exploratory confidence intervals for seasonal means

**Selected findings:**

| Metric | Result |
|---|---:|
| Overall mean passing yards per team-game | 236.90 |
| Overall median | 232.00 |
| Highest seasonal average | 2015 — 259.16 |
| Lowest seasonal average | 2003 — 213.94 |
| Change from 2000 to 2025 | +1.23% |
| Pooled skewness | 0.328 |
| Pooled excess kurtosis | 0.169 |

Passing production exhibited a pronounced increase during the early-to-mid 2010s, followed by a general decline in more recent seasons.

Although the average in 2025 slightly exceeded that of 2000, the endpoint comparison obscures substantial changes during the intervening years.

Passing-yard distributions generally exhibited less positive asymmetry than rushing-yard distributions.

### 03 — Dataset Validation and Exploratory Linear Algebra

**Status: Initial investigation completed**

Evaluated the structure, completeness, and numerical characteristics of the historical NFL dataset.

**Data validation:**
- Inspected available variables and data types
- Examined missing-value patterns
- Investigated duplicate team-game observations
- Checked game-level grouping consistency
- Examined the availability of numerical features

**Linear algebra methods:**
- Matrix rank analysis
- Singular Value Decomposition (SVD)
- Principal Component Analysis (PCA)
- Explained-variance analysis
- Principal component loadings
- Two-dimensional component projections

**Selected findings:**

| Metric | Result |
|---|---:|
| Numerical variables examined | 128 |
| Calculated centered matrix rank | 116 |
| Numerical rank deficiency | 12 |
| Variance explained by PC1 | 10.81% |
| Variance explained by PC2 | 6.22% |
| Components explaining 90% variance | 59 |
| Components explaining 95% variance | 70 |

The analysis identified numerical dependencies among features and demonstrated that much of the standardized variance could be represented using fewer principal components.

Preliminary loading analysis suggested that the first two components were associated with different aspects of offensive production, including passing, rushing, scoring, and punting activity.

PCA was used for exploratory visualization and structural analysis, not as evidence of predictive importance.

---

## Research Roadmap

The project will expand through the following phases.

| Phase | Research focus | Status |
|---|---|---|
| 1 | League-wide offensive EDA | Completed for rushing and passing |
| 2 | Individual team-season analysis | Next |
| 3 | Player-level exploratory analysis | Planned |
| 4 | Team composition and positional analysis | Planned |
| 5 | Opponent and matchup evaluation | Planned |
| 6 | Statistical inference and regression | Planned |
| 7 | Feature engineering | Planned |
| 8 | Machine learning and forecasting | Planned |
| 9 | Probabilistic modeling and backtesting | Planned |

### Next Phase: Individual Team-Season Analysis

The next investigation will transition from pooled league-wide observations to individual team performance.

Planned analyses include:

- Seasonal rushing and passing yards per game by team
- Team rankings and historical comparisons
- Game-to-game consistency and variability
- Offensive balance between rushing and passing
- Differences between high- and low-production teams
- Relationships between offensive production and opponent characteristics

The objective is to determine which teams consistently performed well and identify patterns that may explain differences in offensive production.

### Future Expansion: Player Performance Analysis

Player-level research will focus on quarterbacks, receivers, running backs, and other positional groups.

Potential areas of investigation include:

- Quarterback efficiency and consistency
- Receiver target share, catch rate, yards after catch, and drops
- Offensive-line performance and quarterback pressure
- Injuries and changes in player availability
- Opponent defensive strengths and weaknesses
- Relationships between player production and supporting personnel

### Long-Term Goal: Probabilistic Player Forecasting

The eventual goal is to develop models that estimate both expected player performance and the uncertainty associated with those estimates.

Potential targets include:

- Quarterback passing yards
- Running back rushing yards
- Receiver targets and receiving yards
- Passing and receiving touchdowns

Models will be evaluated using chronological out-of-sample testing, comparisons against baseline forecasts, predictive intervals, and probability calibration.

The objective is not simply to predict an expected statistic, but to evaluate the reliability of the forecast and estimate probabilities associated with performance thresholds.

---

## Statistical Methodology

The project follows a progressive analytical framework:

**Descriptive analysis → Data validation → Statistical inference → Feature engineering → Predictive modeling → Validation**

Statistical techniques will be selected according to the research question, available data, and required assumptions.

Important considerations include observational dependence, changes in NFL rules and strategies, sample size, uncertainty quantification, and prevention of data leakage in forecasting experiments.

The completed analyses are primarily descriptive. Observed correlations, differences in historical averages, and PCA components should not be interpreted as causal relationships.

---

## Technology Stack

**Programming and data manipulation:** Python, Pandas, NumPy

**Statistical analysis:** SciPy, statistical inference, regression methods

**Visualization:** Matplotlib, Seaborn

**Numerical methods:** Linear algebra, SVD, PCA, scikit-learn

**Development environment:** Jupyter Notebooks, Visual Studio Code, Git, GitHub

Additional tools and methods may be introduced as the project expands.

---

## Project Status

**Active development — Historical league-wide exploration completed.**

The current repository documents historical offensive performance and numerical exploration of the NFL team-game dataset.

The next stage will focus on individual team-season performance and comparative offensive analysis.

*Beyond the Chains is an independent educational and research project and is not affiliated with the NFL.*