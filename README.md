Home Credit Default Risk --- Feature Engineering & Winner-Inspired Experiments

An end-to-end machine learning case study for the Kaggle Home Credit
Default Risk competition, focused on relational feature
engineering, LightGBM, controlled experimentation, and learning
from high-ranking Kaggle solutions.

Project Overview

Home Credit aims to predict whether a loan applicant is likely to
experience payment difficulties. The competition is evaluated using
ROC-AUC.

This project was built incrementally from scratch. Each major feature
family was introduced as a separate experiment and evaluated with
cross-validation.

The second phase studies ideas from strong Kaggle solutions using:

Understand → Implement → Validate → Keep or Reject

The objective is not only to improve the leaderboard score, but to
understand why a technique works and whether it adds information beyond
the existing pipeline.

Results

Model: LightGBM

Validation: Stratified K-Fold

Metric: ROC-AUC

Reconstructed full-CV baseline: 0.791721

Kaggle private leaderboard score: 0.78868

The strongest improvements came from relational feature engineering and
recent behavioral information rather than extensive hyperparameter
tuning.

Dataset Structure

application_train / application_test
              |
              +-- bureau
              |      |
              |      +-- bureau_balance
              |
              +-- previous_application
              +-- installments_payments
              +-- POS_CASH_balance
              +-- credit_card_balance

The central challenge is to transform many historical records into
meaningful customer-level features without losing important behavioral
information.

Modeling Strategy

Raw application data
        ↓
Application-level ratios
        ↓
Bureau + Bureau Balance
        ↓
Previous Applications
        ↓
Installments + POS_CASH + Credit Cards
        ↓
Recent 6M / 12M / 24M behavior
        ↓
Robust ratio transformations
        ↓
LightGBM + Stratified K-Fold CV
        ↓
Winner-inspired hypothesis testing

Feature Engineering

Application

Financial ratios represent burden relative to income instead of relying
only on absolute values.

Examples: CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO,
CREDIT_ANNUITY_RATIO, and EXT_SOURCE features.

EXT_SOURCE_1, EXT_SOURCE_2, and EXT_SOURCE_3 remained among the
strongest predictors.

Bureau & Bureau Balance

External credit-history features include credit counts, active/closed
loans, credit and debt amounts, debt-to-credit ratios, credit age,
overdue behavior, prolongations, and monthly status history.

Previous Applications

Previous Home Credit applications were summarized using application
counts, approval/refusal behavior, requested/granted credit,
credit/application ratios, and application timing.

Installments

Installment history models late-payment ratios, 30+ day lateness,
payment delay, underpayment, payment amounts, and scheduled amounts.

Important recent signals included INST_24M_MAX_DELAY and
INST_24M_LATE_RATIO.

POS_CASH

POS history captures days past due, contract history, remaining
installments, and completion behavior.

Credit Card

Credit-card features include utilization, maximum utilization, drawing
behavior, ATM usage, payment/minimum-payment ratios, balances, and
delinquency.

CC_6M_UTIL_MAX was an important recent signal.

Recency Engineering

A major finding was that recent behavior was more informative than
simply adding more lifetime aggregates.

Historical windows were created for 6, 12, and 24 months across
installments, credit cards, and bureau history.

INST_6M_*   INST_12M_*   INST_24M_*
CC_6M_*     CC_12M_*     CC_24M_*
BUR_6M_*    BUR_12M_*    BUR_24M_*

Trend features compare recent behavior with broader history, such as
INST_LATE_TREND_6M_24M, CC_UTIL_TREND_6M_24M, and
BUR_DEBT_CREDIT_TREND_6M_24M.

Robust Ratio Features

Extreme bureau debt-to-credit ratios were stabilized with:

percentile clipping at the 0.5th and 99.5th percentiles;

signed-log transformation: sign(x) * log(1 + abs(x)).

These produced a small positive CV improvement.

Experiment History

Version   Main experiment              CV ROC-AUC Outcome

V1        Raw application                0.759101 Baseline
V2        EXT engineering                0.758507 Rejected
V3        Financial ratios               0.767108 Kept
V4        Employment features            0.766892 Rejected
V5        Bureau history                 0.771489 Kept
V6        Bureau balance                 0.772063 Kept
V7        Previous applications          0.777159 Kept
V8        Installment behavior           0.783466 Kept
V9        POS_CASH history               0.785156 Kept
V10       Credit-card history            0.787422 Kept
V11       Recent behavior                0.789166 Kept
V12       Recent bureau                  0.791330 Kept
V13       Robust ratios                  0.791549 Kept
V14A      Cross-table interactions       0.791589 Marginal

Baseline Reproduction

Metric                                Score

Original V13 OOF ROC-AUC           0.791549
Reconstructed OOF ROC-AUC      0.791721
Difference                    +0.000172

The close reproduction allowed the feature baseline to be frozen before
winner-inspired experiments.

Winner-Inspired Experiments

Full five-fold training became relatively expensive, so the research
phase uses the same first three folds from the fixed five-fold split
for fast screening. The screening baseline is 0.792002.

Experiment   Technique               Added     3-fold AUC           Delta Decision
features

Baseline     Reconstructed             ---       0.792002             --- Reference
V13

V17          Last-N                    +21   0.792210   +0.000208 Keep
installment                                                  candidate
events

V18          Multi-level               +29       0.791907       -0.000095 Park
installments

V17 --- Last-N Installment Events

V17 asks not only "what happened in the last six months?" but also "what
happened in the customer's last 3, 5, or 10 payment events?"

This represents event-order recency, which differs from
calendar-time recency.

Baseline: 0.792002
V17:      0.792210
Delta:   +0.000208

The improvement is small but suggests event order may contain
information not fully represented by fixed calendar windows.

V18 --- Multi-Level Installment Aggregation

V18 tested
installment events → SK_ID_PREV → previous-loan features → SK_ID_CURR.

Delta: -0.000095. The idea was parked because the additional
hierarchy appeared largely redundant with the existing installment
features.

V19 --- Conditional Previous Applications

Previous applications were aggregated separately for Approved and
Refused states.

Delta: -0.000175. This experiment was rejected because it did not
improve ranking performance over the existing previous-application
features.

What Worked

Relational feature engineering produced the largest improvements.

Installment-payment behavior was highly informative.

Recent behavior outperformed blindly adding more historical
aggregates.

Robust transformations helped unstable financial ratios.

Fixed validation was essential when gains were small.

Event-order recency is a promising research direction.

What Did Not Work

Negative experiments are intentionally documented:

standalone EXT_SOURCE engineering in V2;

employment features in V4;

many cross-table interactions;

multi-level installment aggregation in V18;

conditional previous-application aggregation in V19.

A technique being intuitive---or appearing in a strong Kaggle
solution---does not guarantee incremental value in an already developed
pipeline.

Validation Methodology

Primary validation:

StratifiedKFold
n_splits = 5
shuffle = True
random_state = 42
metric = ROC-AUC

Winner-inspired research uses the same first three folds for faster,
directly comparable screening.

Repository Structure

home-credit-default-risk/
├── README.md
├── home_credit_default_risk.ipynb
├── requirements.txt
├── .gitignore
└── data/   # local only; not committed

The repository is intentionally centered around a single documented
notebook.

Installation

pip install -r requirements.txt

Main dependencies: NumPy, pandas, scikit-learn, LightGBM, and Jupyter.

Data

The dataset is provided through the Kaggle Home Credit Default Risk
competition and is not included in this repository.

Place the competition CSV files in ./data/ (or configure
HOME_CREDIT_DATA_DIR).

Expected files include:

application_train.csv
application_test.csv
bureau.csv
bureau_balance.csv
previous_application.csv
installments_payments.csv
POS_CASH_balance.csv
credit_card_balance.csv

Running the Project

Obtain the competition dataset.

Put the CSV files in ./data/.

Install dependencies.

Open home_credit_default_risk.ipynb.

Restart the kernel.

Run all cells from top to bottom.

A fresh-kernel run verifies that the notebook is fully reproducible.

Technologies

Python · pandas · NumPy · scikit-learn · LightGBM · Jupyter Notebook ·
Git · GitHub · Kaggle

Key Takeaway

Understand the data
        ↓
Form a hypothesis
        ↓
Engineer a meaningful representation
        ↓
Validate under fixed conditions
        ↓
Keep or reject
        ↓
Document what was learned

The same process is used when studying winning solutions: the goal is to
understand what information a technique adds, not to treat winning
code as a black box.

Future Work

Last-N / lag features for additional relational tables

Further event-order recency experiments

Feature importance and redundancy analysis

Adversarial validation

Alternative boosting models

Multi-seed LightGBM

LightGBM / XGBoost / CatBoost blending

Rank averaging

Out-of-fold stacking

Ensembling will be studied only after extracting as much value as
possible from feature engineering and model understanding.

Author

This repository documents a learning-focused project in credit-risk
modeling, relational feature engineering, and experimental machine
learning.

Feedback and suggestions are welcome.
