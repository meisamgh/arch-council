# Feature & Model Structure Research Engine

A production-oriented system for automatically researching:

- **which features may matter**
- **which model structures may be strong for the problem**
- whether those ideas can actually be built from the available data
- whether they survive deterministic validation
- whether they produce measurable improvement

> **Research proposes.  
> Mapping determines feasibility.  
> Deterministic systems validate correctness.  
> Experiments determine value.**

---

# Core Idea

The system separates two research problems:

```text
What information might predict the target?

What modeling structure might best exploit that information?
```

Therefore it contains two independent research components:

```text
Feature Seeker
+
Structure Seeker
```

Both initially operate **without seeing the dataset**.

This prevents available columns or the current modeling stack from limiting research.

---

# Architecture

```text
                     Problem Contract
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       FEATURE SEEKER            STRUCTURE SEEKER
        NO DB ACCESS              NO DB ACCESS
              │                         │
              ▼                         ▼
     Feature Research            Model Research
              │                         │
      Papers / GitHub            Papers / GitHub
      Kaggle / Similar           Kaggle / Benchmarks
      Problems / Domains         Similar Problems
              │                         │
              ▼                         ▼
      ResearchFeature[]          ModelStructure[]
              │                         │

      ═════════════ DATA BOUNDARY ═════════════

              │                         │
              ▼                         ▼
       FEATURE MAPPER             MODEL MAPPER
              │                         │
              ▼                         ▼
        MappedFeature              ModelSpec
              │                         │
              └────────────┬────────────┘
                           ▼
                 Deterministic Validation
                           │
                           ▼
                    Feature Generation
                           │
                           ▼
                Feature × Model Experiments
                           │
                           ▼
                   Coarse Evaluation
                           │
                           ▼
                    Fine Evaluation
                           │
                           ▼
                      Ablation
                           │
                           ▼
                 Final Portfolio Freeze
                           │
                           ▼
                    SEALED HOLDOUT
                           │
                           ▼
                 Research + Experiment KB
```

---

# 1. Problem Contract

Example:

```yaml
problem:
  task: customer_churn
  objective: predict churn within 30 days
  prediction_horizon: 30_days
  entity: customer
  domain: subscription_product
```

The contract describes the problem.

It does **not** expose database columns.

---

# 2. Search Contract

The user controls research depth.

```yaml
research:
  requested_features: 300
  requested_feature_families: 30
  requested_model_structures: 3

  search:
    direct_problem: true
    similar_problems: true
    mechanisms: true
    analogies: true

  sources:
    - papers
    - github
    - kaggle

  limits:
    max_rounds: 6
    max_queries: 100
```

Example request:

```text
Find 300 feature ideas and 3 strong model structures
for predicting customer churn.
```

---

# 3. Feature Seeker

The Feature Seeker answers:

> What information could help predict this outcome?

It researches:

```text
same problem
similar problems
analogous domains
mechanisms
papers
Kaggle
GitHub
domain literature
```

It does not see:

```text
database
schema
raw rows
target values
```

---

## Example Feature

```yaml
name: recent_activity_ratio

concept: engagement_deterioration

explanation:
  Compares recent activity with historical activity.

mechanism:
  A sharp decline in engagement may precede churn.

required_information:
  - customer identifier
  - activity event
  - timestamp

typical_formula:
  recent_activity / historical_activity

similar_problems:
  - subscription_cancelation
  - customer_dormancy
  - player_abandonment
```

The seeker says:

```text
activity timestamp
```

not:

```text
sessions.session_date
```

because it does not know the database.

---

# 4. Feature Families

Research should discover meaningful concepts rather than trivial permutations.

Example:

```text
Feature Family:
Recent vs Historical Engagement
```

Possible operations:

```text
ratio
difference
trend
acceleration
volatility
```

Possible windows:

```text
7d
14d
30d
60d
90d
```

The deterministic generator can later produce:

```text
activity_ratio_7_30
activity_ratio_14_90
activity_difference_7_30
activity_slope_30
activity_acceleration
```

So:

> **LLM discovers the concept.  
> Deterministic generation creates variants.**

---

# 5. Structure Seeker

The Structure Seeker answers:

> What modeling strategies have strong justification for this type of problem?

It researches:

- papers
- Kaggle solutions
- GitHub projects
- benchmark approaches
- similar prediction problems

It does not simply return model names.

It returns complete **model structures**.

---

## Example: Churn

### Structure 1 — CatBoost Baseline

```text
Features
   ↓
CatBoost
   ↓
Churn probability
```

Useful when the problem contains:

```text
numerical variables
categorical variables
nonlinear interactions
missing values
```

---

### Structure 2 — Boosting Stack

```text
                  Features
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       XGBoost              CatBoost
          │                     │
          ▼                     ▼
      OOF prediction        OOF prediction
          └──────────┬──────────┘
                     ▼
          L2 Logistic Regression
                     │
                     ▼
            Final Probability
```

Important:

```text
Meta-model training must use
out-of-fold base predictions.
```

---

### Structure 3 — Survival Modeling

For problems where the timing of churn matters:

```text
Customer History
       ↓
Survival Model
       ↓
Probability of remaining active
7 / 30 / 60 / 90 days
```

This represents a genuinely different modeling assumption rather than merely another classifier.

---

# 6. ModelStructure

```yaml
name: dual_boosting_stack

problem_type:
  binary_classification

architecture:
  base_models:
    - XGBoost
    - CatBoost

  meta_model:
    - L2 Logistic Regression

training:
  meta_features:
    out_of_fold_predictions

validation:
  chronological_split

rationale:
  Different boosting algorithms may capture
  complementary interactions.

risks:
  - correlated base predictions
  - unnecessary complexity
  - calibration drift

required_data_properties:
  - tabular observations
  - binary target
```

---

# 7. Feature Mapper

After research, a separate mapper receives:

```text
ResearchFeature
+
database schema
+
column descriptions
+
relationships
```

It classifies each feature as:

```text
APPLICABLE
DERIVABLE
PARTIALLY_APPLICABLE
NOT_AVAILABLE
```

Example:

```text
Feature:
purchase regularity

Database:
customer_id
purchase_date

Result:
DERIVABLE
```

---

# 8. Model Mapper

The Model Mapper determines whether a proposed structure is compatible with the dataset.

Example:

```text
Structure:
Survival Model

Required:
event time
censoring information

Available:
churn timestamp ✓
observation end date ✓

Result:
APPLICABLE
```

Another example:

```text
Structure:
Sequential Transformer

Required:
ordered event sequence

Dataset:
monthly aggregated customer table only

Result:
NOT_APPLICABLE
```

---

# 9. Deterministic Validation

LLM mapping alone never approves a feature.

The system verifies:

```text
schema existence
data types
join path
join cardinality
temporal validity
point-in-time correctness
computability
leakage
```

For every feature at prediction time `T`:

```text
feature(T)

must depend only on

information available <= T
```

Any deterministic failure blocks the feature.

This extends the original pipeline's schema, formula, point-in-time, leakage, computability, and duplicate checks.

---

# 10. FeatureSpec

Only validated mappings become executable `FeatureSpec` objects.

```yaml
name: session_ratio_7_30

concept:
  engagement_deterioration

entity:
  user_id

lookback:
  30d

formula:
  sessions_7d / sessions_30d

source_feature:
  recent_activity_ratio

validation_status:
  verified
```

---

# 11. Candidate Generation

Valid concepts can be expanded using:

```text
Native Feature Grammar
OpenFE
Featuretools
Custom Generators
```

External tools generate candidates only.

They do not decide whether a feature is:

```text
safe
non-leaking
statistically useful
selected
```

That follows the same separation already defined in the original architecture.

---

# 12. Feature × Model Search

The system evaluates not only:

```text
Which features work?
```

but:

```text
Which feature portfolio
works best with which structure?
```

Example:

```text
Feature Portfolio A
     │
     ├── CatBoost
     ├── XGBoost
     └── Boosting Stack

Feature Portfolio B
     │
     ├── CatBoost
     ├── XGBoost
     └── Boosting Stack
```

This avoids assuming that a feature's value is independent of model architecture.

---

# 13. Coarse-to-Fine Evaluation

Large search spaces are reduced progressively.

```text
10,000 generated features
        ↓
6,000 valid
        ↓
3,000 unique
        ↓
500 coarse survivors
        ↓
100 fine evaluations
        ↓
20–30 candidate features
```

Model structures may similarly be reduced:

```text
10 researched structures
        ↓
5 applicable
        ↓
3 coarse survivors
        ↓
2 finalists
```

The original engine already uses this coarse-to-fine principle to control large hypothesis spaces.

---

# 14. Evaluation

Cheap screening may include:

```text
coverage
missingness
variance
stability
mutual information
cheap-model marginal gain
compute cost
```

Fine evaluation may include:

```text
chronological validation
repeated folds
out-of-fold predictions
incremental lift
stability
calibration
ablation
```

---

# 15. Portfolio Freeze

Feature and model selection must happen **before** the final holdout.

```text
Research
   ↓
Mapping
   ↓
Validation
   ↓
Experiments
   ↓
Feature + Model Selection
   ↓
PORTFOLIO FREEZE
   ↓
SEALED HOLDOUT
```

The sealed holdout measures final generalization.

It must not be used to decide which feature or model structure to keep.

---

# 16. Knowledge Base

The system remembers:

```text
problem

feature family
feature hypothesis
source
mechanism

model structure
architecture
model assumptions

mapping status
validation results
model lift
stability
similarity
compute cost
rejection reason
```

Example:

```text
Feature:
activity_ratio_7_30

Validation:
PASS

Experiment:
+0.006 AUC

Best structure:
CatBoost


Feature:
support_sentiment_trend

Mapping:
NOT_AVAILABLE


Structure:
XGBoost + CatBoost stack

Experiment:
+0.003 AUC over CatBoost

Cost:
2.4× training time
```

---

# 17. Research Feedback

Experiments guide future research.

Example:

```text
Observed:

Most trend-based engagement features
were highly redundant.

But volatility features produced
stable incremental lift.
```

Research feedback:

```text
Explore:

engagement irregularity
activity entropy
sequence changes
behavioral volatility
```

The Feature Seeker receives this **research feedback**, not raw database contents.

---

# 18. Stopping Conditions

Research stops when one of these occurs:

```text
requested feature count reached

requested structure count reached

novelty exhausted

max rounds reached

query budget reached

compute budget reached
```

Important:

```text
MAX_ROUNDS
```

does not mean:

```text
CONVERGED
```

The system should record the real stop reason.

---

# 19. Roles

## Feature Seeker — LLM

```text
research
analogy
mechanism reasoning
feature discovery
```

No approval authority.

---

## Structure Seeker — LLM

```text
research model architectures
explain assumptions
propose training protocols
```

No approval authority.

---

## Mapper — LLM

```text
semantic feature mapping
model applicability mapping
join-path proposals
```

No approval authority.

---

## Deterministic Validators

```text
schema
join integrity
temporal correctness
point-in-time safety
leakage
computability
```

Hard-block authority.

---

## Experiment Engine

Determines empirical value.

---

# 20. Repository Structure

```text
src/
│
├── contracts/
│   ├── problem.py
│   ├── search_contract.py
│   ├── research_feature.py
│   ├── model_structure.py
│   ├── mapped_feature.py
│   ├── model_spec.py
│   └── feature_spec.py
│
├── feature_seeker/
│   ├── seeker.py
│   ├── planner.py
│   ├── extractor.py
│   ├── analogy.py
│   └── sources/
│
├── structure_seeker/
│   ├── seeker.py
│   ├── planner.py
│   ├── extractor.py
│   └── sources/
│
├── mapping/
│   ├── feature_mapper.py
│   ├── model_mapper.py
│   ├── join_paths.py
│   └── verifier.py
│
├── generators/
│   ├── grammar.py
│   ├── openfe_adapter.py
│   └── featuretools_adapter.py
│
├── validation/
│   ├── schema.py
│   ├── temporal.py
│   ├── joins.py
│   ├── leakage.py
│   └── computability.py
│
├── evaluation/
│   ├── coarse.py
│   ├── fine.py
│   ├── stacking.py
│   ├── ablation.py
│   └── portfolio.py
│
├── knowledge/
│   └── research_kb.py
│
└── orchestration/
    ├── pipeline.py
    ├── budget.py
    └── stopping.py
```

---

# V1

Start with:

```text
1 Feature Seeker
1 Structure Seeker
1 Mapper

+
deterministic validators
+
experiment engine
+
knowledge base
```

Do not create many specialized runtime agents initially.

Use agents where reasoning is ambiguous.

Use deterministic software where rules can be enforced.

---

# Example

Request:

```text
Find 300 feature ideas and
3 strong structures for predicting
customer churn within 30 days.
```

Output:

```text
FEATURE RESEARCH

300 hypotheses
32 feature families


STRUCTURE RESEARCH

1. CatBoost baseline

2. XGBoost + CatBoost
   → OOF predictions
   → L2 Logistic Regression

3. Survival-based churn model
```

After mapping:

```text
Features

APPLICABLE       132
DERIVABLE         91
PARTIAL            28
NOT AVAILABLE      49


Structures

CatBoost                    APPLICABLE

XGB + CatBoost Stack        APPLICABLE

Survival Model              APPLICABLE
```

Then experiments determine which feature-model combination actually works.

---

# Core Principle

```text
Problem
   ↓
Research WHAT might matter
   ↓
Research HOW it might best be modeled
   ↓
Map both ideas to available data
   ↓
Validate deterministically
   ↓
Generate feature variants
   ↓
Run controlled experiments
   ↓
Select feature + model portfolio
   ↓
Freeze
   ↓
Evaluate once on sealed holdout
   ↓
Learn what to research next
```

The project is therefore not simply an automatic feature-engineering system.

It is an **ML Research Engine** for:

> **Feature Discovery → Structure Discovery → Applicability → Validation → Experimentation → Learning**