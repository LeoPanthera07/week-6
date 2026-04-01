# Day 32 PM – Session Logic

## Dataset Design
Synthetic insurance claims dataset (3000 records) with features that realistically drive fraud:
- claim_amount: larger claims are more likely to be inflated
- num_claims_past: repeat claimants are higher risk
- days_since_policy: new policies with big claims are suspicious
- num_witnesses: fewer witnesses → more suspicious
- claim_hour: night hours (< 6 AM) are suspicious
- vehicle_age: older vehicles with large claims are suspicious
- deductible_ratio: very low deductible relative to claim is suspicious
- policy_premium: context feature

## Part A Logic

### Decision Tree (max_depth=5)
A shallow tree provides interpretable rules.
Rules are extracted by tracing tree paths to leaves with the highest sample count.
The extracted rules show the most common split patterns that separate fraud from non-fraud.

### Random Forest (tuned for Recall)
RandomizedSearchCV optimises for recall because FN cost = 10 × FP cost.
Missing a fraud case is 10 times more expensive than a false alarm.
The class_weight parameter is included in the search grid so the model can penalise false negatives directly.
oob_score=True is set so we also get a free OOB estimate alongside the CV results.

### Cost-Sensitive Evaluation
Total cost = FP × 1 + FN × 10.
This formula makes the evaluation reflect the actual business cost, not just accuracy.
RF should show a lower total cost because it achieves higher recall.

## Part B Logic
Boosting is fundamentally different from bagging:
- Bagging: parallel trees, reduce variance
- Boosting: sequential trees, reduce bias and errors from previous stage

## Part C Logic
- Q1: Use minimal trees after accuracy plateaus; production latency is a real constraint.
- Q2: compare_models() uses StratifiedKFold to preserve class balance; cross_validate returns mean and std natively.
- Q3: Missing random_state + too few trees = unstable importance rankings.

## Part D Logic
OOB error is a reliable free cross-validation proxy when data is i.i.d. and balanced.
It diverges from test error under distribution shift, time-series structure, or severe class imbalance.
