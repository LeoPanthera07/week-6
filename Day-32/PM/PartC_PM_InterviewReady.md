# Day 32 PM – Part C: Interview Ready

## Q1 – 1000 trees vs 100 trees: Should you use 1000?

No, not automatically.

When accuracy stops improving beyond 100 trees, using 1000 trees gives the same prediction quality at a much higher cost.

**Tradeoffs:**

| Dimension | 100 Trees | 1000 Trees |
|---|---|---|
| Compute cost | Low | ~10x higher |
| Prediction latency | Fast | Slow (matters in real-time APIs) |
| Marginal improvement | Baseline | Negligible after ~200 trees |
| Production deployment | Easy to serve | Requires more memory and infra |

**Practical rule:**
Use the smallest number of trees where test score stabilises.
In most cases this is between 100 and 300 trees.
For a fraud detection API scoring claims in real time, 1000 trees could make response times unacceptable.

---

## Q2 – `compare_models()` function

```python
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score

def compare_models(X, y, models_dict):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "f1":       make_scorer(f1_score, zero_division=0)
    }
    rows = []
    for name, model in models_dict.items():
        start = time.time()
        results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        elapsed = time.time() - start
        rows.append({
            "Model":        name,
            "Acc mean":     round(results["test_accuracy"].mean(), 4),
            "Acc std":      round(results["test_accuracy"].std(), 4),
            "F1 mean":      round(results["test_f1"].mean(), 4),
            "F1 std":       round(results["test_f1"].std(), 4),
            "Train time s": round(elapsed, 3)
        })
    return pd.DataFrame(rows)
```

**Usage:**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Logistic Reg":  LogisticRegression(max_iter=1000)
}

result_df = compare_models(X, y, models)
print(result_df)
```

---

## Q3 – Debug: Feature importances differ on every run

**Root cause:**
`RandomForestClassifier(n_estimators=10)` is called without `random_state`.

Without `random_state`, each call uses a different internal random seed.
With only 10 trees, the forest is very small and highly sensitive to which random bootstrapped samples and feature subsets are selected.
So each run builds a completely different set of 10 trees, producing very different feature importance rankings.

**Fix:**
```python
rf1 = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)
rf2 = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)
# Now feature_importances_ will be identical
```

**Lesson:**
Always set `random_state` for reproducibility.
Also, 10 trees is far too few for stable importance estimates.
Using 100+ trees makes importances much more stable even without a fixed seed.
