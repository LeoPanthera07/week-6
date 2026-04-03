# Day 33 PM – Part C: Interview Ready

## Q1 – 100 features, 50 samples

**Best algorithms:**
1. **Naive Bayes** – Works well with high-dimensional sparse data, even with few samples.
2. **Logistic Regression** – Linear model, stable with regularization, good baseline.
3. **SVM Linear** – Effective in high dimensions with small samples if linearly separable.

**Algorithms that will fail:**
1. **KNN** – Curse of dimensionality makes distance meaningless with 100 features.
2. **Random Forest** – Needs more samples than features to learn stable splits.
3. **MLP** – Requires hundreds of samples per parameter, will overfit massively.

**Why:** With 50 samples and 100 features, the curse of dimensionality dominates.
Models that assume feature independence (Naive Bayes) or use regularization (LR, SVM) are most robust.

---

## Q2 – `model_selection_report()` function

```python
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score

def model_selection_report(X, y, models_dict):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"acc": make_scorer(accuracy_score), "f1": make_scorer(f1_score)}

    rows = []
    all_acc_scores = []

    for name, model in models_dict.items():
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
        acc_scores = cv_results["test_acc"]
        all_acc_scores.append(acc_scores)

        rows.append({
            "Model": name,
            "Acc mean": acc_scores.mean(),
            "Acc std": acc_scores.std(),
            "F1 mean": cv_results["test_f1"].mean(),
            "F1 std": cv_results["test_f1"].std()
        })

    df = pd.DataFrame(rows)

    # Statistical best model (paired t-test against all others)
    best_idx = np.argmax([scores.mean() for scores in all_acc_scores])
    p_values = [ttest_rel(all_acc_scores[best_idx], scores).pvalue 
                for scores in all_acc_scores if scores is not all_acc_scores[best_idx]]
    if all(p < 0.05 for p in p_values):
        df.loc[best_idx, "Statistical Best"] = "Yes"

    return df.sort_values("Acc mean", ascending=False)

# Usage example:
# models = {"LR": LogisticRegression(), "RF": RandomForestClassifier()}
# report = model_selection_report(X, y, models)
# print(report)
```

**Features:**
- 5-fold StratifiedKFold to preserve class balance
- Returns formatted DataFrame with mean/std
- Identifies statistical best model using paired t-test on CV folds
- Works with any model dict

---

## Q3 – SVM(RBF) train=1.0, test=0.52

**Diagnosis:** Classic **overfitting due to RBF kernel flexibility + insufficient regularization**.

**3 specific fixes:**

1. **Increase regularization:** `C=0.1` (default 1.0) reduces model complexity.
2. **Reduce gamma:** `gamma="scale"` or `gamma=0.001` prevents kernel from creating overly complex boundaries.
3. **Add cross-validation:** Use `GridSearchCV` on C and gamma with 5-fold CV instead of train/test split.

**Code fix:**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {"C": [0.1, 1], "gamma": ["scale", 0.001]}
svm = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5)
svm.fit(X_train, y_train)
```

**Why it happens:** RBF kernel can memorize training data perfectly (train=1.0) but fails to generalize to new patterns (test=0.52).
