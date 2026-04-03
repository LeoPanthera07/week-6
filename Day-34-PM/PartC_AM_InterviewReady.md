# Day 34 AM – Part C: Interview Ready

## Q1 – SVM vs Logistic Regression: Both find linear boundaries. What's different?

**Logistic Regression** finds a boundary that maximizes the log-likelihood of the data.
It minimizes the cross-entropy loss, which considers all data points and their probabilities.

**SVM (Linear)** finds a boundary that maximizes the **margin** — the gap between the boundary and the closest data points from each class (support vectors).
It only cares about the points near the boundary, ignoring the rest.

**Key difference:**
- LR: minimizes loss over all points, gives probabilities.
- SVM: maximizes margin, only depends on support vectors.

**When to prefer each:**

| Situation | Prefer |
|---|---|
| Need probabilities or calibrated output | Logistic Regression |
| Clean margin separation, sparse data | SVM Linear |
| Text classification (TF-IDF) | SVM Linear (faster, better margin) |
| Noisy, overlapping classes | Logistic Regression (softer decision) |
| Interpretable coefficients | Logistic Regression |

---

## Q2 – KNN from Scratch (NumPy only)

```python
import numpy as np

def knn_from_scratch(X_train, y_train, X_test, k):
    y_train = np.array(y_train)
    predictions = []
    for x in X_test:
        diffs = X_train - x
        distances = np.sqrt((diffs ** 2).sum(axis=1))
        neighbor_indices = np.argsort(distances)[:k]
        neighbor_labels = y_train[neighbor_indices]
        majority = np.bincount(neighbor_labels).argmax()
        predictions.append(majority)
    return np.array(predictions)
```

**Usage:**
```python
y_pred = knn_from_scratch(X_train_sc, y_train, X_test_sc, k=3)
print("Accuracy:", (y_pred == y_test).mean())
```

**Logic:**
- For each test point, compute Euclidean distance to all training points.
- Sort distances and take the k closest neighbors.
- Return the majority class label among those k neighbors.

---

## Q3 – Debug: SVM 0.50 accuracy (random performance)

**Root cause:** Features are not scaled.

```python
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)  # salary (50k-200k), age (20-60)
```

`salary` has values in the range 50,000–200,000.
`age` has values in the range 20–60.

RBF kernel computes distance using Euclidean distance.
Because salary dominates the distance calculation by a factor of ~3,000, the model effectively ignores age and cannot learn the real boundary.
The result is near-random performance (0.50 on binary classification).

**3 specific fixes:**

1. **Scale features before training:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0))
])
svm.fit(X_train, y_train)
```

2. **Tune C and gamma with GridSearchCV** after scaling:
```python
param_grid = {"svm__C": [0.1, 1, 10], "svm__gamma": ["scale", 0.001, 0.01]}
gs = GridSearchCV(svm, param_grid, cv=5)
gs.fit(X_train, y_train)
```

3. **Check class balance**: if 0.50 is actually the majority class proportion, the problem may be class imbalance, not just scaling. Add `class_weight="balanced"` to SVC.
