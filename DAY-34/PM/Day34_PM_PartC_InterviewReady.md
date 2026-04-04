# Day 34 PM – Part C: Interview Ready

## Q1 – Complete ML pipeline for 1000 samples and 200 features

A practical pipeline starts with understanding the data: check missing values, feature types, target balance, and whether the problem is classification or clustering.
Then split the data into train and test sets before doing any transformation that learns from the data.

Next, choose preprocessing based on feature properties.
If many features are correlated, PCA is a good option to reduce redundancy and noise.
If the problem is supervised classification, I would test at least three models from this week:
- **Logistic Regression** as a simple interpretable baseline
- **Random Forest** as a strong tabular-data model that captures non-linear interactions
- **SVM** when the boundary may be complex and the dataset is not too large

Then I would compare models with 5-fold cross-validation on the training set, tune the best one, and evaluate once on the test set.
If interpretability matters, I might prefer Logistic Regression or a shallow Decision Tree even if Random Forest is slightly better.
For deployment, I would save the whole preprocessing + model pipeline, monitor accuracy drift, and retrain when the data distribution changes.

---

## Q2 – `weekly_model_comparison(X, y)` function

```python
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def weekly_model_comparison(X, y, use_pca=False, n_components=0.95):
    models = {
        'LR': LogisticRegression(max_iter=1000, random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale'),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)

    rows = []
    for name, model in models.items():
        steps = [('scaler', StandardScaler())]
        if use_pca:
            steps.append(('pca', PCA(n_components=n_components)))
        steps.append(('model', model))
        pipe = Pipeline(steps)
        scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
        rows.append({
            'Model': name,
            'Mean Accuracy': scores.mean(),
            'Std': scores.std()
        })
    return pd.DataFrame(rows).sort_values('Mean Accuracy', ascending=False)
```

---

## Q3 – PCA keeps 95% variance but accuracy drops from 0.92 to 0.85. Why?

Three common reasons:

1. **Variance is not the same as predictive power.**
PCA keeps directions with large variance, but the target signal may lie in lower-variance directions that get removed.

2. **Important feature interactions may be lost.**
After PCA, components are linear combinations of original features, so some task-specific structure may become harder for the model to use.

3. **The model may not need PCA.**
Tree-based models like Random Forest often handle high-dimensional raw features well, so PCA can actually remove useful information instead of helping.

A smaller drop can be acceptable if PCA gives speed, simpler deployment, or lower storage cost, but a drop from 0.92 to 0.85 is large enough to investigate carefully.
