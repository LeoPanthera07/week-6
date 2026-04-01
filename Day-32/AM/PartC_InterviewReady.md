# Day 32 – Part C: Interview Ready

## Q1 – Bias-Variance Tradeoff

A single Decision Tree with no depth restriction can memorize the training data.
It learns very specific rules and splits every possible pattern, giving:
- Low bias (fits training data well)
- High variance (small changes in data change the tree a lot)

In this assignment, a Decision Tree with max_depth=4 achieves:
- Accuracy: 0.8767
- ROC-AUC: 0.9555

Random Forest averages 200 trees (from RandomizedSearchCV best params), each built on a bootstrapped sample with log2 random features at each split.
Averaging cancels out the variance from individual noisy trees, giving:
- Accuracy: 0.9483
- ROC-AUC: 0.9907

**Diagram (ASCII):**

```
Variance
  ^
  |  DT (max_depth=4)
  |         *
  |  RF (200 trees)
  |   *
  |________________________> Depth / complexity
```

Bagging reduces variance by:
1. Training each tree on a different bootstrapped sample
2. Averaging outputs so individual wrong predictions cancel out

## Q2 – Overfitting Curve Function

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def plot_overfitting_curve(X, y, max_depths):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=32, stratify=y
    )
    train_scores = []
    test_scores = []
    for depth in max_depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=32)
        clf.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
        test_scores.append(accuracy_score(y_test, clf.predict(X_test)))
    plt.figure(figsize=(6, 4))
    plt.plot(max_depths, train_scores, marker="o", label="Train")
    plt.plot(max_depths, test_scores, marker="o", label="Test")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Overfitting Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()
    optimal = max_depths[test_scores.index(max(test_scores))]
    print("Optimal max_depth:", optimal)
```

From our results:
- DT at max_depth=4 gives test accuracy 0.8767 and is not overfitting.
- Deeper trees will increase train accuracy but test accuracy will plateau and eventually drop.
- The optimal depth is where test accuracy is highest before it starts declining.

## Q3 – Debug: Random Forest train = test = 0.95

This is generally not a problem. It can mean:
- The model generalizes well and is neither underfitting nor overfitting.
- The dataset is clean and well-structured, which is the case here.

In our dataset, RF achieves test accuracy 0.9483 which is close to train accuracy.
This is acceptable and expected behavior for a well-tuned ensemble.

However, to rule out issues you should check:
- Whether any feature is a data-leakage proxy for the target
- Whether the test set is too small to give stable estimates
- Whether stratified splitting was used (we used stratify=y, so class balance is preserved)

In this case, all three checks pass.
