# Day 34 PM – Part D: AI-Augmented Week 6 Study Guide

## Prompt Used

`Create a structured Week 6 study guide for Saturday's assessment covering Logistic Regression, Decision Tree, Random Forest, AdaBoost, XGBoost, LightGBM, Voting, Stacking, SVM, KNN, K-Means, DBSCAN, and PCA. Include key concepts, common interview questions, and common code patterns.`

---

## Verified Week 6 Study Guide

### 1. Logistic Regression
- **Concept:** Linear classifier that models class probability.
- **Use:** Baseline for classification, interpretable model.
- **Interview question:** Why is it called regression if it is used for classification?
- **Code pattern:** `LogisticRegression(C=1.0, max_iter=1000)`

### 2. Decision Tree
- **Concept:** Greedy rule-based splitting model.
- **Use:** Interpretability and simple decision rules.
- **Interview question:** Why do deep trees overfit?
- **Code pattern:** `DecisionTreeClassifier(max_depth=5)`

### 3. Random Forest
- **Concept:** Bagging ensemble of decision trees.
- **Use:** Strong default for tabular data.
- **Interview question:** How does bagging reduce variance?
- **Code pattern:** `RandomForestClassifier(n_estimators=100)`

### 4. AdaBoost
- **Concept:** Sequential boosting where later learners focus on earlier mistakes.
- **Use:** Simple boosting baseline.
- **Interview question:** How is boosting different from bagging?
- **Code pattern:** `AdaBoostClassifier(n_estimators=100, learning_rate=0.5)`

### 5. XGBoost
- **Concept:** Regularized gradient boosting.
- **Use:** High-performance tabular modeling.
- **Interview question:** Why does XGBoost often outperform plain boosting?
- **Code pattern:** `xgb.XGBClassifier(n_estimators=200, max_depth=4)`

### 6. LightGBM
- **Concept:** Efficient histogram-based gradient boosting.
- **Use:** Large datasets and fast training.
- **Interview question:** What makes LightGBM faster than traditional boosting?
- **Code pattern:** `lgb.LGBMClassifier(n_estimators=200, num_leaves=31)`

### 7. Voting
- **Concept:** Combine predictions from multiple models.
- **Use:** Easy ensemble when models are complementary.
- **Interview question:** Difference between hard and soft voting?
- **Code pattern:** `VotingClassifier(estimators=[...], voting='soft')`

### 8. Stacking
- **Concept:** Meta-model learns from base-model predictions.
- **Use:** Stronger ensemble when models are diverse.
- **Interview question:** Why can stacking beat voting?
- **Code pattern:** `StackingClassifier(estimators=[...], final_estimator=LogisticRegression())`

### 9. SVM
- **Concept:** Margin-based classifier, can use kernels.
- **Use:** Small to medium datasets, especially with clean separation.
- **Interview question:** What do C and gamma control?
- **Code pattern:** `SVC(kernel='rbf', C=1.0, gamma='scale')`

### 10. KNN
- **Concept:** Predict using nearest neighbors.
- **Use:** Small datasets with local structure.
- **Interview question:** Why is scaling important for KNN?
- **Code pattern:** `KNeighborsClassifier(n_neighbors=5)`

### 11. K-Means
- **Concept:** Centroid-based clustering.
- **Use:** Compact groups when k is known or estimated.
- **Interview question:** Why can K-Means converge to a local optimum?
- **Code pattern:** `KMeans(n_clusters=3, n_init=10)`

### 12. DBSCAN
- **Concept:** Density-based clustering with noise detection.
- **Use:** Irregular cluster shapes and outlier detection.
- **Interview question:** What do eps and min_samples control?
- **Code pattern:** `DBSCAN(eps=0.5, min_samples=5)`

### 13. PCA
- **Concept:** Dimensionality reduction by preserving maximum variance.
- **Use:** Visualization, compression, and correlated feature reduction.
- **Interview question:** Why can PCA reduce accuracy even when it keeps 95% variance?
- **Code pattern:** `PCA(n_components=2)`

---

## Accuracy Check

The guide is accurate overall, but two things needed correction:
- PCA is not a model for prediction by itself; it is a preprocessing or visualization step.
- XGBoost and LightGBM need external libraries, so code snippets work only if those packages are installed.

---

## Missing Concepts Added

The AI guide often misses these important ideas:
- Scaling is essential for SVM and KNN.
- Cluster labels are arbitrary in K-Means, so ARI/NMI are better than raw label matching.
- Random Forest feature importance and permutation importance are not the same.
- PCA directions with highest variance are not always the most predictive for the target.

---

## Fast Saturday Revision Plan

1. Start with LR, DT, RF because they are the most common interview models.
2. Revise AdaBoost, XGBoost, LightGBM as the boosting family.
3. Revise Voting and Stacking as ensemble combinations.
4. Revise SVM and KNN with strong focus on scaling and hyperparameters.
5. End with K-Means, DBSCAN, and PCA for unsupervised learning and dimensionality reduction.
