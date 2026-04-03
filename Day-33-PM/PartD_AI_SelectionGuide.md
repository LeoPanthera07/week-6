# Day 33 PM – Part D: AI-Augmented Algorithm Selection Guide

## Prompt Used
`Create a decision tree for algorithm selection given dataset characteristics: size (small <1k, medium 1k-100k, large >100k), features (numerical, categorical, text, images), linearity (linear, non-linear), labels (binary, multi-class). Only use algorithms from this week: Logistic Regression, Decision Tree, Random Forest, SVM RBF, SVM Linear, KNN, Naive Bayes, MLP.`

---

## AI Response (Summary)

```
Dataset Size?
├── Small (<1k)
│   ├── Text data → SVM Linear or Naive Bayes
│   ├── Numerical → KNN or Decision Tree
│   └── Categorical → Naive Bayes
├── Medium (1k-100k)
│   ├── Linear → Logistic Regression or SVM Linear
│   └── Non-linear → Random Forest or SVM RBF
└── Large (>100k)
    ├── Images → MLP
    └── Tabular → Random Forest
```

---

## Verification Against Experience

**Accurate recommendations:**
- SVM Linear and Naive Bayes for text: ✓ Linear SVM is classic for TF-IDF text data.
- KNN and Decision Tree for small numerical: ✓ Both work well on small clean data.
- Random Forest for medium non-linear: ✓ Best general-purpose choice.
- Logistic Regression for linear: ✓ Good baseline.

**Edge cases AI missed:**
1. **High-dimensional sparse data (e.g. 100k features, 1k samples):** SVM Linear or Naive Bayes, *not* KNN (curse of dimensionality).
2. **Severe class imbalance:** All models need `class_weight` or `scale_pos_weight` parameter.
3. **Time-series data:** None of these 8 models handle temporal structure; use lagged features.
4. **Mixed data types:** Decision Tree handles without encoding; others need preprocessing.

---

## Improved Algorithm Selection Guide

```
1. DATASET SIZE
├── Small (<1k)
│   ├── Text/sparse → SVM Linear or Naive Bayes
│   ├── Tabular numerical → KNN (k=3-5) or Decision Tree (max_depth=4)
│   └── Categorical → Naive Bayes
├── Medium (1k-100k)
│   ├── Linear patterns → Logistic Regression or SVM Linear
│   └── Non-linear → Random Forest (n_estimators=100) or SVM RBF (C=1, gamma=scale)
└── Large (>100k)
    ├── Tabular → Random Forest or Logistic Regression
    └── Images → MLP (hidden_layers=2x3x)

2. SPECIAL CASES (overrides above)
├── High dimensions (>10k features) → SVM Linear or Naive Bayes
├── Class imbalance → Add class_weight="balanced" to all tree-based models
├── Interpretability needed → Decision Tree or Logistic Regression
└── Very simple baseline → Logistic Regression
```

**Personal note:** Random Forest is the "default choice" for most tabular classification problems unless you have a specific reason to prefer something else.
