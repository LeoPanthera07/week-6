# Day 34 AM – Session Logic

## Part A: Digit Classifier
- `load_digits` from sklearn is a compact 8x8 pixel version of MNIST (1797 samples, 64 features).
- StandardScaler is applied before both SVM and KNN because both models are distance-based.
- SVM GridSearchCV tunes C and gamma on scaled data to find the optimal margin and kernel width.
- KNN optimal K is found via 5-fold CV loop from k=1 to k=15.
- Confusion matrices highlight misclassified digit pairs visually.
- Off-diagonal analysis with `np.fill_diagonal(cm, 0)` cleanly identifies most-confused pairs.

## Part B: FAISS
- FAISS `IndexFlatL2` gives exact same results as sklearn KNN on small data.
- Speed advantage only appears on large datasets (100k+ vectors).
- The notebook includes try/except so it gracefully handles cases where FAISS is not installed.

## Part C Logic
- Q1: SVM maximizes margin (only support vectors matter), LR minimizes log-loss (all points matter).
- Q2: KNN from scratch loops over test points, computes Euclidean distance manually with numpy broadcasting.
- Q3: Missing StandardScaler causes salary (50k-200k) to dominate age (20-60) in RBF distance, breaking the model.

## Part D Logic
- Visualization shows C effect on boundary smoothness vs tightness.
- Kernel trick analogy: marble-lifting in 3D is verified and improved with dot-product explanation.
