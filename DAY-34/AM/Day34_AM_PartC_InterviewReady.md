# Day 34 AM – Part C: Interview Ready

## Q1 – Why is K-Means called greedy?

K-Means is called greedy because at each iteration it makes the locally best choice:
- assign each point to the nearest centroid
- recompute centroids using current assignments

It does not look ahead to check whether a different assignment today could produce a better global clustering later.
So yes, K-Means can get stuck in a local minimum.

### How KMeans++ helps
KMeans++ chooses initial centroids more carefully.
Instead of picking all starting points randomly, it spreads them out.
This reduces the chance of poor initialization and usually gives faster convergence and better final clusters.

---

## Q2 – K-Means from scratch using NumPy

```python
import numpy as np

def kmeans(X, k, max_iter=100):
    rng = np.random.default_rng(42)
    centroids = X[rng.choice(len(X), size=k, replace=False)]
    for _ in range(max_iter):
        distances = np.sqrt(((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids
```

### Logic
- Randomly initialize `k` centroids.
- Compute distance from each point to each centroid.
- Assign each point to the nearest centroid.
- Recompute centroids as mean of assigned points.
- Stop when centroids stop changing.

---

## Q3 – K=5 gives silhouette score 0.25. Is that good?

A silhouette score of **0.25** is generally weak.
It suggests clusters are not very well separated and many points may lie close to cluster boundaries.

### What to investigate next
- Try different values of `k` such as 2, 3, 4, 6.
- Plot silhouette score vs `k`.
- Visualize clusters using PCA or t-SNE.
- Check whether scaling was applied.
- See whether the data really has 5 natural groups.
- Compare with DBSCAN or hierarchical clustering.

A low silhouette score does not always mean the clustering is useless, but it does mean you should not trust `k=5` immediately.
