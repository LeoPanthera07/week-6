# Day 34 AM – Session Logic

## Part A
- Load Iris and remove labels from the feature matrix during clustering.
- Scale features so all dimensions contribute fairly.
- Run K-Means with `k=3` because Iris has three known species.
- Compare K-Means clusters with the real species only after clustering.
- Use ARI and NMI because raw cluster labels are arbitrary and cannot be compared directly by numeric value.
- Use PCA to plot true labels and predicted cluster labels side by side.
- Run DBSCAN to check whether a density-based method also finds similar groups.

## Part B
- Use AgglomerativeClustering with `n_clusters=3` as the self-study extension.
- Compare its ARI with K-Means on the same scaled Iris data.
- Create a dendrogram using Ward linkage to visualize hierarchical merges.

## Part C
- Explain greediness as local optimization without global guarantee.
- Implement K-Means from scratch with NumPy using distance, assignment, and centroid update steps.
- Interpret silhouette score 0.25 as weak clustering and suggest practical next checks.

## Part D
- Use a fruit analogy to explain K-Means, DBSCAN, and Hierarchical Clustering in simple language.
- Verify what the analogy gets right.
- Improve it by pointing out failure cases and parameter sensitivity.
