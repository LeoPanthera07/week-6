# FAISS vs sklearn KNN – Comparison Findings

## What is FAISS?
FAISS (Facebook AI Similarity Search) is a library for efficient similarity search over large vector datasets.
It supports both exact and approximate nearest neighbor search.
It is used at Instagram, Spotify, and Pinterest for recommendation systems and RAG pipelines.

## How it differs from sklearn KNN
| Aspect | sklearn KNN | FAISS |
|---|---|---|
| Search type | Exact brute-force | Exact (IndexFlatL2) or Approximate (IVF, HNSW) |
| Speed (small data) | Fast | Similar or slightly slower (overhead) |
| Speed (large data, >1M) | Very slow | 100x+ faster |
| GPU support | No | Yes (faiss-gpu) |
| Integration | sklearn pipeline | Manual numpy |

## Results on Digits Dataset (1000 queries)
After running the notebook:
- Both methods should give identical accuracy (exact search with IndexFlatL2).
- Speed difference is small on 1,797 samples.
- FAISS becomes significantly faster when dataset is millions of vectors.

## Why It Matters
FAISS powers the retrieval step in modern RAG (Retrieval Augmented Generation) systems.
When a language model needs to find relevant documents from millions of embeddings, FAISS returns results in milliseconds.

## When to Use FAISS Over sklearn KNN
- Dataset > 100k vectors
- Need GPU-accelerated search
- Building recommendation or semantic search systems
- Real-time prediction (< 10ms latency requirement)
