# Trademarkia - Semantic Search (20 Newsgroups)

This repository implements a lightweight semantic search system with fuzzy clustering and a cluster-aware semantic cache. It satisfies the assignment components and includes scripts to reproduce experiments.

## What's included
- `app/` - FastAPI app and core modules
  - `vectorstore.py` - embedding pipeline using `sentence-transformers` + FAISS index (normalized vectors, IndexFlatIP)
  - `clustering.py` - PCA + GaussianMixtureModel for fuzzy clustering; saves `StandardScaler` + PCA + GMM
  - `cache.py` - in-memory cluster-aware semantic cache (no Redis)
  - `main.py` - FastAPI server (POST /query, GET /cache/stats, DELETE /cache)
- `scripts/` - helper scripts
  - `build_index.py` - build embeddings, FAISS index, fit clusters
  - `smoke_test.py` - small end-to-end check (cache miss->add->hit behavior)
  - `cluster_report.py` - cluster inspections: top terms, exemplars, boundary docs
  - `threshold_sweep.py` - sweep cache similarity threshold and record hit/miss behavior
- `data/` - generated artifacts (index, embeddings, meta, cluster_report, threshold_sweep.csv)
- `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `README.md`

## Design decisions (brief)

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` — chosen for a good speed/quality tradeoff for semantic retrieval on a medium corpus.
- Vector store: FAISS `IndexFlatIP` over L2-normalized vectors. Inner product equals cosine similarity after normalization; IndexFlatIP is simple and reliable for a dataset of this size.
- Clustering: PCA (to pca_dim=50) followed by `GaussianMixture`. GMM returns soft probabilities per document (a distribution), which satisfies the fuzzy clustering requirement. Number of clusters chosen by BIC sweeping k∈[8,25]; BIC selected 8 for this run.
- Semantic cache: in-memory, partitioned by dominant cluster (entries stored per dominant cluster) and queried only within top-K clusters for the query. Lookup uses cosine similarity (dot product on normalized embeddings) and a tunable threshold `SIMILARITY_THRESHOLD` in `app/cache.py`.

## Reproducibility / How I validated
1. Create venv and install deps
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Build index and clusters
   ```bash
   python3 scripts/build_index.py
   ```
3. Run smoke test
   ```bash
   python3 scripts/smoke_test.py
   ```
4. Run threshold sweep (produces `data/threshold_sweep.csv`)
   ```bash
   python3 scripts/threshold_sweep.py
   ```
5. Run cluster report (produces `data/cluster_report.txt`)
   ```bash
   python3 scripts/cluster_report.py
   ```

## Cache threshold exploration summary

I ran a sweep of thresholds in [0.5, 0.95] and recorded hit rates; results are in `data/threshold_sweep.csv`. Example finding: two clearly related queries ("What is the debate around gun control?" and "Discussion on firearms legislation and rights") had cosine similarity ≈ 0.655 while sharing the same dominant cluster — therefore a conservative default threshold like 0.86 produces zero reuse for such paraphrases. Lower thresholds (≈0.6–0.7) increase hit rate considerably while risking occasional less-ideal reuse. See the CSV for exact numbers and pick the threshold that matches your risk tolerance.

## Part 2 evidence (clusters)
- BIC selected 8 clusters. `scripts/cluster_report.py` shows top TF-IDF terms per cluster and exemplar documents; it also lists boundary documents by entropy to show where the model is uncertain.

## Remaining optional items
- Unit tests (minimal tests can be added); I can add them on request.
- Cache persistence across restarts (optional enhancement).
- Docker image build/test (I can run a local docker build and smoke test if you want).

## Submission
Push this repo to GitHub and ensure `recruitments@trademarkia.com` has access. The service can be started with:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

If you want, I will now run the threshold sweep and unit tests and attach the CSV + a brief verdict on the best threshold. (I can also add a short visual plot.)
