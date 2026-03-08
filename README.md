README — Trademarkia semantic search (20 Newsgroups)

Hi — this is the project I built for the Trademarkia assignment: a lightweight semantic search system over the 20 Newsgroups dataset. I focused on three things you asked for:

- meaningful, reproducible embeddings + a simple FAISS vector store;
- fuzzy (soft) clustering so documents have distributions over topics; and
- a cluster-aware semantic cache that avoids recomputing answers for semantically equivalent queries.

I wrote this to be reproducible on a laptop. Below you'll find the design rationale, how to run everything locally, what the API does, and where to look for the experiments and evidence I mention in the write-up.

Why this design (short)
- Embeddings: I use `sentence-transformers/all-MiniLM-L6-v2`. It gives good semantic quality for retrieval tasks while being very fast and small enough to run locally.
- Vector store: FAISS `IndexFlatIP` over L2-normalized vectors. This makes cosine-similarity queries fast and reliable for this dataset size and is easy to persist/load.
- Clustering: PCA -> GaussianMixture. GMM provides soft assignments (probability distributions per document), which is what you requested instead of hard labels. I select the number of clusters using BIC across a reasonable range — this is evidence-driven, not arbitrary.
- Cache: an in-memory, cluster-partitioned cache (no Redis). Each cache entry stores the query, its normalized embedding, the result, and the cluster distribution. At lookup we consider only the query's top-K clusters and compute cosine-similarity against cached embeddings there. If similarity >= threshold (tunable) we return a cache hit.

Quick reproduction (macOS / zsh)

1) Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Build the index and clusters (this downloads the 20 Newsgroups data and computes embeddings; ~few minutes depending on hardware):

```bash
python3 scripts/build_index.py
```

3) Optional: run a quick smoke test that demonstrates cache behavior (no server required):

```bash
python3 scripts/smoke_test.py
```

4) Start the API server (FastAPI + uvicorn):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API endpoints

- POST /query — accepts JSON {"query": "<natural language query>"} and returns:

```json
{
	"query": "...",
	"cache_hit": true|false,
	"matched_query": "...", // when cache_hit==true
	"similarity_score": 0.91, // when cache_hit==true
	"result": "...",
	"dominant_cluster": 3
}
```

- GET /cache/stats — returns current cache metrics:

```json
{
	"total_entries": 42,
	"hit_count": 17,
	"miss_count": 25,
	"hit_rate": 0.405
}
```

- DELETE /cache — flushes the cache and resets stats.

Where to look for the pieces I mentioned

- `app/vectorstore.py` — embedding model usage, FAISS index creation & persistence. I save normalized embeddings to `data/embeddings.npy` to speed up clustering.
- `app/clustering.py` — PCA + GaussianMixture pipeline, scaler persistence, BIC-based selection of cluster count.
- `app/cache.py` — the cluster-aware semantic cache (tunable threshold and top-K clusters to search).
- `app/main.py` — FastAPI endpoints wiring everything together.
- `scripts/build_index.py` — full pipeline to load data, embed, build FAISS and fit GMM.
- `scripts/cluster_report.py` — produces `data/cluster_report.txt` showing top TF-IDF terms per cluster, exemplar docs, and boundary cases (high-entropy docs).
- `scripts/threshold_sweep.py` — runs a sweep across similarity thresholds and records hit/miss behavior to `data/threshold_sweep.csv` (this is the experiment you asked for in Part 3).
- `scripts/plot_clusters.py` — creates a sampled t-SNE visualization saved to `analysis/cluster_tsne.png` (useful for a reviewer's quick intuition about cluster separability).

Design & implementation notes (so a reviewer understands my tradeoffs)

- Preprocessing: I removed headers/footers/quotes from the raw 20 Newsgroups posts (scikit-learn loader supports that) and dropped extremely short posts (<20 characters). The motivation is to avoid non-topical metadata (email addresses, paths) and remove very low-content documents that add noise to embeddings.
- Embedding choices: I purposely selected a small, fast SentenceTransformer so this can run on a standard laptop within a reasonable time. Larger models would improve retrieval quality but at a heavy cost to reproducibility.
- Clustering choices: GMM + PCA is stable for this size, and BIC gives an evidence-based number of clusters. The clusters are soft — you get a per-document distribution. I included `scripts/cluster_report.py` so you can inspect clusters and boundary cases yourself.
- Cache choices: The cache is deliberately simple and written from first principles — no Redis or third-party cache. It partitions entries by dominant cluster and only searches the query’s top-K clusters to limit comparisons. The similarity threshold is the single most important tunable; `app/cache.py` contains the default and comments on how to explore it.

Experiments & recommended threshold

I included `scripts/threshold_sweep.py` which simulates a set of queries and computes hit rates for thresholds from 0.5 to 0.95. In my runs, two clearly related queries had cosine similarity ≈ 0.65; therefore the conservative default (0.86) yields very few false positives but also low hit rates. A value around 0.6–0.7 is a reasonable tradeoff if you want more reuse. The sweep CSV (`data/threshold_sweep.csv`) contains the exact numbers; plot it and pick the threshold based on the hit/precision tradeoff you prefer.

Tests and reproducibility

- I included a couple of minimal unit tests under `tests/` (run with `python -m pytest`). They cover basic cache semantics and validate that embeddings return a 1-D numpy array.
- The build process is deterministic given the same model download; to save time, I persist embeddings (so you don't re-encode if you run builds repeatedly).

Docker

There is a `Dockerfile` and a `docker-compose.yml` if you prefer a containerized run. I didn't push large binary data to the repo (the `data/` folder is in `.gitignore`) — if you want a fully self-contained image with the index embedded, I can produce one and upload it as a release artifact.

Submission notes for reviewers

- The repository is at: https://github.com/rashyy17/Trademarkia-AI-Task
- Please add `recruitments@trademarkia.com` as a collaborator so the submission form can access the repo.
- The most important artifacts for your review are `submission.md` (short summary), `analysis/cluster_summary.md`, `analysis/cluster_tsne.png`, `data/cluster_report.txt`, and `data/threshold_sweep.csv`.

If you want this README tailored into a single-page PDF summary for the submission form I can generate that too.

Questions or next steps

If you want I can:
- produce a small plot (PNG) of threshold vs hit rate and add it under `analysis/` (I already have a CSV);
- build a Docker image and verify the container starts on port 8000;
- add cache persistence across restarts or a per-partition FAISS index for cache lookups for extra scalability.

Thanks — if you want any phrasing changed in this README (shorter, more formal, or more conversational for Loom narration), tell me and I'll update it and push.

# Quick commands summary (copy-paste)

```bash
# prepare venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# build artifacts (embeddings, FAISS, clusters)
python3 scripts/build_index.py

# run smoke test
python3 scripts/smoke_test.py

# run threshold sweep
python3 scripts/threshold_sweep.py

# start API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# basic curl examples (replace with your queries)
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d '{"query":"What is the debate around gun control?"}'
curl http://127.0.0.1:8000/cache/stats
curl -X DELETE http://127.0.0.1:8000/cache
```

---
Last updated: March 2026


Docker

A `Dockerfile` is included if you want to containerize the service. See the file for details.

Submission

Provide the repository link and grant access to recruitments@trademarkia.com as requested.