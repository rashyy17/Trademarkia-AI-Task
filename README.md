# Trademarkia - Semantic Search (20 Newsgroups)

Lightweight semantic search with fuzzy clustering and a cluster-aware semantic cache.

Quickstart

1. Create and activate a venv (macOS zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Build the index and clusters (this will load the 20 newsgroups corpus and create embeddings):

```bash
python3 scripts/build_index.py
```

3. Run the FastAPI app with uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints

- POST /query  - JSON {"query": "..."}
- GET /cache/stats
- DELETE /cache

Notes

- Embedding model: sentence-transformers/all-MiniLM-L6-v2 (small, fast, good cosine embeddings). Chosen for speed and reasonable quality for this assignment.
- Vector store: FAISS (IndexFlatIP on L2-normalized vectors) for fast approximate nearest-neighbour search.
- Clustering: Gaussian Mixture Model (soft assignments) chosen to produce per-document distributions rather than hard labels. The number of clusters is chosen via BIC over a range.
- Cache: in-memory, cluster-aware, embedding-based cache (no Redis). Tunable similarity threshold in `app/cache.py`.

Smoke test

After building the index, you can run a quick smoke test (no server needed) to validate the end-to-end flow:

```bash
python3 scripts/smoke_test.py
```

If the smoke test shows misses then hits for similar queries, the cache is functioning.

Docker

A `Dockerfile` is included if you want to containerize the service. See the file for details.

Submission

Provide the repository link and grant access to recruitments@trademarkia.com as requested.