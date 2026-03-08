"""
FastAPI app wiring vectorstore, clustering, and semantic cache.

Endpoints implemented as per assignment:
- POST /query
- GET /cache/stats
- DELETE /cache

Behavior:
- On startup attempt to load existing index/clusters; if missing, instruct user to run scripts/build_index.py
- On POST /query: embed query, get cluster distribution, check cache. On miss, run vector search and store result in cache.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os

from .vectorstore import VectorStore
from .clustering import Clusterer
from .cache import SemanticCache

app = FastAPI(title="Trademarkia Semantic Search")

# singletons
VECTORSTORE = VectorStore()
CLUSTERER = Clusterer()
CACHE = SemanticCache()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class QueryRequest(BaseModel):
    query: str


@app.on_event("startup")
def startup_event():
    # Load index and clusters if available. Building can be done via scripts/build_index.py
    try:
        VECTORSTORE.load()
    except Exception as e:
        print("VectorStore index not found. Please run `python3 scripts/build_index.py` to build the index.")
    try:
        CLUSTERER.load()
    except Exception:
        print("Clusters not found. Please run `python3 scripts/build_index.py` to build clusters.")


@app.post("/query")
def query(req: QueryRequest):
    q = req.query
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query must be non-empty")
    emb = VECTORSTORE.embed(q)
    try:
        cluster_dist = CLUSTERER.predict_proba(np.array([emb]))[0]
        dominant = int(np.argmax(cluster_dist))
    except Exception:
        # If clusterer not ready, fallback to uniform
        cluster_dist = np.ones(1) / 1
        dominant = 0

    # check cache
    hit = CACHE.lookup(emb, cluster_dist)
    if hit:
        entry, sim = hit
        return {
            "query": q,
            "cache_hit": True,
            "matched_query": entry.query,
            "similarity_score": sim,
            "result": entry.result,
            "dominant_cluster": entry.dominant_cluster,
        }

    # cache miss: run vector search
    results = VECTORSTORE.search(q, k=3)
    # For simplicity, return the top document as `result` (could be aggregated)
    if len(results) == 0:
        result_text = ""
    else:
        idx, score, text = results[0]
        result_text = text[:1000]

    # add to cache
    CACHE.add(q, emb, result_text, cluster_dist)
    return {
        "query": q,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result_text,
        "dominant_cluster": dominant,
    }


@app.get("/cache/stats")
def cache_stats():
    return CACHE.stats()


@app.delete("/cache")
def flush_cache():
    CACHE.flush()
    return {"status": "flushed"}
