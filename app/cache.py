"""
Cluster-aware semantic cache implemented from first principles.

Design summary:
- Each cache entry stores: query text, embedding (normalized), result (string), dominant_cluster (int), cluster_distribution (np.array).
- Lookup strategy:
  1. Compute query embedding and cluster distribution.
  2. For efficiency, only compare against cached entries whose dominant cluster is in the top-N clusters of the query (we use top 3 by probability). This partitions the cache and reduces comparisons as cache grows.
  3. Compute cosine similarity (dot product on normalized vectors) to nearest cached embeddings in those partitions.
  4. If similarity >= SIMILARITY_THRESHOLD, consider it a cache hit and return the matched entry.

Tunable parameters:
- SIMILARITY_THRESHOLD: cosine threshold for considering two queries "close enough". Lower values increase hit rate but may return less-accurate reuse.
- TOP_K_CLUSTERS: number of clusters to consider for partitioned lookup.

We also maintain stats: hit_count, miss_count, total_entries.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

SIMILARITY_THRESHOLD = 0.86  # tunable; see experiments/ for sweep experiments
TOP_K_CLUSTERS = 3


class CacheEntry:
    def __init__(self, query: str, embedding: np.ndarray, result: Any, cluster_dist: np.ndarray):
        self.query = query
        self.embedding = embedding.astype("float32")
        self.result = result
        self.cluster_dist = cluster_dist
        self.dominant_cluster = int(np.argmax(cluster_dist))


class SemanticCache:
    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD, top_k_clusters: int = TOP_K_CLUSTERS):
        # partitioned by dominant cluster -> list of CacheEntry
        self.partitions: Dict[int, List[CacheEntry]] = {}
        self.similarity_threshold = float(similarity_threshold)
        self.top_k_clusters = int(top_k_clusters)
        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, embedding: np.ndarray, cluster_dist: np.ndarray) -> Optional[Tuple[CacheEntry, float]]:
        """Return (entry, similarity) if hit, else None."""
        # select top clusters of the query to limit search space
        top_idxs = np.argsort(cluster_dist)[-self.top_k_clusters:][::-1]
        emb = embedding.astype("float32")
        best_sim = -1.0
        best_entry = None
        for c in top_idxs:
            if c not in self.partitions:
                continue
            # linear scan within partition (could be optimized with per-partition index)
            for entry in self.partitions[c]:
                sim = float(np.dot(emb, entry.embedding))  # both normalized
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry
        if best_entry is not None and best_sim >= self.similarity_threshold:
            self.hit_count += 1
            return best_entry, best_sim
        self.miss_count += 1
        return None

    def add(self, query: str, embedding: np.ndarray, result: Any, cluster_dist: np.ndarray):
        entry = CacheEntry(query, embedding, result, cluster_dist)
        p = entry.dominant_cluster
        if p not in self.partitions:
            self.partitions[p] = []
        self.partitions[p].append(entry)

    def stats(self) -> Dict[str, Any]:
        total_entries = sum(len(v) for v in self.partitions.values())
        hit_rate = self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0.0
        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
        }

    def flush(self):
        self.partitions = {}
        self.hit_count = 0
        self.miss_count = 0

