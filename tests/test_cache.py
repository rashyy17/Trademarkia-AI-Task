import numpy as np
from app.cache import SemanticCache


def test_cache_basic_hit_miss():
    # create two orthogonal embeddings and same cluster dist
    e1 = np.array([1.0, 0.0], dtype="float32")
    e2 = np.array([0.0, 1.0], dtype="float32")
    cd = np.array([1.0, 0.0])
    cache = SemanticCache(similarity_threshold=0.9, top_k_clusters=1)
    # initially miss
    assert cache.lookup(e1, cd) is None
    cache.add("q1", e1, "result1", cd)
    # similar vector should hit
    hit = cache.lookup(e1, cd)
    assert hit is not None
    # orthogonal vector should miss
    assert cache.lookup(e2, cd) is None
