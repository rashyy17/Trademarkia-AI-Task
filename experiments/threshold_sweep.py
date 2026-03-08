"""
Simple experiment to sweep similarity threshold and report approximate cache hit behavior.

Run after building index/clusters and making some queries.
"""
import numpy as np
from app.cache import SemanticCache


def simulate_sweep(similarities):
    # similarities: list of floats showing top-similarity for each simulated query
    thresholds = np.linspace(0.6, 0.95, 8)
    for t in thresholds:
        cache = SemanticCache(similarity_threshold=t)
        hits = sum(1 for s in similarities if s >= t)
        total = len(similarities)
        print(f"threshold={t:.2f} hits={hits}/{total} hit_rate={hits/total:.3f}")


if __name__ == '__main__':
    # Example: simulated similarities
    sims = [0.91, 0.86, 0.78, 0.65, 0.92, 0.88, 0.7, 0.95, 0.6, 0.82]
    simulate_sweep(sims)
