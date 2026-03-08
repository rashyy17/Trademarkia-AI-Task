"""
Run a threshold sweep for the semantic cache to measure hit/miss tradeoffs.

This script:
- Samples seed queries from the corpus (short snippets)
- For each seed, also adds a similar query by using vectorstore nearest neighbour (to simulate paraphrases)
- For each threshold value, simulates queries in sequence against a fresh SemanticCache and records hit/miss stats

Outputs `data/threshold_sweep.csv` with columns: threshold, total_queries, hits, misses, hit_rate
"""
import os
import csv
import random
import numpy as np

from app.vectorstore import VectorStore
from app.clustering import Clusterer
from app.cache import SemanticCache
from app.vectorstore import DATA_DIR, META_PATH


def load_text_snippets(n=200, maxlen=120):
    import pickle
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f)['meta']
    texts = [m['text'] for m in meta]
    # sample unique indices
    idxs = random.sample(range(len(texts)), min(n, len(texts)))
    snippets = []
    for i in idxs:
        t = texts[i].replace('\n', ' ').strip()
        snippets.append(t[:maxlen])
    return snippets


def build_queries(vs, seeds):
    queries = []
    for s in seeds:
        queries.append(s)
        # find a nearest neighbour to act as a paraphrase / semantically similar query
        res = vs.search(s, k=2)
        if res and len(res) > 1:
            # neighbor text
            queries.append(res[1][2][:120])
    return queries


def sweep(thresholds, queries, vs, cl, top_k_clusters=3):
    out = []
    for t in thresholds:
        cache = SemanticCache(similarity_threshold=t, top_k_clusters=top_k_clusters)
        hits = 0
        misses = 0
        for q in queries:
            emb = vs.embed(q)
            cd = cl.predict_proba(np.array([emb]))[0]
            hit = cache.lookup(emb, cd)
            if hit:
                hits += 1
            else:
                misses += 1
                # simulate compute result and add
                res = vs.search(q, k=1)
                text = res[0][2] if res else ""
                cache.add(q, emb, text, cd)
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0
        out.append((t, total, hits, misses, hit_rate))
        print(f"threshold={t:.3f} hits={hits}/{total} hit_rate={hit_rate:.3f}")
    return out


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Loading vectorstore and clusterer...")
    vs = VectorStore()
    vs.load()
    cl = Clusterer()
    cl.load()

    seeds = load_text_snippets(n=150)
    queries = build_queries(vs, seeds)
    # shuffle queries to simulate real usage
    random.shuffle(queries)

    thresholds = list(np.linspace(0.5, 0.95, 10))
    results = sweep(thresholds, queries, vs, cl, top_k_clusters=3)

    out_path = os.path.join(DATA_DIR, 'threshold_sweep.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['threshold', 'total_queries', 'hits', 'misses', 'hit_rate'])
        for row in results:
            w.writerow(row)

    print(f"Sweep results written to {out_path}")


if __name__ == '__main__':
    main()
