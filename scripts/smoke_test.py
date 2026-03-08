"""
Smoke test to build the index/clusters and run a few sample queries through the vectorstore and semantic cache.

Run after installing requirements. This script demonstrates a small end-to-end flow without starting the FastAPI server.
"""
from app.vectorstore import VectorStore
from app.clustering import Clusterer
from app.cache import SemanticCache


def main():
    print("Building / loading vectorstore and clusters...")
    vs = VectorStore()
    vs.build()

    # load clusterer
    cl = Clusterer()
    try:
        cl.load()
        print(f"Loaded clusterer with {cl.n_components} components")
    except Exception:
        print("No clusterer found. Please run scripts/build_index.py first.")
        return

    cache = SemanticCache()

    queries = [
        "What is the debate around gun control?",
        "Discussion on firearms legislation and rights",
        "How to cook pasta al dente",
    ]

    for q in queries:
        emb = vs.embed(q)
        cluster_dist = cl.predict_proba([emb])[0]
        hit = cache.lookup(emb, cluster_dist)
        if hit:
            entry, sim = hit
            print(f"HIT: '{q}' matched '{entry.query}' (sim={sim:.3f})")
        else:
            res = vs.search(q, k=1)
            text = res[0][2][:200] if res else ""
            cache.add(q, emb, text, cluster_dist)
            print(f"MISS: '{q}' -> cached result (snippet) '{text[:80]}...' )")

    # Re-run a similar query to test hit
    q2 = "Gun laws and firearm regulation discussion"
    emb2 = vs.embed(q2)
    cd2 = cl.predict_proba([emb2])[0]
    hit = cache.lookup(emb2, cd2)
    if hit:
        entry, sim = hit
        print(f"Second-phase HIT: '{q2}' matched '{entry.query}' (sim={sim:.3f})")
    else:
        print("Second-phase MISS: cache did not match; consider lowering threshold for more aggressive reuse.")


if __name__ == '__main__':
    main()
