"""
Generate a short cluster report: exemplars per cluster, top terms, and boundary cases.

Produces `data/cluster_report.txt` and prints a concise summary to stdout.
"""
import os
import numpy as np
import pickle
from collections import defaultdict
from math import log

from app.vectorstore import DATA_DIR, EMB_PATH, META_PATH
from app.clustering import Clusterer, CLUSTER_PATH

from sklearn.feature_extraction.text import TfidfVectorizer


def entropy(probs):
    # small numerical safe entropy (nats)
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    probs = np.clip(probs, 1e-12, 1.0)
    return -float((probs * np.log(probs)).sum())


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(EMB_PATH) or not os.path.exists(META_PATH) or not os.path.exists(CLUSTER_PATH):
        print("Please run `python3 scripts/build_index.py` first to build embeddings and clusters.")
        return

    print("Loading embeddings and metadata...")
    embeddings = np.load(EMB_PATH)
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f)["meta"]
    texts = [m['text'] for m in meta]

    print("Loading clusterer...")
    cl = Clusterer()
    cl.load()
    probs = cl.predict_proba(embeddings)
    n_clusters = cl.n_components

    # exemplar docs per cluster (highest probability)
    exemplars = defaultdict(list)
    for i in range(n_clusters):
        # indices sorted by prob desc
        idxs = np.argsort(probs[:, i])[::-1][:10]
        exemplars[i] = idxs.tolist()

    # compute entropy per doc to find boundary/uncertain docs
    ent = np.array([entropy(p) for p in probs])
    uncertain_idxs = np.argsort(ent)[::-1][:20]

    # top terms per cluster via weighted TF-IDF (weights = cluster prob)
    print("Computing TF-IDF matrix (this may take a moment)...")
    vect = TfidfVectorizer(max_features=20000, stop_words='english')
    X = vect.fit_transform(texts)
    terms = np.array(vect.get_feature_names_out())

    top_terms = {}
    for k in range(n_clusters):
        weights = probs[:, k]
        # weighted TF-IDF sum across docs
        score_vec = X.T @ weights
        # handle sparse/dense result
        if hasattr(score_vec, "A1"):
            score_vec = score_vec.A1
        else:
            score_vec = np.asarray(score_vec).ravel()
        top_idx = np.argsort(score_vec)[::-1][:20]
        top_terms[k] = terms[top_idx].tolist()

    # write report
    out_path = os.path.join(DATA_DIR, 'cluster_report.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('Cluster Report\n')
        f.write('================\n\n')
        f.write(f'Number of clusters: {n_clusters}\n')
        f.write('\nTop terms per cluster:\n')
        for k in range(n_clusters):
            f.write(f'\nCluster {k}: ' + ', '.join(top_terms[k][:10]) + '\n')

        f.write('\nExemplar documents (top 3 snippets) per cluster:\n')
        for k in range(n_clusters):
            f.write(f'\nCluster {k}:\n')
            for idx in exemplars[k][:3]:
                txt = texts[idx].replace('\n', ' ')[:400]
                f.write(f' - (doc {idx}) {txt}\n')

        f.write('\nBoundary / uncertain documents (high entropy):\n')
        for idx in uncertain_idxs:
            f.write(f'\n(doc {idx}) entropy={ent[idx]:.4f} top_probs={np.sort(probs[idx])[-3:][::-1]}\n')
            f.write(texts[idx].replace('\n', ' ')[:800] + '\n')

    print(f'Report written to {out_path}')
    # print brief summary
    print('\nSummary:')
    for k in range(n_clusters):
        print(f'Cluster {k}: top terms: {", ".join(top_terms[k][:6])}')
        ex_idx = exemplars[k][0]
        print(f'  exemplar doc {ex_idx} snippet: {texts[ex_idx][:120].replace("\n"," ")}...')

    print('\nTop uncertain documents (by entropy):')
    for idx in uncertain_idxs[:5]:
        print(f' doc {idx} entropy={ent[idx]:.4f} top_probs={np.sort(probs[idx])[-3:][::-1]}')


if __name__ == '__main__':
    main()
