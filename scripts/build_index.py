"""
Script to build the vector index and clusters and persist them to disk.

Usage:
    python3 scripts/build_index.py

This will:
- Load the 20 Newsgroups corpus (using sklearn helper)
- Build embeddings and FAISS index
- Fit Clusterer on embeddings and persist PCA+GMM

Note: This may take a few minutes depending on CPU.
"""
import os
import pickle
import numpy as np

from app.vectorstore import VectorStore, DATA_DIR, INDEX_PATH, META_PATH
from app.clustering import Clusterer, CLUSTER_PATH
from app.vectorstore import EMB_PATH


def main():
    vs = VectorStore()
    vs.build(force=True)
    # Load saved embeddings if present to avoid re-encoding the entire corpus
    if os.path.exists(EMB_PATH):
        print("Loading saved embeddings from disk...")
        embeddings = np.load(EMB_PATH)
    else:
        texts = [m["text"] for m in vs.meta]
        print("Re-encoding corpus to compute embeddings for clustering...")
        embeddings = vs.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        np.save(EMB_PATH, embeddings.astype("float32"))

    print("Fitting Clusterer (this may take a bit)...")
    clusterer = Clusterer(n_components=None, pca_dim=50)
    clusterer.fit(embeddings, max_components=25, min_components=8)
    clusterer.save()
    print("Clustering saved to disk.")


if __name__ == "__main__":
    main()
