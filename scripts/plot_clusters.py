"""
Generate a 2D t-SNE visualization of the corpus embeddings colored by dominant cluster.

Saves `analysis/cluster_tsne.png` and prints counts per cluster used in the plot.
"""
import os
import numpy as np
import random

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from app.vectorstore import EMB_PATH, META_PATH, DATA_DIR
from app.clustering import Clusterer

import matplotlib.pyplot as plt


def main(sample_size=2000, random_state=0):
    os.makedirs('analysis', exist_ok=True)
    print('Loading embeddings and clusterer...')
    embeddings = np.load(EMB_PATH)
    cl = Clusterer()
    cl.load()
    probs = cl.predict_proba(embeddings)
    dominant = np.argmax(probs, axis=1)

    n = embeddings.shape[0]
    # stratified sampling: try to sample up to sample_size balanced across clusters
    clusters = np.unique(dominant)
    idxs = []
    per_cluster = max(1, sample_size // len(clusters))
    for c in clusters:
        cidx = np.where(dominant == c)[0]
        if len(cidx) <= per_cluster:
            idxs.extend(cidx.tolist())
        else:
            idxs.extend(np.random.RandomState(random_state).choice(cidx, per_cluster, replace=False).tolist())

    idxs = np.array(idxs)
    X = embeddings[idxs]
    labels = dominant[idxs]

    print(f'Sampled {len(idxs)} points across {len(clusters)} clusters')

    # reduce with PCA then TSNE for speed
    print('Running PCA -> TSNE (this may take ~30s for 2000 points)...')
    pca = PCA(n_components=min(50, X.shape[1]))
    Xp = pca.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=40, random_state=random_state, init='pca')
    X2 = tsne.fit_transform(Xp)

    # plot
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab10')
    for c in clusters:
        sel = labels == c
        plt.scatter(X2[sel, 0], X2[sel, 1], s=6, color=cmap(int(c) % 10), label=f'cluster {c}', alpha=0.7)
    plt.legend(markerscale=3, fontsize='small')
    plt.title('t-SNE of 20 Newsgroups embeddings (sampled)')
    out_path = os.path.join('analysis', 'cluster_tsne.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
