"""
Fuzzy clustering using Gaussian Mixture Models (soft assignments).

Design justification:
- Requirement: soft / fuzzy clusters (a distribution per document). GaussianMixture provides `predict_proba` which gives a distribution over components.
- Number of clusters is chosen by minimizing BIC across a reasonable range. BIC is useful because it penalizes extra components.
- We reduce dimensionality before clustering (PCA) to make GMM stable and faster.
"""
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLUSTER_PATH = os.path.join(DATA_DIR, "clusters.pkl")


class Clusterer:
    def __init__(self, n_components: Optional[int] = None, pca_dim: int = 50):
        self.n_components = n_components
        self.pca_dim = pca_dim
        self.pca = None
        self.scaler = None
        self.gmm = None

    def fit(self, embeddings: np.ndarray, max_components: int = 30, min_components: int = 8):
        """Fit a PCA + GMM pipeline. If n_components is None, choose by BIC over [min_components, max_components].

        embeddings: expected shape (n_samples, dim) and already normalized. We scale and reduce dimensionality before GMM.
        """
        # Persist the scaler so that predict_proba uses the same scaling as fit
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(embeddings)
        self.pca = PCA(n_components=min(self.pca_dim, X.shape[1]))
        Xr = self.pca.fit_transform(X)

        if self.n_components is None:
            best_bic = np.inf
            best_k = None
            best_gmm = None
            for k in range(min_components, min(max_components, len(Xr)) + 1):
                gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
                gmm.fit(Xr)
                bic = gmm.bic(Xr)
                print(f"GMM k={k} BIC={bic}")
                if bic < best_bic:
                    best_bic = bic
                    best_k = k
                    best_gmm = gmm
            self.gmm = best_gmm
            self.n_components = best_k
            print(f"Selected n_components={self.n_components} by BIC")
        else:
            self.gmm = GaussianMixture(n_components=self.n_components, covariance_type="full", random_state=0)
            self.gmm.fit(Xr)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Return soft cluster distributions for given embeddings."""
        if self.pca is None or self.gmm is None:
            raise RuntimeError("Clusterer not fitted")
        # apply the same preprocessing
        if self.scaler is None:
            # defensive fallback
            X = embeddings
        else:
            X = self.scaler.transform(embeddings)
        Xr = self.pca.transform(X)
        return self.gmm.predict_proba(Xr)

    def save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CLUSTER_PATH, "wb") as f:
            pickle.dump({
                "pca": self.pca,
                "gmm": self.gmm,
                "n_components": self.n_components,
                "scaler": self.scaler,
            }, f)

    def load(self):
        with open(CLUSTER_PATH, "rb") as f:
            payload = pickle.load(f)
        self.pca = payload["pca"]
        self.gmm = payload["gmm"]
        self.n_components = int(payload.get("n_components", self.gmm.n_components))
        self.scaler = payload.get("scaler", None)
