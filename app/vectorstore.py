"""
Vector store using sentence-transformers for embeddings and FAISS for vector search.

Design notes (in-code justification):
- We use `all-MiniLM-L6-v2` from sentence-transformers: it's a compact, fast model with good performance for semantic retrieval.
- We normalize embeddings and use FAISS IndexFlatIP so that inner product equals cosine similarity over normalized vectors (fast, simple, reliable for small/medium corpora).
- We persist FAISS index and metadata to disk for quick startup.
"""
import os
import pickle
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
META_PATH = os.path.join(DATA_DIR, "meta.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"
EMB_PATH = os.path.join(DATA_DIR, "embeddings.npy")


class VectorStore:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.meta = None  # list of docs (dicts with id, text, target)

    def prepare_corpus(self) -> List[str]:
        """Load 20 Newsgroups using scikit-learn helper.

        Preprocessing choices (commented justification):
        - We keep the body text largely intact but remove headers that often duplicate or reveal metadata like 'From', 'Subject', 'Path'.
          These headers can leak non-semantic signals (email addresses, organization names) that aren't helpful for topical semantic search.
        - We do light trimming and drop extremely short posts (<20 chars) which rarely carry useful topical content.
        """
        raw = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        texts = []
        metas = []
        # iterate in parallel and keep targets aligned after filtering short/empty posts
        for i, t in enumerate(raw.data):
            if not t:
                continue
            txt = t.strip()
            if len(txt) < 20:
                # drop extremely short posts which rarely carry topical signal
                continue
            texts.append(txt)
            metas.append({"text": txt, "target": int(raw.target[i])})
        self.meta = metas
        return texts

    def build(self, force: bool = False):
        os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH) and not force:
            print("Index and metadata found on disk. Loading...")
            self.load()
            return

        print("Preparing corpus...")
        texts = self.prepare_corpus()
        print(f"Encoding {len(texts)} documents with {MODEL_NAME}...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # Normalize for cosine using inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings.astype("float32"))

        # persist
        faiss.write_index(index, INDEX_PATH)
        # persist metadata and embeddings to speed up downstream clustering and reloads
        with open(META_PATH, "wb") as f:
            pickle.dump({"meta": self.meta}, f)
        np.save(EMB_PATH, embeddings.astype("float32"))
        self.index = index
        print("Index built and saved.")

    def load(self):
        with open(META_PATH, "rb") as f:
            payload = pickle.load(f)
        self.meta = payload["meta"]
        self.index = faiss.read_index(INDEX_PATH)

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float, str]]:
        emb = self.model.encode([query], convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        D, I = self.index.search(emb.astype("float32"), k)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append((int(idx), float(score), self.meta[idx]["text"]))
        return results

    def embed(self, text: str) -> np.ndarray:
        emb = self.model.encode([text], convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb[0]
