import numpy as np
from app.vectorstore import VectorStore


def test_embed_shape():
    vs = VectorStore()
    # instantiate model but don't require index
    emb = vs.embed("hello world")
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
