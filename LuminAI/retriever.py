# luminai/retriever.py
from typing import List, Dict, Tuple
import numpy as np
import faiss
from .models import get_embedder, get_reranker
from .ingest import load_index


def search(query: str, k: int = 5, rerank: bool = True) -> List[Dict]:
    index, vectors, m = load_index()
    emb = get_embedder().encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(emb, max(k*3, k))  # over-retrieve a bit
    hits = [{
        "score": float(s),
        "text": m["texts"][int(i)],
        "meta": m["meta"][int(i)]
    } for i, s in zip(I[0], D[0]) if int(i) >= 0]

    if rerank and hits:
        rr = get_reranker()
        pairs = [(query, h["text"]) for h in hits]
        rr_scores = rr.predict(pairs).tolist()
        for h, rs in zip(hits, rr_scores):
            h["rerank_score"] = float(rs)
        hits.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
    return hits[:k]
