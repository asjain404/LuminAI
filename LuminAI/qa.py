# luminai/qa.py
from typing import Dict, List
from .models import get_qa, get_summarizer
from .retriever import search

def answer(question: str, k: int = 5) -> Dict:
    hits = search(question, k=k)
    qa = get_qa()
    best = {"answer": "", "score": 0.0, "context": "", "source": None}
    for h in hits:
        out = qa({"question": question, "context": h["text"]})
        if out["score"] > best["score"]:
            best = {
                "answer": out.get("answer", ""),
                "score": float(out.get("score", 0.0)),
                "context": h["text"],
                "source": h["meta"]["file"]
            }
    return {"question": question, "result": best, "hits": hits}

def summarize_evidence(texts: List[str], max_words: int = 120) -> str:
    if not texts:
        return ""
    summarizer = get_summarizer()
    joined = " ".join(texts)[:4000]  # keep small for CPU
    s = summarizer(joined, max_length=180, min_length=60, do_sample=False)
    return s[0]["summary_text"]
