# luminai/models.py
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sentence_transformers import CrossEncoder

@lru_cache(maxsize=1)
def get_embedder():
    # fast CPU embedder
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def get_qa():
    # extractive QA: CPU
    return pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)

@lru_cache(maxsize=1)
def get_summarizer():
    # small summarizer for 3-5 sentence drafts (optional)
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

@lru_cache(maxsize=1)
def get_reranker():
    # fast CPU cross-encoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
