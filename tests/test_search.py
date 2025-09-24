import bootstrap

from luminai.retriever import search
def test_search_runs():
    hits = search("lighting", k=3)
    assert isinstance(hits, list) and len(hits) > 0
    assert "text" in hits[0] and "meta" in hits[0]
