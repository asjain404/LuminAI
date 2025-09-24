from luminai.ingest import index_ready, INDEX_DIR
def test_index_exists():
    assert index_ready(), f"Index not found at {INDEX_DIR}"
