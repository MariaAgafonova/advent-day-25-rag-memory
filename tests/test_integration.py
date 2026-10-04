import numpy as np
import pytest
from document_indexer.embeddings import get_embedding_service
from document_indexer.indexing import build_index
from document_indexer.storage import load_index
from document_indexer.search import search

@pytest.mark.integration
def test_real_model_roundtrip(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "android.md").write_text("# Background work\nWorkManager schedules reliable background tasks that persist across app restarts.\n\n# UI state\nViewModel stores UI state across configuration changes.", encoding="utf-8")
    service = get_embedding_service()
    assert len(service.revision) == 40
    for strategy in ("fixed", "structural"):
        path = tmp_path / (strategy + ".sqlite")
        build_index(raw, strategy, path, service=service)
        index = load_index(path)
        assert index.matrix.shape[1] == 384
        assert np.isfinite(index.matrix).all()
        result = search(index, "Как запускать фоновые задачи?", service=service)
        assert "WorkManager" in result["results"][0]["text"]
    with pytest.raises(ValueError, match="truncation"):
        service.encode_documents(["background " * 1000])
