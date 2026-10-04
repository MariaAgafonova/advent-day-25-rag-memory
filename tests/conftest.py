from pathlib import Path
import re
import numpy as np
import pytest
from document_indexer.models import Document, digest
from document_indexer.normalize import line_map

class WordTokenizer:
    def offsets(self, text):
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    def count(self, text):
        return len(self.offsets(text))

class FakeEmbedding:
    model_id = "test-only"
    revision = "a" * 40
    dimension = 3
    load_seconds = 0
    tokenization = WordTokenizer()
    def encode_documents(self, texts, progress=None):
        result = np.zeros((len(texts), 3), dtype="<f4")
        for i, text in enumerate(texts):
            result[i, i % 3] = 1
        if progress:
            progress(len(texts), len(texts))
        return result
    def encode_query(self, query):
        return np.array([1, 0, 0], dtype="<f4")

@pytest.fixture
def tokenizer():
    return WordTokenizer()

@pytest.fixture
def service():
    return FakeEmbedding()

@pytest.fixture
def make_doc():
    def make(text, format="md", source="test.md"):
        return Document(source, "Test", None, format, text, digest(text), line_map(text))
    return make

@pytest.fixture
def built(tmp_path, service):
    from document_indexer.indexing import build_index
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "one.md").write_text("# One\nalpha beta gamma delta\n\n# Two\nepsilon zeta eta theta\n", encoding="utf-8")
    paths = {}
    for strategy in ("fixed", "structural"):
        path = tmp_path / (strategy + ".sqlite")
        build_index(raw, strategy, path, size=6, overlap=2, service=service)
        paths[strategy] = path
    return raw, paths
