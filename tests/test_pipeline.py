import json
import sqlite3
import threading
from dataclasses import replace
import numpy as np
import pytest
from document_indexer.loaders import load_document, load_corpus, corpus_stats, save_upload
from document_indexer.chunkers import chunk_document
from document_indexer.sections import extract_sections
from document_indexer.storage import load_index, read_index, write_index, compatibility
from document_indexer.indexing import build_index, is_stale
from document_indexer.search import rank, search
from document_indexer.evaluate import metrics, validate_questions, relevance, evaluate
from document_indexer.jobs import JobManager

def test_utf8_bom_normalization_and_encoding(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfTitle\r\n\r\n    code\x00\rEnd")
    doc = load_document(path, tmp_path)
    assert doc.text == "Title\n\n    code\nEnd"
    assert doc.lines == [0, 6, 7, 16]
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        load_document(path, tmp_path)

def test_empty_and_duplicate_stats(tmp_path):
    for name, text in [("a.md", "abc"), ("b.txt", "abc"), ("empty.txt", "")]:
        (tmp_path / name).write_text(text)
    docs, errors = load_corpus(tmp_path)
    stats = corpus_stats(docs, errors)
    assert stats["empty_files"] == ["empty.txt"]
    assert len(stats["duplicates"]) == 1
    assert stats["unique_words"] == 1
    assert not stats["volume_requirement_met"]

def test_pdf_pages(tmp_path):
    import pymupdf
    path = tmp_path / "pages.pdf"
    pdf = pymupdf.open()
    pdf.new_page().insert_text((50, 50), "First page")
    pdf.new_page()
    pdf.new_page().insert_text((50, 50), "Third page")
    pdf.save(path)
    pdf.close()
    doc = load_document(path, tmp_path)
    assert len(doc.pages) == 3
    assert "OCR" in doc.warnings[0]
    for p, expected in zip(doc.pages, ["First page", "", "Third page"]):
        assert doc.text[p["start_char"]:p["end_char"]].strip() == expected
    assert len(extract_sections(doc)) == 3

def test_headings_and_fences(make_doc):
    doc = make_doc("intro\n# A\ntext\n```kotlin\n# fake\n```\n## B\nb\n### C\nc\n# D\nd\n")
    assert [s.heading_path for s in extract_sections(doc)] == [["Introduction"], ["A"], ["A", "B"], ["A", "B", "C"], ["D"]]

@pytest.mark.parametrize("length,expected", [(1, 1), (300, 1), (301, 2), (550, 2), (551, 3)])
def test_fixed_windows(length, expected, make_doc, tokenizer):
    doc = make_doc("  " + " ".join(f"t{i}" for i in range(length)) + "\n\n", "txt")
    chunks = chunk_document(doc, "fixed", tokenizer)
    assert len(chunks) == expected
    assert chunks[0].start_char == 0 and chunks[-1].end_char == len(doc.text)
    coverage = set()
    for chunk in chunks:
        assert chunk.text == doc.text[chunk.start_char:chunk.end_char]
        coverage.update(range(chunk.start_char, chunk.end_char))
    assert coverage == set(range(len(doc.text)))
    for a, b in zip(chunks, chunks[1:]):
        assert tokenizer.count(doc.text[b.start_char:a.end_char]) == 50
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in chunk_document(doc, "fixed", tokenizer)]

@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (3, 3), (3, -1)])
def test_invalid_settings(size, overlap, make_doc, tokenizer):
    with pytest.raises(ValueError):
        chunk_document(make_doc("one two"), "fixed", tokenizer, size, overlap)

def test_structural_sections_code_and_overlap(make_doc, tokenizer):
    doc = make_doc("# A\nsmall paragraph\n\n```\ncode line\n\nmore code\n```\n\n# B\n" + "word " * 80)
    chunks = chunk_document(doc, "structural", tokenizer, 20, 5)
    assert all(len(c.sections) == 1 for c in chunks)
    code = "```\ncode line\n\nmore code\n```"
    assert sum(code in c.text for c in chunks) == 1
    bchunks = [c for c in chunks if c.sections == [["B"]]]
    assert len(bchunks) > 1
    assert tokenizer.count(doc.text[bchunks[1].start_char:bchunks[0].end_char]) == 5
    assert bchunks[-1].end_char == len(doc.text)

def test_fixed_all_intersected_sections(make_doc, tokenizer):
    chunks = chunk_document(make_doc("# A\na\n# B\nb"), "fixed", tokenizer)
    assert chunks[0].sections == [["A"], ["B"]]

def test_storage_roundtrip_and_ranking(built):
    _, paths = built
    index = read_index(paths["fixed"])
    assert index.matrix.dtype == np.dtype("<f4")
    assert np.allclose(np.linalg.norm(index.matrix, axis=1), 1)
    assert compatibility(index, read_index(paths["structural"])) == []
    results = rank(index, np.array([1, 0, 0]), 10)
    assert results[0]["score"] == 1
    for a, b in zip(results, results[1:]):
        assert a["score"] >= b["score"]
        if a["score"] == b["score"]:
            assert a["chunk_id"] < b["chunk_id"]

def test_corrupted_blob(built):
    _, paths = built
    with sqlite3.connect(paths["fixed"]) as conn:
        conn.execute("UPDATE chunks SET vector=?", (b"bad",))
    with pytest.raises(ValueError, match="BLOB"):
        read_index(paths["fixed"])

def test_atomic_old_index_survives_failure(built, monkeypatch):
    _, paths = built
    path = paths["fixed"]
    original = path.read_bytes()
    index = read_index(path)
    # Fails inside the new database transaction (duplicate chunk primary key).
    with pytest.raises(sqlite3.IntegrityError):
        write_index(path, index.config, list(index.documents.values()), index.chunks + index.chunks[:1], np.vstack([index.matrix, index.matrix[:1]]))
    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.tmp"))
    # Fails at final publication as well.
    monkeypatch.setattr("document_indexer.storage.os.replace", lambda *args: (_ for _ in ()).throw(OSError("locked")))
    with pytest.raises(OSError):
        write_index(path, index.config, list(index.documents.values()), index.chunks, index.matrix)
    assert path.read_bytes() == original

def test_snapshot_staleness_and_cache_version(built, service):
    raw, paths = built
    old = load_index(paths["fixed"])
    (raw / "one.md").write_text("# Changed\nnew corpus")
    docs, _ = load_corpus(raw)
    assert is_stale(old, docs, 6, 2)
    assert "alpha" in old.documents["one.md"].text
    build_index(raw, "fixed", paths["fixed"], 6, 2, service=service)
    new = load_index(paths["fixed"])
    assert old.config["build_id"] != new.config["build_id"]

def test_metrics():
    result = metrics([1, 2, None, 5, 6])
    assert result == pytest.approx({"Hit@1": .2, "Hit@5": .6, "MRR@5": .34})

def question_for(index):
    doc = next(iter(index.documents.values()))
    start = doc.text.index("alpha")
    return {"questions": [{"id": "q1", "question": "alpha?", "answers": [{"source": doc.source, "document_hash": doc.content_hash,
                             "start_char": start, "end_char": start + 5, "answer_excerpt": "alpha"}]}]}

def test_annotation_validation_and_containment(built):
    _, paths = built
    index = read_index(paths["fixed"])
    data = question_for(index)
    validate_questions(data, index.documents)
    answer = data["questions"][0]["answers"][0]
    assert relevance({"source": "one.md", "start_char": 0, "end_char": 100}, [answer])
    assert not relevance({"source": "one.md", "start_char": answer["start_char"] + 1, "end_char": 100}, [answer])
    answer["answer_excerpt"] = "wrong"
    with pytest.raises(ValueError, match="excerpt"):
        validate_questions(data, index.documents)
    answer["document_hash"] = "wrong"
    with pytest.raises(ValueError, match="Устаревшая"):
        validate_questions(data, index.documents)

def test_evaluation_measurements(built, service):
    _, paths = built
    fixed, structural = read_index(paths["fixed"]), read_index(paths["structural"])
    report = evaluate(fixed, structural, question_for(fixed), repeats=2, service=service)
    assert report["question_count"] == 1
    assert report["strategies"]["fixed"]["timing"]["samples"] == 2
    assert report["strategies"]["fixed"]["metrics"]["Hit@1"] == 1

def test_empty_query_index_and_corrupt_db(built, service, tmp_path):
    _, paths = built
    index = read_index(paths["fixed"])
    with pytest.raises(ValueError, match="непустой"):
        search(index, "  ", service=service)
    empty = replace(index, chunks=[], matrix=np.empty((0, 3)))
    assert search(empty, "q", service=service)["results"] == []
    bad = tmp_path / "bad.sqlite"
    bad.write_text("not sqlite")
    with pytest.raises(ValueError, match="повреждён"):
        read_index(bad)

def test_upload_safety(tmp_path):
    first = save_upload(tmp_path, "../../CON.txt", b"hello")
    second = save_upload(tmp_path, "../../CON.txt", b"world")
    assert first != second and first.startswith("document_")
    assert (tmp_path / first).read_bytes() == b"hello"
    with pytest.raises(ValueError):
        save_upload(tmp_path, "evil.exe", b"hello")
    with pytest.raises(ValueError, match="20"):
        save_upload(tmp_path, "large.txt", b"x" * (20 * 1024 * 1024 + 1))

def test_jobs_no_duplicate_and_failure():
    manager = JobManager()
    release = threading.Event()
    calls = []
    def action(progress):
        calls.append(1)
        progress("chunks", 1, 2)
        release.wait(3)
        return 42
    first = manager.submit("build", action)
    assert manager.submit("build", action) == first
    release.set()
    manager.close()
    assert manager.snapshot(first)["status"] == "completed"
    assert calls == [1]
    manager = JobManager()
    failed = manager.submit("fail", lambda p: 1 / 0)
    manager.close()
    assert manager.snapshot(failed)["status"] == "failed"

def test_source_html_escaped(make_doc, tokenizer):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
    from views.shared import source_html
    doc = make_doc("# Hi\n<script>alert(1)</script>")
    chunk = chunk_document(doc, "fixed", tokenizer)[0]
    html = source_html(doc, chunk)
    assert "<script>" not in html and "&lt;script&gt;" in html
