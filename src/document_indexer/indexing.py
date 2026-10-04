from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter
from uuid import uuid4
import json
from .models import to_dict
from .loaders import load_corpus, corpus_hash, corpus_stats
from .chunkers import chunk_document
from .sections import extract_sections
from .embeddings import get_embedding_service, MODEL_ID
from .normalize import NORMALIZATION
from .storage import write_index, SCHEMA_VERSION
from .tokenization import validate_settings

def build_index(input_dir, strategy, output, size=300, overlap=50, device="cpu", service=None, progress=None):
    validate_settings(size, overlap)
    emit = progress or (lambda *args: None)
    started = perf_counter()
    emit("загрузка", 0, None)
    documents, errors = load_corpus(input_dir)
    if errors:
        raise ValueError("Исправьте ошибки загрузки: " + json.dumps(errors, ensure_ascii=False))
    if not any(d.text.strip() for d in documents):
        raise ValueError("Корпус пуст: добавьте документы с текстом")
    emit("модель", 0, None)
    service = service or get_embedding_service(device=device)
    emit("разделы", 0, len(documents))
    for i, doc in enumerate(documents, 1):
        extract_sections(doc)
        emit("разделы", i, len(documents))
    chunks = []
    for i, doc in enumerate(documents, 1):
        chunks.extend(chunk_document(doc, strategy, service.tokenization, size, overlap))
        emit("chunking", i, len(documents))
    vectors = service.encode_documents([c.text for c in chunks], lambda done, total: emit("эмбеддинги", done, total))
    config = {"schema_version": SCHEMA_VERSION, "build_id": str(uuid4()), "corpus_hash": corpus_hash(documents),
              "model_id": service.model_id, "revision": service.revision, "dimension": service.dimension,
              "dtype": "<f4", "strategy": strategy, "chunking": {"size": size, "overlap": overlap},
              "normalization": NORMALIZATION, "embedding_input": {"document": "passage: ", "query": "query: ", "normalize": True},
              "dependencies": {p: version(p) for p in ("sentence-transformers", "transformers", "numpy", "PyMuPDF", "streamlit")},
              "created_at": datetime.now(timezone.utc).isoformat(), "model_load_seconds": service.load_seconds,
              "corpus_stats": corpus_stats(documents, errors, service.tokenization), "chunk_count": len(chunks),
              "device": device, "build_seconds_before_save": perf_counter() - started}
    emit("сохранение", 0, len(chunks))
    write_index(output, config, documents, chunks, vectors)
    output = Path(output)
    with output.with_suffix(".jsonl").open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(to_dict(chunk), ensure_ascii=False) + "\n")
    timing = {"build_id": config["build_id"], "total_seconds": perf_counter() - started}
    output.with_suffix(".timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    emit("готово", len(chunks), len(chunks))
    return {"config": config, **timing}

def is_stale(index, documents, size=300, overlap=50):
    return index.config["corpus_hash"] != corpus_hash(documents) or index.config["chunking"] != {"size": size, "overlap": overlap}
