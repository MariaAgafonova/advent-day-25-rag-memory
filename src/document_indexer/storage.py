from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
import json
import os
import sqlite3
import tempfile
from contextlib import closing
import numpy as np
from .models import Document, Chunk, to_dict, canonical, digest
from .embeddings import validate_vectors

SCHEMA_VERSION = 1

@dataclass
class Index:
    config: dict
    documents: dict[str, Document]
    chunks: list[Chunk]
    matrix: np.ndarray

def write_index(path, config, documents, chunks, vectors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = validate_vectors(vectors, len(chunks), config["dimension"])
    fd, temporary = tempfile.mkstemp(prefix=path.stem + "-", suffix=".sqlite.tmp", dir=path.parent)
    os.close(fd)
    try:
        conn = sqlite3.connect(temporary)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript("""
                CREATE TABLE index_config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE documents (source TEXT PRIMARY KEY, content_hash TEXT NOT NULL, metadata TEXT NOT NULL);
                CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, source TEXT NOT NULL REFERENCES documents(source),
                                     ordinal INTEGER NOT NULL, metadata TEXT NOT NULL, vector BLOB NOT NULL);
                CREATE INDEX chunks_source ON chunks(source, ordinal);
            """)
            conn.executemany("INSERT INTO index_config VALUES (?,?)", [(k, canonical(v)) for k, v in config.items()])
            conn.executemany("INSERT INTO documents VALUES (?,?,?)",
                             [(d.source, d.content_hash, canonical(to_dict(d))) for d in documents])
            conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?)",
                             [(c.chunk_id, c.source, c.ordinal, canonical(to_dict(c)), v.astype("<f4").tobytes())
                              for c, v in zip(chunks, vectors)])
            conn.commit()
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Ошибка целостности SQLite")
        finally:
            conn.close()
        # Validate using the same reader before publishing a new version.
        read_index(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def _connect(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def read_config(path):
    try:
        with closing(_connect(path)) as conn:
            config = {k: json.loads(v) for k, v in conn.execute("SELECT key,value FROM index_config")}
        if config["schema_version"] != SCHEMA_VERSION or config["dtype"] != "<f4":
            raise ValueError("Неподдерживаемая схема или dtype")
        if not isinstance(config["dimension"], int) or not 0 < config["dimension"] < 100000:
            raise ValueError("Недопустимая размерность")
        return config
    except (sqlite3.Error, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Индекс недоступен или повреждён: {exc}") from exc

def read_index(path):
    config = read_config(path)
    try:
        conn = _connect(path)
        try:
            documents = {source: Document(**json.loads(meta)) for source, meta in conn.execute("SELECT source,metadata FROM documents")}
            chunks, vectors = [], []
            for meta, blob in conn.execute("SELECT metadata,vector FROM chunks ORDER BY source,ordinal,chunk_id"):
                if len(blob) != config["dimension"] * 4:
                    raise ValueError("Повреждённый BLOB: длина не равна dimension × 4")
                chunk = Chunk(**json.loads(meta))
                doc = documents[chunk.source]
                if not 0 <= chunk.start_char < chunk.end_char <= len(doc.text):
                    raise ValueError("Повреждены границы чанка")
                if doc.text[chunk.start_char:chunk.end_char] != chunk.text or digest(chunk.text) != chunk.content_hash:
                    raise ValueError("Повреждён текст чанка")
                chunks.append(chunk)
                vectors.append(np.frombuffer(blob, dtype="<f4"))
            for doc in documents.values():
                if digest(doc.text) != doc.content_hash:
                    raise ValueError("Повреждён снимок документа")
            matrix = np.vstack(vectors) if vectors else np.empty((0, config["dimension"]), dtype="<f4")
            matrix = validate_vectors(matrix, len(chunks), config["dimension"])
            matrix.flags.writeable = False
            return Index(config, documents, chunks, matrix)
        finally:
            conn.close()
    except (sqlite3.Error, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Индекс повреждён: {exc}") from exc

@lru_cache(maxsize=6)
def _cached_index(path, build_id, mtime_ns, size):
    return read_index(path)

def load_index(path):
    path = Path(path).resolve()
    config = read_config(path)
    stat = path.stat()
    return _cached_index(str(path), config["build_id"], stat.st_mtime_ns, stat.st_size)

def compatibility(fixed, structural):
    reasons = []
    for key in ("schema_version", "corpus_hash", "model_id", "revision", "dimension", "dtype", "normalization", "embedding_input", "chunking"):
        if fixed.config.get(key) != structural.config.get(key):
            reasons.append(f"Различается {key}")
    if fixed.config["strategy"] != "fixed" or structural.config["strategy"] != "structural":
        reasons.append("Ожидались fixed и structural")
    return reasons
