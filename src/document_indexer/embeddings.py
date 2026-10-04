from functools import lru_cache
from threading import RLock
from time import perf_counter
import re
import numpy as np
from .tokenization import Tokenization

MODEL_ID = "intfloat/multilingual-e5-small"
_load_lock = RLock()

def validate_vectors(vectors, count, dimension):
    vectors = np.asarray(vectors, dtype="<f4")
    if vectors.shape != (count, dimension):
        raise ValueError(f"Неверная форма векторов: {vectors.shape}, ожидается {(count, dimension)}")
    if not np.isfinite(vectors).all():
        raise ValueError("Векторы содержат NaN/Infinity")
    if count and not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
        raise ValueError("Векторы не нормализованы")
    return vectors

class EmbeddingService:
    def __init__(self, model_id=MODEL_ID, revision=None, device="cpu", batch_size=16):
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        started = perf_counter()
        self.model = SentenceTransformer(model_id, revision=revision, device=device, trust_remote_code=False)
        self.model_id = model_id
        self.revision = getattr(self.model[0].auto_model.config, "_commit_hash", None)
        if not self.revision or not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError("Не удалось установить фактическую revision модели")
        if revision and self.revision != revision:
            raise ValueError("Загружена другая revision модели")
        self.tokenization = Tokenization(self.model.tokenizer)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.limit = min(self.model.max_seq_length, self.model.tokenizer.model_max_length)
        self.batch_size = batch_size
        self.lock = RLock()
        self.load_seconds = perf_counter() - started

    def _encode(self, texts, prefix, progress=None):
        # Prefix is owned by this boundary only. Metadata never enters the model.
        inputs = [text if text.startswith(prefix) else prefix + text for text in texts]
        with self.lock:
            for i, text in enumerate(inputs):
                length = len(self.model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
                if length > self.limit:
                    raise ValueError(f"Вход {i + 1}: {length} токенов > {self.limit}; truncation запрещён. Уменьшите размер чанка/запрос.")
            batches = []
            for start in range(0, len(inputs), self.batch_size):
                batch = self.model.encode(inputs[start:start + self.batch_size], batch_size=self.batch_size,
                                          normalize_embeddings=True, convert_to_numpy=True,
                                          show_progress_bar=False, prompt="")
                batches.append(batch)
                if progress:
                    progress(min(start + self.batch_size, len(inputs)), len(inputs))
        result = np.vstack(batches) if batches else np.empty((0, self.dimension), dtype="<f4")
        return validate_vectors(result, len(texts), self.dimension)

    def encode_documents(self, texts, progress=None):
        return self._encode(texts, "passage: ", progress)

    def encode_query(self, query):
        if not query.strip():
            raise ValueError("Введите непустой запрос")
        return self._encode([query.strip()], "query: ")[0]

_services = {}

def get_embedding_service(model_id=MODEL_ID, revision=None, device="cpu"):
    with _load_lock:
        key = (model_id, revision, device)
        if key not in _services:
            service = EmbeddingService(model_id, revision, device)
            _services[key] = service
            _services[(model_id, service.revision, device)] = service
        return _services[key]
