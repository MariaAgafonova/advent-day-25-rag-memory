from time import perf_counter
import numpy as np
from .embeddings import get_embedding_service
from .models import to_dict
from .storage import load_index

def rank(index, query_vector, top_k=5, source=None):
    if top_k <= 0:
        raise ValueError("top-k должен быть положительным")
    query_vector = np.asarray(query_vector)
    if query_vector.shape != (index.config["dimension"],) or not np.isfinite(query_vector).all():
        raise ValueError("Неверный вектор запроса")
    scores = index.matrix @ query_vector
    candidates = [i for i, c in enumerate(index.chunks) if source is None or c.source == source]
    selected = sorted(candidates, key=lambda i: (-float(scores[i]), index.chunks[i].chunk_id))[:top_k]
    return [{"rank": n, "score": float(scores[i]), **to_dict(index.chunks[i])} for n, i in enumerate(selected, 1)]

def search(index, query, top_k=5, source=None, service=None):
    if not query.strip():
        raise ValueError("Введите непустой запрос")
    if not index.chunks:
        return {"results": [], "seconds": 0.0, "message": "Индекс пуст"}
    service = service or get_embedding_service(index.config["model_id"], index.config["revision"])
    if (service.model_id, service.revision, service.dimension) != (index.config["model_id"], index.config["revision"], index.config["dimension"]):
        raise ValueError("Модель запроса несовместима с индексом")
    started = perf_counter()
    vector = service.encode_query(query)
    results = rank(index, vector, top_k, source)
    return {"results": results, "seconds": perf_counter() - started}

def search_both(fixed, structural, query, top_k=5, source=None):
    from .storage import compatibility
    reasons = compatibility(fixed, structural)
    if reasons:
        raise ValueError("Индексы несовместимы: " + "; ".join(reasons))
    return {"fixed": search(fixed, query, top_k, source), "structural": search(structural, query, top_k, source)}
