from dataclasses import dataclass, asdict
import math
import re
from .relevance_filter import similarity_filter

MODES = ("baseline", "rewrite", "filter", "rewrite_filter")
ALL_MODES = (*MODES, "reranker")

@dataclass(frozen=True)
class PipelineSettings:
    mode: str = "baseline"
    retrieval_k: int = 15
    final_k: int = 5
    filter_method: str = "similarity_threshold"
    similarity_threshold: float | None = None
    calibration_id: str | None = None
    reranker_model: str | None = None
    reranker_revision: str | None = None
    reranker_threshold: float | None = None
    reranker_max_length: int = 512

    @property
    def rewrite_enabled(self):
        return self.mode in {"rewrite", "rewrite_filter"}

    @property
    def filter_enabled(self):
        return self.mode in {"filter", "rewrite_filter"}

    @property
    def rerank_enabled(self):
        return self.mode == "reranker" or (self.filter_enabled and self.filter_method == "cross_encoder")

    def validate(self):
        if self.mode not in ALL_MODES:
            raise ValueError("Неизвестный режим Дня 23")
        if type(self.retrieval_k) is not int or type(self.final_k) is not int or not 0 < self.final_k <= self.retrieval_k:
            raise ValueError("Требуется 0 < final_k <= retrieval_k (целые числа)")
        if self.mode == "reranker" and self.filter_method != "cross_encoder":
            raise ValueError("Режим reranker требует filter_method=cross_encoder")
        if self.filter_method == "cross_encoder":
            if not self.reranker_model or not re.fullmatch(r"[a-f0-9]{40}", self.reranker_revision or ""):
                raise ValueError("Укажите модель реранкера и точную revision (40 hex-символов)")
            if self.similarity_threshold is not None:
                raise ValueError("Cosine-порог нельзя применять к score реранкера")
            if self.reranker_threshold is not None and not math.isfinite(self.reranker_threshold):
                raise ValueError("Порог реранкера должен быть конечным числом")
            if type(self.reranker_max_length) is not int or not 8 <= self.reranker_max_length <= 512:
                raise ValueError("Лимит реранкера: от 8 до 512 токенов")
            if self.calibration_id:
                raise ValueError("Калибровка cosine-фильтра неприменима к реранкеру; используйте отдельный ручной порог")
        elif self.filter_method == "similarity_threshold":
            if self.reranker_model or self.reranker_revision or self.reranker_threshold is not None:
                raise ValueError("Настройки реранкера неприменимы к cosine-фильтру")
            if self.filter_enabled or self.similarity_threshold is not None:
                similarity_filter([], self.similarity_threshold)
        else:
            raise ValueError("Неизвестный метод обработки")

    def snapshot(self):
        return {**asdict(self), "rewrite_enabled": self.rewrite_enabled,
                "calibration_status": "calibrated" if self.calibration_id else "uncalibrated"}

def process_candidates(chunks, settings, original_query=None, reranker_service=None):
    settings.validate()
    if settings.rerank_enabled:
        from .reranker import get_reranker_service, RerankerError
        service = reranker_service or get_reranker_service(settings.reranker_model, settings.reranker_revision, settings.reranker_max_length)
        if (service.model_id, service.revision, service.max_length) != (settings.reranker_model, settings.reranker_revision, settings.reranker_max_length):
            raise RerankerError("Загруженный реранкер не соответствует настройкам")
        scores = service.score(original_query, [c.text for c in chunks])
        if len(scores) != len(chunks) or any(not math.isfinite(s.score) for s in scores):
            raise RerankerError("Реранкер вернул некорректные scores")
        order = sorted(range(len(chunks)), key=lambda i: (-scores[i].score, chunks[i].rank))
        passed = [i for i in order if settings.reranker_threshold is None or scores[i].score >= settings.reranker_threshold]
        selected_ids, passed_ids = set(passed[:settings.final_k]), set(passed)
        ranks = {i: rank for rank, i in enumerate(order, 1)}
        decisions = [{**asdict(c), "retrieval_rank": c.rank, "reranker_rank": ranks[i], "reranker_score": scores[i].score,
            "reranker_input_tokens": scores[i].input_tokens, "reranker_used_tokens": scores[i].used_tokens,
            "reranker_truncated": scores[i].truncated, "reranker_max_length": service.max_length,
            "score_type": "raw_logit", "filter_applied": settings.reranker_threshold is not None,
            "passed_filter": i in passed_ids, "selected": i in selected_ids, "included_in_context": False, "cited": False,
            "rejection_reason": "below_reranker_threshold" if i not in passed_ids else "final_k_limit" if i not in selected_ids else None}
            for i, c in enumerate(chunks)]
        return [chunks[i] for i in passed[:settings.final_k]], decisions
    passed = similarity_filter(chunks, settings.similarity_threshold) if settings.filter_enabled else list(chunks)
    selected = passed[:settings.final_k]
    passed_ids, selected_ids = {c.chunk_id for c in passed}, {c.chunk_id for c in selected}
    decisions = []
    for chunk in chunks:
        passes, chosen = chunk.chunk_id in passed_ids, chunk.chunk_id in selected_ids
        decisions.append({**asdict(chunk), "retrieval_rank": chunk.rank, "reranker_rank": None,
            "reranker_score": None, "filter_applied": settings.filter_enabled, "passed_filter": passes,
            "selected": chosen, "included_in_context": False, "cited": False,
            "rejection_reason": "below_threshold" if not passes else "final_k_limit" if not chosen else None})
    return selected, decisions
