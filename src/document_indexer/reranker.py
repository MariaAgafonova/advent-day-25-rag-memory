"""Multilingual cross-encoder. Raw logits, pinned revision, explicit truncation."""
from dataclasses import dataclass
from threading import RLock
from time import perf_counter
import math
import re

DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
DEFAULT_MAX_LENGTH = 512

class RerankerError(ValueError):
    """Only safe, user-facing messages are used at this boundary."""

@dataclass(frozen=True)
class RerankerScore:
    score: float
    input_tokens: int
    used_tokens: int
    truncated: bool

class RerankerService:
    def __init__(self, model_id=DEFAULT_MODEL, revision=DEFAULT_REVISION, max_length=DEFAULT_MAX_LENGTH):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise RerankerError("Укажите точную revision реранкера (40 hex-символов)")
        started = perf_counter()
        torch.set_num_threads(min(4, torch.get_num_threads()))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision,
                trust_remote_code=False, use_safetensors=True).to("cpu").eval()
        except Exception:
            raise RerankerError("Не удалось загрузить реранкер. Проверьте доступность модели и её revision") from None
        if self.model.config.num_labels != 1 or self.model.config._commit_hash != revision:
            raise RerankerError("Реранкер должен иметь один выходной score и совпадающую revision")
        if type(max_length) is not int or not 8 <= max_length <= min(512, self.tokenizer.model_max_length):
            raise RerankerError("Недопустимый лимит входа реранкера; максимум 512 токенов")
        self.model_id, self.revision, self.max_length = model_id, revision, max_length
        self.lock = RLock()
        self.load_latency_ms = (perf_counter() - started) * 1000

    def score(self, original_query, texts):
        import torch
        if not original_query.strip():
            raise RerankerError("Для реранкера нужен исходный вопрос")
        with self.lock, torch.inference_mode():
            query_tokens = len(self.tokenizer(original_query, add_special_tokens=False)["input_ids"])
            if query_tokens + self.tokenizer.num_special_tokens_to_add(pair=True) >= self.max_length:
                raise RerankerError("Исходный вопрос превышает лимит реранкера; вопрос не обрезается")
            result = []
            for start in range(0, len(texts), 4):
                batch = texts[start:start + 4]
                queries = [original_query] * len(batch)
                full = self.tokenizer(queries, batch, truncation=False, padding=False)["input_ids"]
                features = self.tokenizer(queries, batch, padding=True, truncation="only_second",
                    max_length=self.max_length, return_tensors="pt")
                # The pinned card/config uses Identity: no sigmoid/softmax or probability claim.
                logits = self.model(**features).logits
                if tuple(logits.shape) != (len(batch), 1):
                    raise RerankerError("Реранкер вернул неверную форму scores")
                for ids, used, value in zip(full, features["attention_mask"].sum(dim=1).tolist(), logits[:, 0].tolist()):
                    if not math.isfinite(value):
                        raise RerankerError("Реранкер вернул неконечный score")
                    result.append(RerankerScore(float(value), len(ids), int(used), len(ids) > int(used)))
            return result

_services = {}
_load_lock = RLock()

def get_reranker_service(model_id=DEFAULT_MODEL, revision=DEFAULT_REVISION, max_length=DEFAULT_MAX_LENGTH):
    key = (model_id, revision, max_length)
    with _load_lock:
        if key not in _services:
            _services[key] = RerankerService(*key)
        return _services[key]
