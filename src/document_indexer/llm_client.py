from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit
from time import perf_counter, sleep
from copy import deepcopy
import os
import requests
from dotenv import dotenv_values
from .rag_models import GenerationSettings

@dataclass(frozen=True)
class LLMConfig:
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    model: str = ""
    timeout_seconds: float = 60
    output_token_parameter: str = "max_tokens"
    thinking: str | None = None

    def missing(self):
        return [name for name, value in [("LLM_API_KEY", self.api_key), ("LLM_BASE_URL", self.base_url), ("LLM_MODEL", self.model)] if not value.strip()]

    def validate(self):
        if self.missing():
            raise ValueError("Заполните " + ", ".join(self.missing()) + " в .env")
        url = urlsplit(self.base_url)
        if url.scheme not in {"https", "http"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("LLM_BASE_URL должен быть URL API без ключей, query и credentials")
        if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Для удалённого провайдера требуется HTTPS")
        if not 1 <= self.timeout_seconds <= 300:
            raise ValueError("Таймаут должен быть от 1 до 300 секунд")
        if self.output_token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Неизвестный LLM_OUTPUT_TOKEN_PARAMETER")
        if self.thinking not in {None, "enabled", "disabled"}:
            raise ValueError("LLM_THINKING / DEEPSEEK_THINKING: enabled или disabled")

    @property
    def provider(self):
        return urlsplit(self.base_url).hostname or "not-configured"

def load_configuration(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    # Explicit path; do not search sibling projects or mutate process environment.
    local_values = dotenv_values(root / ".env", interpolate=False)
    aliases = {"LLM_API_KEY": "DEEPSEEK_API_KEY", "LLM_BASE_URL": "DEEPSEEK_BASE_URL",
               "LLM_MODEL": "DEEPSEEK_MODEL", "LLM_TIMEOUT_SECONDS": "DEEPSEEK_TIMEOUT_SECONDS",
               "LLM_MAX_OUTPUT_TOKENS": "DEEPSEEK_MAX_TOKENS", "LLM_THINKING": "DEEPSEEK_THINKING"}
    def get(key, default):
        for values in (os.environ, local_values):
            value = values.get(key) or values.get(aliases.get(key, ""))
            if value:
                return value
        return default
    try:
        config = LLMConfig(get("LLM_API_KEY", ""), get("LLM_BASE_URL", "").rstrip("/"), get("LLM_MODEL", ""),
                           float(get("LLM_TIMEOUT_SECONDS", 60)), get("LLM_OUTPUT_TOKEN_PARAMETER", "max_tokens"), get("LLM_THINKING", None))
        settings = GenerationSettings(float(get("LLM_TEMPERATURE", .2)), int(get("LLM_MAX_OUTPUT_TOKENS", 512)),
                                      5, int(get("RAG_MAX_CONTEXT_TOKENS", 3000)), int(get("LLM_CONTEXT_WINDOW_TOKENS", 8192)))
        settings.validate()
        return config, settings
    except (TypeError, ValueError):
        raise ValueError("Проверьте числовые настройки LLM в .env; значения не выводятся") from None

@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: float
    usage: dict | None = None
    finish_reason: str | None = None
    error: dict | None = None
    attempts: int = 0
    api_request: dict | None = None

class LLMClient(Protocol):
    model: str
    provider: str
    def complete(self, messages: list[dict], settings: GenerationSettings) -> LLMResponse: ...

class ChatCompletionsClient:
    def __init__(self, config: LLMConfig, transport=None):
        self.config = config
        self.model = config.model
        self.provider = config.provider
        self.generation_options = {"thinking": config.thinking, "output_token_parameter": config.output_token_parameter,
                                   "timeout_seconds": config.timeout_seconds}
        self.transport = transport or requests.Session()

    def complete_structured(self, messages, settings, schema):
        # DeepSeek Chat Completions supports JSON mode; schema validation stays local.
        # Unknown compatible endpoints receive the schema in the prompt without guessing capabilities.
        response_format = {'type': 'json_object'} if self.provider == 'api.deepseek.com' else None
        return self.complete(messages, settings, response_format=response_format)

    def complete(self, messages, settings, response_format=None):
        started = perf_counter()
        try:
            self.config.validate()
            settings.validate()
        except ValueError:
            return LLMResponse("", self.provider, self.model, 0, error={"code": "configuration", "message": "Проверьте настройки LLM_API_KEY, LLM_BASE_URL, LLM_MODEL и параметры генерации"})
        payload = {"model": self.model, "messages": messages, "temperature": settings.temperature,
                   self.config.output_token_parameter: settings.max_output_tokens, "stream": False}
        if self.config.thinking is not None:
            payload["thinking"] = {"type": self.config.thinking}
        if response_format is not None:
            payload['response_format'] = response_format
        # Capture the actual wire body once; never include authentication headers.
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        api_request = {"method": "POST", "url": endpoint, "timeout_seconds": self.config.timeout_seconds,
                       "body": deepcopy(payload)}
        error = None
        for attempt in range(1, 3):
            try:
                response = self.transport.post(endpoint, json=payload,
                    headers={"Authorization": "Bearer " + self.config.api_key}, timeout=self.config.timeout_seconds,
                    allow_redirects=False)
                status = response.status_code
                if status != 200:
                    temporary = status in {408, 429, 500, 502, 503, 504}
                    error = {"code": f"http_{status}", "message": f"Провайдер вернул HTTP {status}", "retryable": temporary}
                    if temporary and attempt < 2:
                        sleep(.5)
                        continue
                    break
                data = response.json()
                choice = data["choices"][0]
                text = choice["message"].get("content")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("empty content")
                usage = data.get("usage")
                usage = {k: v for k, v in (usage or {}).items() if k in {"prompt_tokens", "completion_tokens", "total_tokens"} and isinstance(v, int) and v >= 0} or None
                return LLMResponse(text.replace(self.config.api_key, "[REDACTED]"), self.provider, self.model,
                    (perf_counter() - started) * 1000, usage, str(choice.get("finish_reason", "unknown"))[:64], attempts=attempt, api_request=api_request)
            except (requests.Timeout, requests.ConnectionError):
                error = {"code": "temporary_connection", "message": "Таймаут или ошибка подключения к LLM", "retryable": True}
                if attempt < 2:
                    sleep(.5)
                    continue
            except (ValueError, KeyError, IndexError, TypeError):
                error = {"code": "invalid_response", "message": "Провайдер вернул некорректный или пустой ответ"}
            except requests.RequestException:
                error = {"code": "request_error", "message": "Запрос к провайдеру не выполнен"}
            break
        return LLMResponse("", self.provider, self.model, (perf_counter() - started) * 1000, error=error, attempts=attempt, api_request=api_request)
