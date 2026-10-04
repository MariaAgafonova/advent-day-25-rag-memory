import os
from unittest.mock import Mock, patch
import pytest
from document_indexer.llm_client import load_configuration, LLMConfig, ChatCompletionsClient
from document_indexer.rag_models import GenerationSettings

def test_deepseek_aliases_and_environment_priority(tmp_path):
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=unit-test-key\nDEEPSEEK_BASE_URL=https://api.deepseek.com\nDEEPSEEK_MODEL=deepseek-flash\nDEEPSEEK_TIMEOUT_SECONDS=90\nDEEPSEEK_MAX_TOKENS=1024\nDEEPSEEK_THINKING=disabled\n", encoding="utf-8")
    with patch.dict(os.environ, {}, clear=True):
        config, settings = load_configuration(tmp_path)
        config.validate()
        assert config.model == "deepseek-flash" and config.thinking == "disabled"
        assert config.timeout_seconds == 90 and settings.max_output_tokens == 1024
        assert config.api_key == "unit-test-key"
    with patch.dict(os.environ, {"LLM_MODEL": "environment-model", "DEEPSEEK_MAX_TOKENS": "128"}, clear=True):
        config, settings = load_configuration(tmp_path)
        assert config.model == "environment-model" and settings.max_output_tokens == 128

def test_thinking_payload_and_safe_settings():
    transport = Mock()
    transport.post.return_value.status_code = 200
    transport.post.return_value.json.return_value = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    client = ChatCompletionsClient(LLMConfig("unit-test-key", "https://api.deepseek.com", "deepseek-flash", 90, thinking="disabled"), transport)
    assert not client.complete([], GenerationSettings(max_output_tokens=1024)).error
    payload = transport.post.call_args.kwargs["json"]
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == 1024 and payload["model"] == "deepseek-flash"
    assert transport.post.call_args.kwargs["timeout"] == 90
    assert "unit-test-key" not in str(client.generation_options)
    generic = ChatCompletionsClient(LLMConfig("unit-test-key", "https://example.com/v1", "model"), transport)
    generic.complete([], GenerationSettings())
    assert "thinking" not in transport.post.call_args.kwargs["json"]

def test_invalid_thinking_rejected():
    with pytest.raises(ValueError, match="THINKING"):
        LLMConfig("unit-test-key", "https://api.deepseek.com", "deepseek-flash", thinking="invalid").validate()
