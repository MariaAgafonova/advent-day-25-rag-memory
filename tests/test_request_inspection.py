from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch
import json
import sys
import pytest
from streamlit.testing.v1 import AppTest
from document_indexer.llm_client import ChatCompletionsClient, LLMConfig
from document_indexer.rag_models import GenerationSettings
from document_indexer.rag import RAGService
from document_indexer.storage import load_index
from document_indexer.experiment_storage import ExperimentStorage
from document_indexer.request_inspection import inspect_request, request_json, request_diff, parameter_comparison

def mock_client(status=200):
    transport = Mock()
    transport.post.return_value.status_code = status
    transport.post.return_value.json.return_value = {"choices": [{"message": {"content": "Ответ"}, "finish_reason": "stop"}]}
    config = LLMConfig("TEST_AUTH_NOT_FOR_STORAGE", "https://example.com/custom/v1", "configured-model", 90, thinking="disabled")
    return ChatCompletionsClient(config, transport), transport

@pytest.mark.parametrize("status", [200, 401])
def test_captured_request_matches_wire_without_auth(status):
    client, transport = mock_client(status)
    messages = [{"role": "system", "content": "Инструкция"}, {"role": "user", "content": "Вопрос?"}]
    result = client.complete(messages, GenerationSettings(max_output_tokens=1024))
    sent = transport.post.call_args
    assert result.api_request["body"] == sent.kwargs["json"]
    assert result.api_request["url"] == sent.args[0]
    assert result.api_request["timeout_seconds"] == sent.kwargs["timeout"]
    assert result.api_request["body"]["thinking"] == {"type": "disabled"}
    serialized = json.dumps(result.api_request)
    assert "TEST_AUTH_NOT_FOR_STORAGE" not in serialized and "Authorization" not in serialized
    original = deepcopy(result.api_request)
    messages[1]["content"] = "Changed afterwards"
    assert result.api_request == original

@pytest.fixture
def captured_run(built, service, tmp_path):
    client, transport = mock_client()
    store = ExperimentStorage(tmp_path / "requests")
    index = load_index(built[1]["structural"])
    run = RAGService(client, store, service).compare_answers("Что означает alpha?", index, GenerationSettings())
    return run, store, transport

def test_persisted_requests_and_diff(captured_run):
    run, store, transport = captured_run
    left, right = store.get_run(run["run_id"])["answers"]
    assert left["api_request"]["body"] == transport.post.call_args_list[0].kwargs["json"]
    assert right["api_request"]["body"] == transport.post.call_args_list[1].kwargs["json"]
    assert "one.md" not in request_json(left) and "one.md" in request_json(right)
    diff = request_diff(left, right)
    assert "BEGIN_UNTRUSTED_CONTEXT_JSON" in diff
    assert "+" in diff and "--- Без RAG" in diff and "+++ С RAG" in diff
    assert all(row["Совпадает"] for row in parameter_comparison(left, right))
    assert inspect_request(left)["captured"]
    assert transport.post.call_count == 2
    # Inspection does not consult a changed provider configuration.
    with patch("document_indexer.llm_client.load_configuration", side_effect=AssertionError("No current config")):
        assert "https://example.com/custom/v1/chat/completions" in request_json(left)

def test_legacy_requests_are_not_presented_as_captured(captured_run):
    run, _, _ = captured_run
    old = deepcopy(run["answers"][0])
    old.pop("api_request")
    old["generation_settings"].pop("output_token_parameter")
    view = inspect_request(old)
    assert not view["captured"] and "URL" in view["notice"]
    assert "url" not in view["request"]
    assert "max_tokens" not in view["request"]["body"]
    assert view["request"]["body"]["messages"] == old["messages"]

def test_request_comparison_ui_does_not_send_again(captured_run):
    run, store, transport = captured_run
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "ui"))
    with patch("views.rag_query.ExperimentStorage", return_value=store):
        app = AppTest.from_file(str(root / "ui/app.py"), default_timeout=30).run()
        app.sidebar.radio[0].set_value("RAG-запрос").run()
        assert not app.exception and not app.error
        assert any(e.label == "Сравнить запросы к API" for e in app.expander)
        assert any(t.label == "Различия" for t in app.tabs)
        code = "\n".join(c.value for c in app.code)
        assert "https://example.com/custom/v1/chat/completions" in code
        assert "BEGIN_UNTRUSTED_CONTEXT_JSON" in code
        assert "TEST_AUTH_NOT_FOR_STORAGE" not in code
        app.run()
        assert not app.exception and transport.post.call_count == 2
