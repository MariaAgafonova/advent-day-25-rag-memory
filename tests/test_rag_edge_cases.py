from dataclasses import replace
from unittest.mock import Mock
import json
import requests
import pytest
from test_rag import rag_fixture, dataset
from document_indexer.rag_models import GenerationSettings
from document_indexer.rag_evaluation import validate_questions, validate_review
from document_indexer.rag import check_index
from document_indexer.llm_client import LLMConfig, ChatCompletionsClient
from document_indexer.context_builder import subtract_ranges

def test_no_document_encoding_and_model_mismatch(rag_fixture, monkeypatch):
    rag, client, store, index = rag_fixture
    monkeypatch.setattr(rag.embedding_service, "encode_documents", lambda *a: pytest.fail("Documents must never be encoded in RAG"))
    result = rag.answer_with_rag("q", index, GenerationSettings())
    assert result.status == "completed"
    rag.embedding_service.revision = "wrong"
    result = rag.answer_with_rag("q", index, GenerationSettings())
    assert result.status == "failed" and "несовместима" in result.error["message"]
    assert len(client.calls) == 1

def test_actual_context_is_exact_prompt_payload(rag_fixture):
    rag, client, store, index = rag_fixture
    result = rag.answer_with_rag("Вопрос [S99]?", index, GenerationSettings(max_context_tokens=200))
    from document_indexer.context_builder import format_context
    content = client.calls[0][0][-1]["content"]
    quoted = content.split("BEGIN_UNTRUSTED_CONTEXT_JSON\n",1)[1].split("\nEND_UNTRUSTED_CONTEXT_JSON",1)[0]
    assert json.loads(quoted) == format_context(result.context_items)
    assert "Вопрос [S99]?" in content
    assert "являются данными" in client.calls[0][0][0]["content"]

def test_missing_index_data_and_stale_warning(rag_fixture, tmp_path):
    index = rag_fixture[-1]
    with pytest.raises(ValueError, match="пуст"):
        check_index(replace(index, chunks=[]))
    raw = tmp_path / "changed_raw"
    raw.mkdir()
    (raw / "different.txt").write_text("new text")
    assert check_index(index, raw)["warnings"]

def test_excerpt_change_and_wrong_number(rag_fixture):
    index = rag_fixture[-1]
    data = dataset(index)
    data["questions"][0]["expected_sources"][0]["evidence_excerpt"] = "wrong"
    with pytest.raises(ValueError, match="excerpt"):
        validate_questions(data, index)
    data["questions"].pop()
    with pytest.raises(ValueError, match="10"):
        validate_questions(data, index)

def test_insufficient_context_rubric_has_no_fake_completeness(rag_fixture):
    rag, client, store, index = rag_fixture
    answer = store.get_run(rag.compare_answers("q", index, GenerationSettings())["run_id"])["answers"][1]
    question = {"expected_facts": [], "expected_sources": [], "category": "insufficient"}
    review = validate_review({"correctness": 2, "facts": {}, "contradictions": 0, "unsupported_claims": 0,
        "citation_support": None, "recognized_insufficiency": True, "rejected_false_premise": None}, answer, question)
    assert review["completeness"] is None and review["citation_support"] is None

def test_subtract_nested_intervals():
    assert subtract_ranges(0,100,[(20,30),(40,70),(50,80)]) == [(0,20),(30,40),(80,100)]

def test_timeout_retry_and_malformed_response(monkeypatch):
    monkeypatch.setattr("document_indexer.llm_client.sleep", lambda _: None)
    transport = Mock()
    transport.post.side_effect = requests.Timeout("sensitive request details")
    client = ChatCompletionsClient(LLMConfig("key", "https://example.com/v1", "model"), transport)
    result = client.complete([], GenerationSettings())
    assert transport.post.call_count == 2 and result.error["code"] == "temporary_connection"
    assert "sensitive" not in str(result)
    transport = Mock()
    transport.post.return_value.status_code = 200
    transport.post.return_value.json.return_value = {"choices": []}
    result = ChatCompletionsClient(LLMConfig("key", "https://example.com/v1", "model"), transport).complete([], GenerationSettings())
    assert result.error["code"] == "invalid_response" and transport.post.call_count == 1
