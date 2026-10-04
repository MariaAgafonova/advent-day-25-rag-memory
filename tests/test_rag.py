from copy import deepcopy
from dataclasses import asdict, replace
from unittest.mock import Mock
import json
import pytest
from document_indexer.llm_client import ChatCompletionsClient, LLMConfig, LLMResponse
from document_indexer.rag_models import GenerationSettings, RetrievedChunk
from document_indexer.context_builder import ContextBuilder, TokenCounter, format_context
from document_indexer.rag import RAGService, check_index
from document_indexer.storage import load_index
from document_indexer.experiment_storage import ExperimentStorage
from document_indexer.rag_evaluation import validate_questions, evaluate_rag, summarize, source_checks, export_run
from document_indexer.citations import validate_citations

class FakeLLM:
    model = "test-only-unknown-tokenizer"
    provider = "unit-test"
    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first
    def complete(self, messages, settings):
        self.calls.append((deepcopy(messages), settings))
        if self.fail_first and len(self.calls) == 1:
            return LLMResponse("", self.provider, self.model, 1, error={"code": "http_503", "message": "test failure"})
        text = "Ответ по фрагменту [S1]." if "BEGIN_UNTRUSTED" in messages[-1]["content"] else "Ответ без документов."
        return LLMResponse(text, self.provider, self.model, 1, {"total_tokens": 10}, "stop")

@pytest.fixture
def rag_fixture(built, service, tmp_path):
    index = load_index(built[1]["structural"])
    client = FakeLLM()
    store = ExperimentStorage(tmp_path / "experiments")
    rag = RAGService(client, store, service)
    return rag, client, store, index

def dataset(index):
    doc = next(iter(index.documents.values()))
    a = doc.text.index("alpha")
    return {"questions": [{"id": f"q{i:02}", "question": f"Вопрос {i}?", "category": "fact", "answerable_from_corpus": True,
        "expected_behavior": "MARKER_EXPECTATION_NEVER_SENT", "expected_facts": [{"id": "f1", "description": "MARKER_EXPECTATION_NEVER_SENT", "weight": 2}],
        "forbidden_claims": ["MARKER_FORBIDDEN_NEVER_SENT"], "expected_sources": [{"source": doc.source, "document_hash": doc.content_hash,
        "start_char": a, "end_char": a+5, "evidence_excerpt": "alpha"}]} for i in range(10)]}

def test_no_rag_isolation_and_settings(rag_fixture):
    rag, client, store, index = rag_fixture
    settings = GenerationSettings()
    run = rag.compare_answers("Что означает alpha?", index, settings)
    assert len(client.calls) == 2
    assert client.calls[0][1] == client.calls[1][1]
    no_messages = json.dumps(client.calls[0][0], ensure_ascii=False)
    assert "one.md" not in no_messages and "gamma" not in no_messages and "passage" not in no_messages
    no_rag, with_rag = run["answers"]
    assert no_rag["retrieved_chunks"] == no_rag["context_items"] == []
    assert no_rag["index_version"] is None
    assert "one.md" in json.dumps(client.calls[1][0])
    assert with_rag["context_items"] and with_rag["citations"] == ["S1"]
    assert store.get_run(run["run_id"]) == run and len(client.calls) == 2

def test_failure_preserves_other_mode(rag_fixture):
    rag, client, store, index = rag_fixture
    client.fail_first = True
    run = rag.compare_answers("alpha?", index, GenerationSettings())
    assert run["status"] == "partial"
    assert run["answers"][0]["error"] and run["answers"][1]["answer"]
    again = rag.compare_answers("alpha?", index, GenerationSettings())
    assert again["run_id"] != run["run_id"]
    assert len(store.list_runs()) == 2

def test_empty_context_and_invalid_citations(rag_fixture, monkeypatch):
    rag, client, store, index = rag_fixture
    monkeypatch.setattr("document_indexer.rag.search", lambda *a, **kw: {"results": []})
    answer = rag.answer_with_rag("alpha?", index, GenerationSettings())
    assert answer.context_items == [] and answer.warnings
    assert "недостаточно" in client.calls[0][0][-1]["content"]
    assert answer.invalid_citations == ["S1"] and answer.status == "citation_error"

def test_budget_overlap_coordinates(make_doc):
    doc = make_doc("alpha beta gamma delta epsilon zeta eta theta " * 10, "txt", "x.txt")
    def chunk(a,b,rank):
        return RetrievedChunk(str(rank), doc.source, doc.title, [["section"]], 1, 1, None, None,
                              doc.text[a:b], .9, rank, a, b, doc.content_hash)
    retrieved = [chunk(0,100,1), chunk(50,150,2), chunk(50,150,3), chunk(300,420,4)]
    counter = TokenCounter("unknown-test")
    settings = GenerationSettings(max_context_tokens=370, context_window_tokens=5000)
    bundle = ContextBuilder(counter).build(retrieved, "q?", settings, {doc.source: doc})
    assert bundle.token_count <= 370
    assert bundle.token_count == counter.count(bundle.text)
    assert "estimate" in bundle.token_count_method
    occupied = set()
    for item in bundle.items:
        assert doc.text[item.start_char:item.end_char] == item.included_text
        positions = set(range(item.start_char,item.end_char))
        assert not positions & occupied
        occupied |= positions
    assert any(i.truncated for i in bundle.items)
    valid, invalid = validate_citations("[S1] [S1] [S999] [S0]", bundle.items)
    assert valid == ["S1"] and invalid == ["S999", "S0"]

def test_context_overhead_and_question_budget(make_doc):
    with pytest.raises(ValueError, match="резерв"):
        ContextBuilder(TokenCounter("unknown-test")).build([], "q" * 10000, GenerationSettings(context_window_tokens=2000), {})

def test_ground_truth_not_sent_unreviewed_not_averaged(rag_fixture):
    rag, client, store, index = rag_fixture
    run = evaluate_rag(rag, index, dataset(index), GenerationSettings())
    assert len(client.calls) == 20
    prompts = json.dumps([m for m, s in client.calls])
    assert "MARKER_EXPECTATION_NEVER_SENT" not in prompts and "MARKER_FORBIDDEN_NEVER_SENT" not in prompts
    stats = summarize(run)
    assert stats["completed_calls"] == 20 and stats["failed_calls"] == 0
    assert stats["modes"]["rag"]["correctness"] is None
    assert stats["modes"]["no-rag"]["source_support"] is None
    question = run["metadata"]["questions"][0]
    assert source_checks(run["answers"][0], question) is None
    assert source_checks(run["answers"][1], question)[0]["included"]
    store.save_review(run["answers"][0]["request_id"], {"correctness": 2, "facts": {"f1": True}, "contradictions": 0,
        "unsupported_claims": None, "citation_support": None, "recognized_insufficiency": None, "rejected_false_premise": None, "comment": "checked"})
    updated = store.get_run(run["run_id"])
    stats = summarize(updated)
    assert stats["modes"]["no-rag"]["correctness"] == 2 and stats["modes"]["no-rag"]["completeness"] == 1
    assert stats["modes"]["rag"]["correctness"] is None
    assert stats["fully_reviewed_questions"] == 0
    export_run(store, run["run_id"])
    assert len(list(store.directory.glob(run["run_id"] + ".*"))) == 3
    assert "N/A" in (store.directory / (run["run_id"] + ".csv")).read_text(encoding="utf-8")

def test_stale_annotations(rag_fixture):
    index = rag_fixture[-1]
    data = dataset(index)
    validate_questions(data, index)
    data["questions"][0]["expected_sources"][0]["document_hash"] = "stale"
    with pytest.raises(ValueError, match="Устаревшая"):
        validate_questions(data, index)

def test_snapshot_survives_source_changes(rag_fixture):
    rag, client, store, index = rag_fixture
    run = rag.compare_answers("q", index, GenerationSettings())
    snapshot = store.snapshot(run["metadata"]["build_id"])
    assert snapshot["documents"]["one.md"]["text"] == index.documents["one.md"].text

@pytest.mark.parametrize("status,calls", [(401,1),(403,1),(400,1),(404,1),(429,2),(503,2)])
def test_adapter_retry_and_secret_redaction(status, calls, monkeypatch):
    monkeypatch.setattr("document_indexer.llm_client.sleep", lambda x: None)
    transport = Mock()
    transport.post.return_value.status_code = status
    transport.post.return_value.text = "SECRET_API_KEY_DO_NOT_SAVE"
    config = LLMConfig("SECRET_API_KEY_DO_NOT_SAVE", "https://example.com/v1", "configured-model")
    client = ChatCompletionsClient(config, transport)
    response = client.complete([{"role": "user", "content": "q"}], GenerationSettings())
    assert transport.post.call_count == calls
    assert "SECRET_API_KEY_DO_NOT_SAVE" not in json.dumps(asdict(response))
    assert "SECRET_API_KEY_DO_NOT_SAVE" not in repr(config)
    assert transport.post.call_args.kwargs["timeout"] == 60
    assert not transport.post.call_args.kwargs["allow_redirects"]

def test_adapter_success_and_missing_key():
    transport = Mock()
    transport.post.return_value.status_code = 200
    transport.post.return_value.json.return_value = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}], "usage": {"total_tokens": 7}}
    client = ChatCompletionsClient(LLMConfig("secret", "https://example.com/v1", "model"), transport)
    result = client.complete([], GenerationSettings())
    assert result.text == "hello" and result.usage == {"total_tokens": 7} and not result.error
    missing = ChatCompletionsClient(LLMConfig(), transport)
    assert missing.complete([], GenerationSettings()).error["code"] == "configuration"
    assert transport.post.call_count == 1

def test_blank_question_no_provider_call(rag_fixture):
    rag, client, store, index = rag_fixture
    result = rag.answer_without_rag("   ", GenerationSettings())
    assert result.status == "failed" and not client.calls
