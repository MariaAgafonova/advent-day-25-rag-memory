from copy import deepcopy
from dataclasses import asdict, replace
from unittest.mock import Mock
import json
import pytest
from document_indexer.post_retrieval import PipelineSettings, MODES, process_candidates
from document_indexer.query_rewrite import QueryRewriter
from document_indexer.llm_client import LLMResponse
from document_indexer.rag_models import GenerationSettings
from document_indexer.rag import RAGService
from document_indexer.storage import load_index
from document_indexer.experiment_storage import ExperimentStorage
from document_indexer.day23_evaluation import (prepare_calibration, calibrate_filter, label_pair,
    validate_split, validate_calibration, evaluate_day23, summarize_day23, export_day23_report, evidence_checks)
from test_rag import dataset

class Day23LLM:
    model = "test-only"
    provider = "test"
    def __init__(self):
        self.calls = []
        self.rewrite_text = '{"rewritten_query":"search alpha"}'
        self.answer_text = "Ответ [S1]"
    def complete(self, messages, settings):
        self.calls.append((deepcopy(messages), settings))
        is_rewrite = '"original_query"' in messages[-1]["content"]
        return LLMResponse(self.rewrite_text if is_rewrite else self.answer_text, self.provider, self.model,
                           1, {"total_tokens": 12, "prompt_tokens": 8, "completion_tokens": 4}, "stop", attempts=1)

@pytest.fixture
def setup23(built, service, tmp_path):
    index = load_index(built[1]["structural"])
    store = ExperimentStorage(tmp_path / "day23")
    client = Day23LLM()
    return RAGService(client, store, service), client, store, index

def ask(fixture, mode="baseline", threshold=None, **kwargs):
    rag, client, store, index = fixture
    pipeline = PipelineSettings(mode=mode, similarity_threshold=threshold, **kwargs)
    return rag.run_query("original question?", mode, GenerationSettings(), index, pipeline=pipeline)

@pytest.mark.parametrize("mode,calls,query", [("baseline", 1, "original question?"), ("rewrite", 2, "search alpha"),
    ("filter", 1, "original question?"), ("rewrite_filter", 2, "search alpha")])
def test_modes_query_generation_and_order(setup23, monkeypatch, mode, calls, query):
    rag, client, store, index = setup23
    original_encode = rag.embedding_service.encode_query
    queries, stages = [], []
    def encode(q):
        queries.append(q)
        return original_encode(q)
    monkeypatch.setattr(rag.embedding_service, "encode_query", encode)
    if mode in {"baseline", "rewrite"}:
        monkeypatch.setattr("document_indexer.post_retrieval.similarity_filter", Mock(side_effect=AssertionError("filter called")))
    if mode in {"baseline", "filter"}:
        monkeypatch.setattr(QueryRewriter, "rewrite", Mock(side_effect=AssertionError("rewrite called")))
    run = rag.run_query("original question?", mode, GenerationSettings(), index,
        progress=lambda stage, *a: stages.append(stage), pipeline=PipelineSettings(mode=mode, similarity_threshold=0 if "filter" in mode else None))
    answer = run["answers"][0]
    assert answer["status"] == "completed", answer["error"]
    assert queries == [query] and len(client.calls) == calls
    assert "original question?" in client.calls[-1][0][-1]["content"]
    assert "search alpha" not in client.calls[-1][0][-1]["content"]
    expected = (["rewrite"] if mode in {"rewrite", "rewrite_filter"} else []) + ["retrieval", "post_retrieval", "context", "generation", "готово: " + mode]
    assert stages == expected
    assert answer["original_query"] == "original question?"

def test_threshold_equality_order_and_unfiltered_baseline(setup23):
    base = ask(setup23)["answers"][0]
    filtered = ask(setup23, "filter", 1)["answers"][0]
    assert all(c["retrieval_score"] == 1 for c in filtered["candidates"] if c["passed_filter"])
    again = ask(setup23)["answers"][0]
    assert base["retrieved_chunks"] == again["retrieved_chunks"]
    assert base["counts"]["passed_filter"] == again["counts"]["passed_filter"]
    assert filtered["counts"]["passed_filter"] < base["counts"]["passed_filter"]
    ranks = [c["retrieval_rank"] for c in filtered["candidates"] if c["passed_filter"]]
    assert ranks == sorted(ranks)
    assert all(c["reranker_score"] is None for c in filtered["candidates"])

@pytest.mark.parametrize("kwargs", [{"retrieval_k": 0}, {"final_k": 0}, {"retrieval_k": 2, "final_k": 3},
    {"retrieval_k": True}, {"final_k": 2.2}, {"filter_method": "cross_encoder"},
    {"mode": "filter"}, {"similarity_threshold": float("nan")}, {"similarity_threshold": float("inf")},
    {"reranker_threshold": .5}, {"similarity_threshold": 1.1}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        PipelineSettings(**kwargs).validate()

def test_all_filtered_no_generation_and_no_returned_sources(setup23, monkeypatch):
    import document_indexer.day23_pipeline as pipeline
    original = pipeline.search
    def low_scores(*a, **kw):
        output = original(*a, **kw)
        for r in output["results"]:
            r["score"] = .4
        return output
    monkeypatch.setattr(pipeline, "search", low_scores)
    run = ask(setup23, "filter", .5)
    result = run["answers"][0]
    assert result["status"] == "no_relevant_context" and run["status"] == "completed"
    assert result["generation_skipped"] and not setup23[1].calls
    assert not result["context_items"] and not result["citations"]
    assert result["retrieved_chunks"] and all(c["rejection_reason"] == "below_threshold" for c in result["candidates"])
    assert summarize_day23(run)["generation_calls"] == 0

def test_retrieval_failure_not_empty_result(setup23, monkeypatch):
    monkeypatch.setattr("document_indexer.day23_pipeline.search", Mock(side_effect=RuntimeError("SECRET")))
    a = ask(setup23)["answers"][0]
    assert a["status"] == "failed" and a["error"]["stage"] == "retrieval"
    assert "SECRET" not in json.dumps(a)

@pytest.mark.parametrize("response", ["", "not json", "[]", '{"rewritten_query":""}', '{"rewritten_query":4}', '{"rewritten_query":"ok","answer":"bad"}', json.dumps({"rewritten_query": "x" * 2001})])
def test_rewrite_fallback(setup23, response):
    setup23[1].rewrite_text = response
    a = ask(setup23, "rewrite")["answers"][0]
    assert a["rewrite"]["status"] == "fallback" and a["rewrite"]["fallback_reason"]
    assert a["original_query"] == a["rewritten_query"]
    assert a["status"] == "completed"

def test_rewrite_token_limit_and_error_redaction(setup23):
    rag, client, _, _ = setup23
    embedding = Mock(limit=3)
    embedding.model.tokenizer.return_value = {"input_ids": [1, 2, 3, 4]}
    result = QueryRewriter(client).rewrite("q", GenerationSettings(), embedding)
    assert result.fallback_reason == "embedding_input_too_long"
    client.complete = Mock(side_effect=ValueError("SECRET"))
    result = QueryRewriter(client).rewrite("q", GenerationSettings())
    assert "SECRET" not in json.dumps(asdict(result)) and result.status == "fallback"

def test_context_citation_ids_after_selection_and_snapshot(setup23):
    setup23[1].answer_text = "Ответ [S1] и [S99]"
    run = ask(setup23, "filter", 1, final_k=1)
    a = run["answers"][0]
    assert a["invalid_citations"] == ["S99"] and a["citations"] == ["S1"]
    assert {i["chunk_id"] for i in a["context_items"]} <= {c["chunk_id"] for c in a["candidates"] if c["selected"]}
    assert [i["citation_id"] for i in a["context_items"]] == ["S1"]
    assert a["context_text"] and a["context_items"][0]["included_text"] in a["context_text"]
    before = len(setup23[1].calls)
    export_day23_report(setup23[2], run["run_id"])
    assert setup23[2].get_run(run["run_id"]) == run and len(setup23[1].calls) == before
    assert all((setup23[2].directory / (run["run_id"] + "." + ext)).exists() for ext in ("json", "csv", "md"))

def calibration_dataset():
    return {"version": "cal-v1", "questions": [{"id": "cal" + str(i), "question": "Calibration query " + str(i)} for i in range(5)]}

def prepare_and_label(fixture):
    rag, client, store, index = fixture
    settings, pipeline = GenerationSettings(), PipelineSettings()
    prepared = prepare_calibration(rag, index, settings, pipeline, calibration_dataset(), dataset(index))
    for p in prepared["pairs"]:
        label_pair(store, prepared["run_id"], p["pair_id"], "relevant" if p["observations"][0]["score"] == 1 else "partial", "Read by test fixture")
    return calibrate_filter(store, prepared["run_id"])

def test_calibration_requires_human_labels_and_split(setup23):
    rag, client, store, index = setup23
    cal = calibration_dataset()
    cal["questions"][0]["id"] = "q00"
    with pytest.raises(ValueError, match="разделены"):
        validate_split(cal, dataset(index))
    prepared = prepare_calibration(rag, index, GenerationSettings(), PipelineSettings(), calibration_dataset(), dataset(index))
    with pytest.raises(ValueError, match="ручная"):
        calibrate_filter(store, prepared["run_id"])
    assert all(p["label"] is None for p in prepared["pairs"])
    assert all(len(p["observations"]) == 2 for p in prepared["pairs"])

def test_calibration_metrics_and_compatibility(setup23):
    cal = prepare_and_label(setup23)
    assert cal["threshold"] == 1
    assert cal["partial_policy"] == "negative" and cal["table"]
    winner = next(r for r in cal["table"] if r["threshold"] == 1)
    assert winner["f1"] == winner["recall"] == winner["precision"] == 1
    rag, client, store, index = setup23
    p = PipelineSettings(similarity_threshold=1, calibration_id=cal["calibration_id"])
    validate_calibration(cal, index, p, client, GenerationSettings())
    with pytest.raises(ValueError, match="несовместима"):
        validate_calibration(cal, index, replace(p, retrieval_k=20), client, GenerationSettings())
    with pytest.raises(ValueError, match="несовместима"):
        validate_calibration(cal, index, p, client, replace(GenerationSettings(), max_context_tokens=4000))

def test_four_mode_evaluation_reuses_rewrite_no_leakage_and_review(setup23):
    cal = prepare_and_label(setup23)
    rag, client, store, index = setup23
    client.calls.clear()
    pipeline = PipelineSettings(similarity_threshold=cal["threshold"], calibration_id=cal["calibration_id"])
    run = evaluate_day23(rag, index, dataset(index), GenerationSettings(), pipeline)
    assert run["status"] == "completed" and len(run["answers"]) == 40
    stats = summarize_day23(run)
    assert len(client.calls) == 50 and stats["generation_calls"] == 40 and stats["rewrite_calls"] == 10
    assert stats["rewrite_usage"]["known_total_tokens"] == 120 and stats["generation_usage"]["known_total_tokens"] == 480
    assert all(m["correctness"] is None for m in stats["modes"].values())
    assert "MARKER_EXPECTATION" not in json.dumps([messages for messages, settings in client.calls])
    for i in range(10):
        rows = run["answers"][i*4:i*4+4]
        assert [a["mode"] for a in rows] == list(MODES)
        assert rows[1]["rewrite"]["rewrite_id"] == rows[3]["rewrite"]["rewrite_id"]
        assert all(a["generation_settings"] == rows[0]["generation_settings"] for a in rows)
        assert all(a["pipeline_settings"]["retrieval_k"] == 15 and a["pipeline_settings"]["final_k"] == 5 for a in rows)
    a = run["answers"][0]
    store.save_review(a["request_id"], {"correctness": 2, "facts": {"f1": True}, "contradictions": 0,
        "unsupported_claims": 0, "citation_support": True, "recognized_insufficiency": None, "rejected_false_premise": None, "comment": "human"})
    assert summarize_day23(store.get_run(run["run_id"]))["modes"]["baseline"]["correctness"] == 2
    assert evidence_checks(a, dataset(index)["questions"][0])[0]["included"]
    export_day23_report(store, run["run_id"])
    assert len(client.calls) == 50

def test_main_evaluation_blocked_without_calibration(setup23):
    rag, client, store, index = setup23
    with pytest.raises(ValueError, match="калибровка"):
        evaluate_day23(rag, index, dataset(index), GenerationSettings(), PipelineSettings())
    assert not client.calls and not store.list_runs()

def test_ui_navigation_and_saved_results_never_call_llm(setup23, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from unittest.mock import patch
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "ui"))
    ask(setup23, "rewrite")
    before = len(setup23[1].calls)
    with patch("views.day23.ExperimentStorage", return_value=setup23[2]), patch("document_indexer.llm_client.ChatCompletionsClient.complete", side_effect=AssertionError("Unexpected API call")):
        app = AppTest.from_file(str(root / "ui/app.py"), default_timeout=30).run()
        app.sidebar.radio[0].set_value("Реранкинг").run()
        next(c for c in app.checkbox if c.key == "day23_advanced").check().run()
        assert not app.exception and not app.error, [e.value for e in app.error]
        next(r for r in app.radio if r.label == "Действие").set_value("Эксперимент").run()
        assert not app.exception and not app.error
        assert next(b for b in app.button if b.label.startswith("Запустить 4")).disabled
        next(r for r in app.radio if r.label == "Действие").set_value("Калибровка").run()
        assert not app.exception and not app.error
        assert len(setup23[1].calls) == before
