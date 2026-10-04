from dataclasses import replace
from unittest.mock import Mock, patch
from pathlib import Path
import json
import pytest
from test_day23 import setup23, ask, prepare_and_label
from test_rag import dataset
from document_indexer.post_retrieval import PipelineSettings
from document_indexer.rag_models import GenerationSettings
from document_indexer.day23_pipeline import compare
from document_indexer.day23_evaluation import summarize_day23, evidence_checks

def test_empty_search_and_budget_are_distinct(setup23, monkeypatch):
    with patch("document_indexer.day23_pipeline.search", return_value={"results": []}):
        empty = ask(setup23)["answers"][0]
    assert empty["status"] == "no_results" and empty["generation_skipped"]
    rag, client, store, index = setup23
    run = rag.run_query("q", "baseline", replace(GenerationSettings(), max_context_tokens=1), index)
    assert run["answers"][0]["status"] == "empty_context"
    assert all(c["rejection_reason"] == "context_budget" for c in run["answers"][0]["candidates"])
    assert not client.calls

def test_comparison_keeps_independent_candidates(setup23):
    rag, client, store, index = setup23
    run = compare(rag, "question", index, GenerationSettings(), PipelineSettings(mode="filter", similarity_threshold=1))
    base, filtered = run["answers"]
    assert base["mode"] == "baseline" and filtered["mode"] == "filter"
    assert base["counts"]["passed_filter"] == 2 and filtered["counts"]["passed_filter"] == 1
    assert base["retrieved_chunks"] == filtered["retrieved_chunks"]

def test_false_rejection_records_removed_evidence(setup23):
    run = ask(setup23)
    a = run["answers"][0]
    question = dataset(setup23[-1])["questions"][0]
    for c in a["candidates"]:
        c["passed_filter"] = False
    checks = evidence_checks(a, question)
    assert checks[0]["retrieved"] and checks[0]["false_rejection"] and not checks[0]["passed_filter"]

def test_generation_error_is_not_context_refusal(setup23):
    from document_indexer.llm_client import LLMResponse
    setup23[1].complete = Mock(return_value=LLMResponse("", "test", "test", 1, error={"code": "http_401", "message": "auth"}, attempts=1))
    a = ask(setup23)["answers"][0]
    assert a["status"] == "failed" and a["error"]["stage"] == "generation"
    assert a["context_items"] and a["generation_calls"] == 1

def test_unknown_usage_is_not_reported_as_known_zero(setup23):
    from document_indexer.llm_client import LLMResponse
    setup23[1].complete = Mock(return_value=LLMResponse("answer [S1]", "test", "test", 1, usage=None, attempts=1))
    stats = summarize_day23(ask(setup23))
    assert stats["generation_usage"]["unknown_usage_calls"] == 1

def test_cli_export_without_api_configuration(setup23, monkeypatch, capsys):
    from document_indexer.cli import main
    run = ask(setup23)
    monkeypatch.setattr("document_indexer.day23_cli.load_configuration", Mock(side_effect=AssertionError("No config needed")))
    assert main(["export-day23-report", "--run-id", run["run_id"], "--output", str(setup23[2].directory)]) == 0

def test_checked_calibration_cannot_be_silently_replaced(setup23):
    cal = prepare_and_label(setup23)
    rag, client, store, index = setup23
    run = rag.run_query("q", "filter", GenerationSettings(), index,
        pipeline=PipelineSettings(mode="filter", similarity_threshold=.2, calibration_id=cal["calibration_id"]))
    a = run["answers"][0]
    assert a["status"] == "failed" and "Порог" in a["error"]["message"]
    assert a["generation_calls"] == 0

def test_actual_question_set_unchanged_and_disjoint():
    from document_indexer.day23_evaluation import validate_split
    root = Path(__file__).resolve().parents[1]
    test = json.loads((root / "data/rag_questions.json").read_text(encoding="utf-8"))
    cal = json.loads((root / "data/day23_calibration_questions.json").read_text(encoding="utf-8"))
    assert len(test["questions"]) == 10
    validate_split(cal, test)

def test_calibration_ui_saves_only_confirmed_human_label(setup23):
    from streamlit.testing.v1 import AppTest
    from document_indexer.day23_evaluation import prepare_calibration
    from test_day23 import calibration_dataset
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "ui"))
    rag, client, store, index = setup23
    prepared = prepare_calibration(rag, index, GenerationSettings(), PipelineSettings(), calibration_dataset(), dataset(index))
    count = len(client.calls)
    with patch("views.day23.ExperimentStorage", return_value=store), patch("document_indexer.llm_client.ChatCompletionsClient.complete", side_effect=AssertionError("No call on view")):
        app = AppTest.from_file(str(root / "ui/app.py"), default_timeout=30).run()
        app.sidebar.radio[0].set_value("Реранкинг").run()
        next(c for c in app.checkbox if c.key == "day23_advanced").check().run()
        next(r for r in app.radio if r.label == "Действие").set_value("Калибровка").run()
        assert not app.error and not app.exception
        next(s for s in app.selectbox if s.label == "Ручная метка").set_value("relevant")
        next(c for c in app.checkbox if c.label.startswith("Я прочитал")).check()
        next(b for b in app.button if b.label == "Сохранить метку").click().run()
        assert not app.error and not app.exception
        assert store.get_artifact(prepared["run_id"])["pairs"][0]["reviewed_by"] == "human"
        assert len(client.calls) == count
