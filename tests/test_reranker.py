from dataclasses import replace
from unittest.mock import patch, Mock
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
import json
import sys
import pytest
from test_day23 import setup23, ask
from document_indexer.reranker import DEFAULT_MODEL, DEFAULT_REVISION, RerankerScore, RerankerError, RerankerService
from document_indexer.post_retrieval import PipelineSettings, process_candidates
from document_indexer.rag_models import RetrievedChunk, GenerationSettings
from document_indexer.day23_pipeline import compare
from document_indexer.day23_evaluation import summarize_day23, report_formats, prepare_calibration

def settings(**kwargs):
    return replace(PipelineSettings(mode="reranker", filter_method="cross_encoder", reranker_model=DEFAULT_MODEL,
        reranker_revision=DEFAULT_REVISION, reranker_threshold=0), **kwargs)

class FakeReranker:
    model_id = DEFAULT_MODEL
    revision = DEFAULT_REVISION
    max_length = 512
    def __init__(self, scores=None, truncated=False):
        self.calls = []
        self.scores = scores
        self.truncated = truncated
    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        scores = self.scores if self.scores is not None else list(range(1, len(texts) + 1))
        return [RerankerScore(s, 600 if self.truncated else 30, 512 if self.truncated else 30, self.truncated) for s in scores]

def chunks(fixture):
    return [RetrievedChunk(**c) for c in ask(fixture)["answers"][0]["retrieved_chunks"]]

def test_rerank_reorders_and_preserves_retrieval_scores(setup23):
    items = chunks(setup23)
    original = [(c.chunk_id, c.rank, c.retrieval_score) for c in items]
    fake = FakeReranker([-2., 3.])
    selected, decisions = process_candidates(items, settings(reranker_threshold=-2.), "original question", fake)
    assert [c.chunk_id for c in selected] == [items[1].chunk_id, items[0].chunk_id]
    assert [c["reranker_rank"] for c in decisions] == [2, 1]
    assert [c["reranker_score"] for c in decisions] == [-2., 3.]
    assert all(c["passed_filter"] for c in decisions)  # Equality to raw threshold passes.
    assert [(c.chunk_id, c.rank, c.retrieval_score) for c in items] == original
    assert fake.calls == [("original question", [c.text for c in items])]

def test_ties_use_original_rank_and_final_k(setup23):
    items = chunks(setup23)
    selected, decisions = process_candidates(items, settings(final_k=1, reranker_threshold=None), "q", FakeReranker([2, 2]))
    assert selected == [items[0]]
    assert decisions[1]["rejection_reason"] == "final_k_limit"
    assert [c["retrieval_rank"] for c in decisions] == [1, 2]

def test_reranker_runs_after_rewrite_with_original_question(setup23):
    rag, client, store, index = setup23
    rag.reranker_service = FakeReranker(truncated=True)
    p = settings(mode="rewrite_filter")
    run = rag.run_query("Исходный вопрос", p.mode, GenerationSettings(), index, pipeline=p)
    a = run["answers"][0]
    assert a["rewritten_query"] == "search alpha"
    assert rag.reranker_service.calls[0][0] == "Исходный вопрос"
    assert "Исходный вопрос" in client.calls[-1][0][-1]["content"]
    assert a["warnings"] and all(c["reranker_truncated"] for c in a["candidates"])
    assert a["processed_chunks"][0]["reranker_rank"] == 1

def test_baseline_does_not_load_reranker_even_in_comparison(setup23):
    rag, client, store, index = setup23
    rag.reranker_service = FakeReranker()
    run = compare(rag, "q", index, GenerationSettings(), settings())
    assert [a["mode"] for a in run["answers"]] == ["baseline", "reranker"]
    assert len(rag.reranker_service.calls) == 1 and len(client.calls) == 2
    assert all(c["reranker_score"] is None for c in run["answers"][0]["candidates"])
    assert run["answers"][1]["candidates"][1]["reranker_rank"] == 1
    store.save_comparison_review(run["run_id"], "reranker", "Checked sources")
    saved = store.get_run(run["run_id"])
    assert saved["comparison_review"]["winner"] == "reranker"
    assert "reranker" in summarize_day23(saved)["modes"]
    exported = report_formats(saved)
    assert "Реранкер" in exported["md"]
    assert set(json.loads(exported["json"])["summary"]["modes"]) == {"baseline", "reranker"}
    assert len(rag.reranker_service.calls) == 1 and len(client.calls) == 2
    with pytest.raises(ValueError):
        store.save_comparison_review(run["run_id"], "filter")

@pytest.mark.parametrize("change", [{"similarity_threshold": .85}, {"reranker_threshold": float("nan")},
    {"reranker_revision": "main"}, {"reranker_model": None}, {"reranker_max_length": 513},
    {"reranker_max_length": True}, {"filter_method": "similarity_threshold"}, {"calibration_id": "cosine-calibration"}])
def test_separate_score_types_and_model_revision(change):
    with pytest.raises(ValueError):
        settings(**change).validate()

def test_model_failure_is_not_no_context_and_no_fallback(setup23):
    rag, client, store, index = setup23
    rag.reranker_service = FakeReranker()
    rag.reranker_service.score = Mock(side_effect=RuntimeError("PRIVATE_INTERNAL_ERROR"))
    run = rag.run_query("q", "reranker", GenerationSettings(), index, pipeline=settings())
    a = run["answers"][0]
    assert a["status"] == "failed" and a["error"]["stage"] == "reranker"
    assert a["retrieved_chunks"] and not client.calls
    assert "PRIVATE_INTERNAL_ERROR" not in json.dumps(a)

@pytest.mark.parametrize("scores", [[float("nan"), 1], [1]])
def test_invalid_scores_are_technical_errors(setup23, scores):
    rag, client, store, index = setup23
    rag.reranker_service = FakeReranker(scores)
    a = rag.run_query("q", "reranker", GenerationSettings(), index, pipeline=settings())["answers"][0]
    assert a["status"] == "failed" and a["error"]["stage"] == "reranker" and not client.calls

def test_raw_score_threshold_can_reject_all(setup23):
    rag, client, store, index = setup23
    rag.reranker_service = FakeReranker([-5, -1])
    a = rag.run_query("q", "reranker", GenerationSettings(), index, pipeline=settings())["answers"][0]
    assert a["status"] == "no_relevant_context" and a["generation_skipped"]
    assert not client.calls and not a["context_items"]
    assert all(c["rejection_reason"] == "below_reranker_threshold" for c in a["candidates"])

class TinyTokenizer:
    model_max_length = 512
    def num_special_tokens_to_add(self, pair=False):
        return 4 if pair else 2
    def __call__(self, first, second=None, **kwargs):
        import torch
        if isinstance(first, str):
            return {"input_ids": list(range(len(first.split())))}
        lengths = [len(q.split()) + len(t.split()) + 4 for q, t in zip(first, second)]
        if kwargs.get("return_tensors"):
            assert kwargs["truncation"] == "only_second"
            lengths = [min(x, kwargs["max_length"]) for x in lengths]
            mask = torch.tensor([[1] * n + [0] * (max(lengths) - n) for n in lengths])
            return {"input_ids": mask.clone(), "attention_mask": mask}
        return {"input_ids": [list(range(n)) for n in lengths]}

def test_input_lengths_and_raw_logits_without_model_download():
    import torch
    service = object.__new__(RerankerService)
    service.tokenizer, service.max_length, service.lock = TinyTokenizer(), 16, RLock()
    service.model = lambda **inputs: SimpleNamespace(logits=torch.tensor([[-3.5]] * len(inputs["input_ids"])))
    result = service.score("original question", ["chunk " * 30, "short passage"])
    assert result[0] == RerankerScore(-3.5, 36, 16, True)
    assert result[1] == RerankerScore(-3.5, 8, 8, False)
    with pytest.raises(RerankerError, match="вопрос"):
        service.score("query " * 13, ["passage"])

def test_cli_reranker_uses_own_threshold(setup23, capsys):
    from document_indexer.cli import main
    from document_indexer.llm_client import LLMConfig
    rag, client, store, index = setup23
    rag.reranker_service = FakeReranker()
    with patch("document_indexer.day23_cli.load_configuration", return_value=(LLMConfig("test", "https://test.invalid", "test-only"), GenerationSettings())), \
         patch("document_indexer.day23_cli.RAGService", return_value=rag), patch("document_indexer.day23_cli.load_index", return_value=index), \
         patch("document_indexer.day23_cli.ExperimentStorage", return_value=store):
        assert main(["ask", "--mode", "reranker", "--question", "q", "--index", "fake", "--reranker-threshold", "-2.5"]) == 0
    a = store.get_run(store.list_runs()[0]["run_id"])["answers"][0]
    assert a["pipeline_settings"]["similarity_threshold"] is None
    assert a["pipeline_settings"]["reranker_threshold"] == -2.5

def test_ui_two_modes_and_separate_thresholds(setup23):
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "ui"))
    rag, client, store, index = setup23
    rag.reranker_service = FakeReranker()
    run = compare(rag, "q", index, GenerationSettings(), settings())
    with patch("views.day23_simple.ExperimentStorage", return_value=store), \
         patch("document_indexer.reranker.get_reranker_service", side_effect=AssertionError("No model loading on render")), \
         patch("document_indexer.llm_client.ChatCompletionsClient.complete", side_effect=AssertionError("No LLM on render")):
        app = AppTest.from_file(str(root / "ui/app.py"), default_timeout=30).run()
        app.sidebar.radio[0].set_value('Реранкинг').run()
        toggle = next(r for r in app.radio if r.label == "Обработка")
        assert toggle.options == ["Фильтр", "Реранкер"]
        assert next(n for n in app.number_input if n.label == "Порог similarity").value == .85
        toggle.set_value("reranker").run()
        assert not app.exception and not app.error, [e.value for e in app.error]
        assert not any(n.label == "Порог реранкера (raw score)" for n in app.number_input)
        next(c for c in app.checkbox if c.label == "Отсекать по score реранкера").check().run()
        assert next(n for n in app.number_input if n.label == "Порог реранкера (raw score)").value == 0
        assert not any(n.label == "Порог similarity" for n in app.number_input)
        assert {s.value for s in app.subheader} >= {"Без обработки", "Реранкер"}
        next(r for r in app.radio if r.label == "Какой ответ лучше?").set_value("reranker")
        next(b for b in app.button if b.label == "Сохранить оценку").click().run()
        assert store.get_run(run["run_id"])["comparison_review"]["winner"] == "reranker"
        next(r for r in app.radio if r.label == "Обработка").set_value("filter").run()
        assert next(n for n in app.number_input if n.label == "Порог similarity").value == .85
        assert len(client.calls) == 2 and len(rag.reranker_service.calls) == 1
