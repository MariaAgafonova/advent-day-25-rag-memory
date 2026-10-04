from pathlib import Path
from unittest.mock import patch
import json
import sys
import pytest
from test_day23 import setup23
from document_indexer.rag_models import GenerationSettings
from document_indexer.post_retrieval import PipelineSettings
from document_indexer.day23_pipeline import compare
from document_indexer.day23_evaluation import report_formats
from document_indexer.llm_client import LLMConfig

def test_simple_comparison_uses_two_modes_no_rewrite_and_keeps_review(setup23):
    rag, client, storage, index = setup23
    with patch("document_indexer.query_rewrite.QueryRewriter.rewrite", side_effect=AssertionError("No rewrite")):
        run = compare(rag, "question", index, GenerationSettings(), PipelineSettings(mode="filter", similarity_threshold=.85))
    assert len(client.calls) == 2
    assert [a["mode"] for a in run["answers"]] == ["baseline", "filter"]
    assert run["metadata"]["report_modes"] == ["baseline", "filter"]
    storage.save_comparison_review(run["run_id"], "equal", "Same supported facts")
    saved = storage.get_run(run["run_id"])
    assert saved["comparison_review"]["winner"] == "equal"
    output = report_formats(saved)
    assert set(json.loads(output["json"])["summary"]["modes"]) == {"baseline", "filter"}
    assert "| Метрика | Без фильтра | С фильтром |" in output["md"]
    assert "Same supported facts" in output["md"]
    assert len(client.calls) == 2
    with pytest.raises(ValueError):
        storage.save_comparison_review(run["run_id"], None)

def test_simple_ui_click_and_history_do_not_require_calibration(setup23):
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "ui"))
    rag, client, storage, index = setup23
    class ImmediateJobs:
        job = None
        def active(self, key):
            return False
        def submit(self, key, action):
            self.job = {"status": "completed", "result": action(lambda *a: None)}
            return "simple-test-job"
        def snapshot(self, key):
            return self.job
    jobs = ImmediateJobs()
    with patch("views.day23_simple.ExperimentStorage", return_value=storage), \
         patch("views.day23_simple.RAGService", return_value=rag), \
         patch("views.day23_simple.load_index", return_value=index), \
         patch("views.day23_simple.load_configuration", return_value=(LLMConfig("test", "https://test.invalid", "test-only"), GenerationSettings())), \
         patch("views.day23_simple.MANAGER", jobs), patch("views.rag_shared.MANAGER", jobs):
        app = AppTest.from_file(str(root / "ui/app.py"), default_timeout=30).run()
        app.sidebar.radio[0].set_value('Реранкинг').run()
        assert not app.exception and not app.error
        assert not any(r.label in {"Действие", "Режим"} for r in app.radio)
        assert [n.value for n in app.number_input] == [15, 5, .85]
        next(a for a in app.text_area if a.label == "Вопрос").set_value("question")
        next(b for b in app.button if b.label == "Сравнить ответы").click().run()
        assert not app.exception and not app.error, [e.value for e in app.error]
        assert len(client.calls) == 2
        assert {s.value for s in app.subheader} >= {"Без обработки", "Фильтр"}
        next(r for r in app.radio if r.label == "Какой ответ лучше?").set_value("filter")
        next(b for b in app.button if b.label == "Сохранить оценку").click().run()
        assert not app.exception and not app.error
        rid = storage.list_runs()[0]["run_id"]
        assert storage.get_run(rid)["comparison_review"]["winner"] == "filter"
        next(n for n in app.number_input if n.label == "Порог similarity").set_value(.9).run()
        assert len(client.calls) == 2
        assert not app.exception and not app.error
        next(n for n in app.number_input if n.label == "Top-K после обработки").set_value(16).run()
        assert next(b for b in app.button if b.label == "Сравнить ответы").disabled
        assert len(client.calls) == 2
