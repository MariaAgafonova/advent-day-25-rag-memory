from pathlib import Path
from unittest.mock import patch
import sys
from streamlit.testing.v1 import AppTest
from test_rag import FakeLLM, dataset
from document_indexer.rag import RAGService
from document_indexer.rag_evaluation import evaluate_rag
from document_indexer.experiment_storage import ExperimentStorage
from document_indexer.storage import load_index

def test_saved_results_review_and_navigation_do_not_call_llm(built, service, tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "ui"))
    index = load_index(built[1]["structural"])
    client = FakeLLM()
    store = ExperimentStorage(tmp_path / "ui_experiments")
    from document_indexer.rag_models import GenerationSettings
    run = evaluate_rag(RAGService(client, store, service), index, dataset(index), GenerationSettings())
    with patch("views.rag_evaluation.ExperimentStorage", return_value=store), patch("views.rag_query.ExperimentStorage", return_value=store):
        app = AppTest.from_file(str(root / "ui/app.py"), default_timeout=30).run()
        app.sidebar.radio[0].set_value("Оценка RAG").run()
        assert not app.exception
        assert not app.error, [e.value for e in app.error]
        # The no-RAG form is the first of two answer forms.
        next(s for s in app.selectbox if s.label.startswith("Корректность")).set_value(2)
        next(c for c in app.checkbox if "MARKER_EXPECTATION" in c.label).check()
        next(b for b in app.button if b.label == "Сохранить ручную оценку").click().run()
        assert not app.exception and not app.error
        saved = store.get_run(run["run_id"])
        assert saved["answers"][0]["review"]["status"] == "reviewed"
        next(b for b in app.button if b.label.startswith("[S1]")).click().run()
        assert not app.exception and not app.error
        app.sidebar.radio[0].set_value("RAG-запрос").run()
        assert not app.exception and not app.error
        app.sidebar.radio[0].set_value("Оценка RAG").run()
        assert not app.exception and not app.error
        assert len(client.calls) == 20  # Only the explicit test setup; UI makes zero calls.
