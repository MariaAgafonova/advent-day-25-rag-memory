from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

def test_navigation_never_starts_indexing():
    app = Path(__file__).resolve().parents[1] / "ui/app.py"
    with patch("document_indexer.indexing.build_index") as build:
        test = AppTest.from_file(str(app), default_timeout=20).run()
        assert not test.exception
        for page in ["Индексация", "Чанки", "Поиск", "Сравнение", "RAG-запрос", "Оценка RAG", "Документы"]:
            test.sidebar.radio[0].set_value(page).run()
            assert not test.exception, [e.message for e in test.exception]
        assert not build.called
