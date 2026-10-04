from pathlib import Path
from unittest.mock import patch
import sys
import json
from streamlit.testing.v1 import AppTest
from document_indexer.rag_models import GenerationSettings
from document_indexer.post_retrieval import PipelineSettings
from test_day23 import setup23
from test_day24 import run24, payload

def test_day24_history_sources_exports_no_llm(setup23):
    run = run24(setup23)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'ui'))
    with patch('views.day24.ExperimentStorage', return_value=setup23[2]), \
         patch('document_indexer.llm_client.ChatCompletionsClient.complete', side_effect=AssertionError('Unexpected LLM')):
        app = AppTest.from_file(str(root / 'ui/app.py'), default_timeout=30).run()
        assert app.sidebar.radio[0].value == 'Мини-чат'
        app.sidebar.radio[0].set_value('Проверяемые цитаты').run()
        assert not app.exception and not app.error
        next(b for b in app.button if b.label == 'Открыть источник').click().run()
        assert not app.exception and not app.error
        assert any('<mark>alpha</mark>' in m.value for m in app.markdown)
        assert any('не выполнена' in c.value for c in app.caption)
        app.radio[0].set_value('10 контрольных вопросов').run()
        assert not app.exception and not app.error
        assert any(b.label == 'Проверить 10 вопросов' for b in app.button)
        assert not any('сравнение' in b.label.lower() for b in app.button)
        assert any('SLA' in m.value for m in app.markdown)
        app.radio[0].set_value('История').run()
        assert not app.exception and not app.error
        assert len(setup23[1].calls) == 1

def test_source_html_escapes_document():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'ui'))
    from views.day24 import source_html
    html = source_html({'text':'<script>bad</script>'}, {'quote':'bad', 'document_quote_start':8, 'document_quote_end':11})
    assert '<script>' not in html and '<mark>bad</mark>' in html
