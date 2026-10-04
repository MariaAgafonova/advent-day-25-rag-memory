from pathlib import Path
from unittest.mock import patch
import sys
from streamlit.testing.v1 import AppTest
from document_indexer.chat_storage import ChatStore
from document_indexer.chat import ChatService, Knowledge
from document_indexer.day25_mock import MockClient, MockEmbedding, MockPlanner, mock_index
from document_indexer.rag_models import GenerationSettings


def test_chat_ui_send_switch_reset_and_sources(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'ui'))
    store = ChatStore(tmp_path / 'chat.sqlite')
    service = ChatService(store, Knowledge(mock_index(), MockClient(), MockEmbedding()),
                          GenerationSettings(context_window_tokens=20000), MockPlanner())
    with patch('views.chat.ChatStore', return_value=store), patch('views.chat.service', return_value=service):
        app = AppTest.from_file(str(root / 'ui/app.py'), default_timeout=30).run()
        assert not app.exception
        app.button(key='new_conversation').click().run()
        cid = store.conversations()[0]['id']
        app.chat_input[0].set_value('Цель: составить план фоновой синхронизации Android-приложения.').run()
        assert not app.exception and not app.error
        assert any('нет подтверждённых' in c.value for c in app.caption)
        app.chat_input[0].set_value('Что делает WorkManager?').run()
        assert not app.exception and not app.error
        assert any('TEST_ONLY/workmanager.md' in e.label for e in app.expander)
        assert len(store.history(cid)) == 4
        app.button(key='new_conversation').click().run()
        other = store.conversations()[0]['id']
        app.selectbox(key='conversation').set_value(cid).run()
        app.button(key='reset_memory').click().run()
        assert not store.state(cid)['goal'] and len(store.history(cid)) == 4
        assert store.state(other)['revision'] == 0
        # History render and switching do not send additional requests.
        assert len(store.history(cid)) == 4


def test_document_and_message_html_render_as_plain_text(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'ui'))
    store = ChatStore(tmp_path / 'chat.sqlite')
    cid = store.create()
    with store.serial(cid):
        store.begin(cid, 'xss', '<script>alert(1)</script>')
        store.finish(cid, 'xss', dict(status='clarification', answer='<img src=x onerror=alert(1)>', sources=[], diagnostics={}))
    with patch('views.chat.ChatStore', return_value=store):
        app = AppTest.from_file(str(root / 'ui/app.py'), default_timeout=30).run()
        assert any('<script>' in t.value for t in app.text)
        assert not any('<script>' in m.value for m in app.markdown)
        assert not app.exception
