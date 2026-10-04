from uuid import uuid4
import os
import streamlit as st
from document_indexer.chat import build_chat
from document_indexer.chat_storage import ChatStore
from document_indexer.chat_memory import FIELDS


@st.cache_resource
def service(root, mode):
    return build_chat(root, retrieval_mode=mode)


def render(root):
    st.title('Мини-чат с RAG и памятью задачи')
    st.caption('Документы подтверждают факты. История и память помогают понимать задачу.')
    store = ChatStore(root / 'output/chat.sqlite')
    with st.sidebar:
        if st.button('Новый диалог', key='new_conversation'):
            st.session_state['conversation'] = store.create()
            st.rerun()
        conversations = store.conversations()
        if not conversations:
            st.info('Создайте первый диалог.')
            return
        ids = [c['id'] for c in conversations]
        titles = {c['id']: c['title'] + ' · ' + c['id'][:6] for c in conversations}
        if st.session_state.get('conversation') not in ids:
            st.session_state['conversation'] = ids[0]
        cid = st.selectbox('Диалоги', ids, format_func=titles.get, key='conversation')
        mode = st.radio('Обработка фрагментов', ['reranker', 'filter'],
                        index=1 if os.getenv('CHAT_RETRIEVAL_MODE') == 'filter' else 0,
                        format_func=lambda value: 'Реранкер' if value == 'reranker' else 'Cosine-фильтр', key='chat_retrieval_mode')
        st.caption('Реранкер: ранжирование без порога logits. Cosine-фильтр: унаследованный порог, не откалиброван.')
    # This lock is brief unless another process is actively handling this conversation.
    with store.serial(cid):
        pass
    main, memory = st.columns([3, 1])
    with memory:
        st.subheader('Память задачи')
        state = store.state(cid)
        st.caption('Ревизия ' + str(state['revision']))
        labels = ['Цель', 'Уточнения', 'Ограничения', 'Термины', 'Решения', 'Открытые вопросы']
        for field, label in zip(FIELDS, labels):
            st.write(label)
            values = [i for i in state[field] if i['active']]
            if not values:
                st.caption('Не задано')
            for item in values:
                st.text((item['key'] + ' — ' if field == 'terms' else '') + item['value'])
                st.caption(('Предположение' if item['origin'] == 'inferred' else 'Указано пользователем') + ' · ' + item['message_id'][:8])
        with st.expander('Резюме и история изменений'):
            st.text(state['summary'])
            st.json(store.revisions(cid))
        if st.button('Сбросить память', key='reset_memory'):
            store.reset(cid)
            st.rerun()
        st.caption('Сброс сохраняет историю. Прежние сообщения не восстанавливают забытые требования.')
    with main:
        messages = store.history(cid)
        for message in messages:
            with st.chat_message(message['role']):
                st.text(message['content'])
                if message['role'] == 'assistant':
                    result = message['result'] or {}
                    st.caption(result.get('status', message['status']))
                    if result.get('status') == 'error':
                        st.error('Не удалось завершить обработку. Диагностика сохранена.')
                    sources = result.get('sources', [])
                    if not sources:
                        st.caption('Источники: нет подтверждённых источников')
                    for index, source in enumerate(sources):
                        with st.expander('Источник ' + str(index + 1) + ': ' + source['source']):
                            st.text('Раздел: ' + str(source['section']))
                            st.text('chunk_id: ' + source['chunk_id'])
                            st.text(source['quote'])
                    with st.expander('Диагностика retrieval'):
                        st.json(result.get('diagnostics', {}))
        last = next((m for m in reversed(messages) if m['role'] == 'user'), None)
        retry = bool(last and last['status'] == 'failed' and st.button('Повторить последнее сообщение'))
        content = st.chat_input('Ваш вопрос или уточнение', max_chars=4000)
        if content or retry:
            rid = last['request_id'] if retry else str(uuid4())
            text = last['content'] if retry else content
            try:
                with st.spinner('Обновляю память, ищу документы и проверяю цитаты…'):
                    service(str(root), mode).send_message(cid, rid, text, retry=retry)
                st.rerun()
            except ValueError as exc:
                # Only our bounded validation errors; provider errors use structured codes.
                st.error(str(exc))
            except Exception:
                st.error('Не удалось подготовить чат. Проверьте индекс и настройки модели.')
