"""Saved evidence, human review, and execution through the shared service."""
from dataclasses import replace
from html import escape
import json
import streamlit as st
from document_indexer.llm_client import load_configuration, ChatCompletionsClient
from document_indexer.storage import load_index
from document_indexer.experiment_storage import ExperimentStorage
from document_indexer.rag import RAGService
from document_indexer.post_retrieval import PipelineSettings
from document_indexer.reranker import DEFAULT_MODEL, DEFAULT_REVISION
from document_indexer.day24_evaluation import evaluate_day24, summarize, report_formats, LABELS
from document_indexer.jobs import MANAGER
from .rag_shared import poll

STATUS = {'answered': 'Ответ по источникам', 'partial': 'Частичный ответ',
          'insufficient_context': 'Недостаточно контекста'}

def source_html(document, quote, radius=400):
    start, end = quote['document_quote_start'], quote['document_quote_end']
    text = document['text']
    if text[start:end] != quote['quote']:
        raise ValueError('Цитата не совпадает с сохранённой версией документа')
    return '<pre>' + escape(text[max(0, start-radius):start]) + '<mark>' + escape(text[start:end]) + '</mark>' + escape(text[end:end+radius]) + '</pre>'

def render_answer(answer, storage, run):
    key = answer['request_id']
    st.subheader('День 24' if answer['mode'] == 'day24' else 'День 23')
    if answer.get('error'):
        st.error('Техническая ошибка: ' + answer['error']['message'])
    elif answer['status'] == 'validation_failed':
        st.error(answer['answer'])
    else:
        st.write(STATUS.get(answer.get('answer_status'), 'Ответ прежнего режима'))
        if answer.get('answer_status') == 'insufficient_context':
            st.warning(answer['answer'])
        else:
            st.markdown(answer['answer'])
    if answer.get('refusal_reason'):
        st.caption('Причина: ' + answer['refusal_reason'])
    structured = answer.get('structured_answer') or {}
    clarification = structured.get('clarification_question') or answer.get('failure_clarification')
    if clarification:
        st.info(clarification)
    if answer['mode'] == 'day24':
        validation = answer.get('validation', {})
        st.caption('Формат: ' + validation.get('structural_validation', {}).get('status', 'не проверен'))
        st.caption('Цитаты в контексте: ' + validation.get('quote_validation', {}).get('status', 'не проверены'))
        st.caption('Смысловая оценка: ' + ('выполнена вручную' if answer.get('review', {}).get('status') == 'reviewed' else 'не выполнена'))
        st.caption('Точное совпадение цитаты не доказывает, что она подтверждает утверждение.')
    for c in structured.get('claims', []):
        st.write(f"{c['claim_id']}: {c['text']} [{', '.join(c['citation_ids'])}]")
    sources = {s['source_id']: s for s in answer.get('resolved_sources', [])}
    checks = {q['citation_id']: q for q in answer.get('validation', {}).get('quote_validation', {}).get('citations', [])}
    for quote in structured.get('citations', []):
        source = sources[quote['context_id']]
        with st.container(border=True):
            st.write(f"{quote['citation_id']} · {source['title']}")
            st.text(quote['quote'])
            st.caption(f"Раздел: {source['section']} · строки {source['line_start']}–{source['line_end']} · страницы {source['page_start']}–{source['page_end']}")
            with st.expander('Технические детали'):
                st.json(source)
                st.json(checks[quote['citation_id']])
            if st.button('Открыть источник', key=key + quote['citation_id']):
                st.session_state[key + '_source'] = quote['citation_id']
            if st.session_state.get(key + '_source') == quote['citation_id']:
                doc = storage.snapshot(run['metadata']['build_id'])['documents'][source['source']]
                st.caption('Сохранённая версия: ' + source['document_hash'])
                st.markdown(source_html(doc, checks[quote['citation_id']]), unsafe_allow_html=True)
    if answer['mode'] == 'day23':
        with st.expander('Источники прежнего ответа'):
            for item in answer['context_items']:
                if item['citation_id'] in answer['citations']:
                    st.caption(f"[{item['citation_id']}] {item['title']} · {item['source']} · строки {item['line_start']}–{item['line_end']}")
                    st.text(item['included_text'])
    st.caption(f"Время: {answer['total_latency_ms'] / 1000:.2f} с · Usage: {answer.get('usage') or 'N/A'}")
    counts = answer.get('counts', {})
    st.caption(f"Найдено {counts.get('retrieved', 0)} → прошло фильтр {counts.get('passed_filter', 0)} → в контексте {counts.get('included', 0)} → процитировано {counts.get('cited', 0)}")
    with st.expander('Контекст, попытки и проверки'):
        st.json({k: answer.get(k) for k in ('candidates', 'context_items', 'context_text', 'validation', 'evidence_attempts')})

def review_form(answer, question, storage):
    if answer['status'] in {'failed', 'validation_failed'}:
        return
    key = 'review_' + answer['request_id']
    with st.expander('Ручная смысловая оценка · ' + answer['mode']):
        st.json(question)
        old = answer.get('review', {})
        if old.get('status') == 'reviewed':
            st.json(old)
        with st.form(key):
            kind = answer.get('answer_status') or st.selectbox('Тип старого ответа', list(STATUS), format_func=STATUS.get, key=key+'kind')
            claims = {}
            for c in (answer.get('structured_answer') or {}).get('claims', []):
                claims[c['claim_id']] = st.selectbox(c['claim_id'] + ': ' + c['text'], LABELS, index=None, key=key+c['claim_id'])
            baseline_claims = []
            if answer['mode'] == 'day23' and kind != 'insufficient_context':
                st.write('Выделите фактические утверждения старого ответа и оцените каждое по контексту.')
                baseline_claims = st.data_editor([{'Утверждение': '', 'Поддержка': None}], num_rows='dynamic',
                    column_config={'Поддержка': st.column_config.SelectboxColumn(options=list(LABELS), required=True)},
                    key=key+'legacy')
            correctness = st.selectbox('Корректность: 0 — неверно, 1 — частично, 2 — верно', [0, 1, 2], index=None, key=key+'correct')
            facts = {f['id']: st.selectbox('Факт: ' + f['description'], [True, False], index=None, key=key+f['id']) for f in question['expected_facts']}
            fields = {'answer_has_extra_facts': 'В answer есть новые факты вне claims?', 'contradictions': 'Есть противоречия?',
                      'answers_question': 'Ответ отвечает на вопрос?', 'context_sufficient': 'Переданный контекст достаточен?',
                      'clarification_requested': 'Есть просьба уточнить?', 'conditions_preserved': 'Условия и исключения сохранены?',
                      'sources_support': 'Выбранные источники поддерживают ответ? Для пустого отказа выберите Нет.'}
            marks = {field: st.selectbox(label, [True, False], index=None, key=key+field) for field, label in fields.items()}
            comment = st.text_area('Обоснование оценки', key=key+'comment')
            confirmed = st.checkbox('Я прочитал(а) ответ, контекст и цитаты и выполнил(а) ручную оценку', key=key+'confirm')
            if st.form_submit_button('Сохранить оценку'):
                try:
                    if baseline_claims:
                        texts = [row['Утверждение'].strip() for row in baseline_claims if row['Утверждение'].strip()]
                        if len(texts) != len(set(texts)):
                            raise ValueError('Утверждения не должны повторяться')
                        claims = {row['Утверждение'].strip(): row['Поддержка'] for row in baseline_claims if row['Утверждение'].strip()}
                    storage.save_review(answer['request_id'], dict(answer_kind=kind, claims=claims, correctness=correctness,
                        facts=facts, **marks, comment=comment, human_confirmed=confirmed))
                    st.success('Оценка сохранена. Генерация не запускалась.')
                    st.rerun()
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))

def render(root):
    st.title('Ответы с проверяемыми цитатами')
    config, defaults = load_configuration(root)
    storage = ExperimentStorage(root / 'output/day24_experiments')
    tab = st.radio('Задача', ['Вопрос', '10 контрольных вопросов', 'История'], horizontal=True)
    with st.expander('Настройки поиска и генерации'):
        mode = st.selectbox('Обработка', ['filter', 'rewrite_filter', 'reranker'])
        before = st.number_input('Top-K поиска', min_value=1, max_value=50, value=15)
        after = st.number_input('Top-K после обработки', min_value=1, max_value=50, value=5)
        if mode == 'reranker':
            cutoff = st.checkbox('Использовать порог raw logit', value=False)
            threshold = st.number_input('Порог raw logit', value=0.) if cutoff else None
            pipeline = PipelineSettings(mode=mode, retrieval_k=int(before), final_k=int(after), filter_method='cross_encoder',
                reranker_model=DEFAULT_MODEL, reranker_revision=DEFAULT_REVISION, reranker_threshold=threshold)
            st.caption('Без порога выполняется только ранжирование; предварительный отказ по score отключён.')
        else:
            threshold = st.number_input('Cosine-порог', min_value=-1., max_value=1., value=.85, step=.01)
            pipeline = PipelineSettings(mode=mode, retrieval_k=int(before), final_k=int(after), similarity_threshold=threshold)
        st.caption('Порог не откалиброван. Высокий score сам по себе не доказывает наличие ответа.')
        limit = st.number_input('Лимит токенов ответа', min_value=1, value=defaults.max_output_tokens)
        settings = replace(defaults, max_output_tokens=int(limit))
        st.caption(f'Модель: {config.model}; temperature={settings.temperature}; контекст={settings.max_context_tokens}; окно={settings.context_window_tokens}')
    def submit(question=None):
        pipeline.validate()
        config.validate()
        index = load_index(root / 'output/structural.sqlite')
        service = RAGService(ChatCompletionsClient(config), storage)
        if question is None:
            dataset = json.loads((root / 'data/day24_questions.json').read_text(encoding='utf-8'))
            operation = lambda progress: evaluate_day24(service, index, dataset, settings, pipeline, progress)
        else:
            operation = lambda progress: service.answer_with_evidence(question, index, settings, pipeline, progress)
        st.session_state['day24_job'] = MANAGER.submit('day24', operation)
    if tab == 'Вопрос':
        with st.form('day24_question_form'):
            question = st.text_area('Вопрос — можно уточнить и запустить повторно', key='day24_question')
            if st.form_submit_button('Получить ответ', disabled=bool(config.missing()) or MANAGER.active('day24')):
                if question.strip():
                    submit(question)
                else:
                    st.error('Введите вопрос')
    elif tab == '10 контрольных вопросов':
        st.write('10 контрольных вопросов для проверки ответов, источников и точных цитат Дня 24. До 10 основных генераций и до 10 исправлений; при отказе до генерации вызова нет.')
        dataset = json.loads((root / 'data/day24_questions.json').read_text(encoding='utf-8'))
        with st.expander('Контрольные вопросы', expanded=True):
            for n, question in enumerate(dataset['questions'], 1):
                st.write(f"{n}. {question['question']}")
        if st.button('Проверить 10 вопросов', disabled=bool(config.missing()) or MANAGER.active('day24')):
            submit()
    poll('day24')
    runs = [r for r in storage.list_runs() if r['kind'].startswith('day24')]
    if tab != 'История':
        runs = [r for r in runs if storage.get_run(r['run_id'])['metadata'].get('report_modes') == ['day24']
                and (tab != '10 контрольных вопросов' or r['kind'] == 'day24_evaluation')]
    if not runs:
        st.info('Пока нет результатов')
        return
    ids = [r['run_id'] for r in runs]
    if st.session_state.get('day24_run') not in ids:
        st.session_state['day24_run'] = ids[0]
    selected = st.selectbox('Сохранённый эксперимент — без LLM', ids, key='day24_run')
    run = storage.get_run(selected)
    if 'day23' in run['metadata'].get('report_modes', []):
        st.caption('Архивное сравнение Дней 23 и 24. Новые проверки выполняются только для Дня 24.')
    with st.expander('Метрики: числители и знаменатели'):
        st.json(summarize(run))
    qids = list(dict.fromkeys(a['question_id'] for a in run['answers']))
    if not qids:
        return
    questions = {q['id']: q for q in run['metadata'].get('questions', [])}
    qid = st.selectbox('Результат вопроса', qids,
        format_func=lambda q: q + (' · ' + questions[q]['question'] if q in questions else ''))
    answers = [a for a in run['answers'] if a['question_id'] == qid]
    for col, answer in zip(st.columns(len(answers)), answers):
        with col:
            render_answer(answer, storage, run)
            if qid in questions:
                review_form(answer, questions[qid], storage)
    refused = next((a for a in answers if a.get('answer_status') in {'partial', 'insufficient_context'} or a['status'] == 'validation_failed'), None)
    if refused:
        with st.form('refine_' + refused['request_id']):
            refined = st.text_area('Уточните вопрос', value=refused['question'])
            if st.form_submit_button('Повторить после уточнения', disabled=bool(config.missing()) or MANAGER.active('day24')):
                submit(refined)
    with st.expander('Экспорт JSON / CSV / Markdown'):
        snapshot = storage.snapshot(run['metadata']['build_id'])
        for ext, content in report_formats(run, snapshot).items():
            st.download_button(ext.upper(), content, f"{run['run_id']}.{ext}", key=selected+ext)
