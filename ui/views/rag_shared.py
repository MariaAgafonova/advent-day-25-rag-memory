from dataclasses import asdict, replace
import streamlit as st
from document_indexer.llm_client import load_configuration
from document_indexer.storage import load_index
from document_indexer.experiment_storage import ExperimentStorage
from document_indexer.rag import check_index
from document_indexer.rag_models import ContextItem
from document_indexer.models import Document
from document_indexer.jobs import MANAGER
from document_indexer.rag_evaluation import report_formats, summarize
from .shared import LABELS, source_html
from .api_requests import render_request

def setup(root, prefix, needs_index=True):
    config, defaults = load_configuration(root)
    configured = not config.missing()
    if not configured:
        st.warning("LLM не настроен. Заполните в .env: " + ", ".join(config.missing()))
    else:
        config.validate()
    st.caption(f"Провайдер: {config.provider} · модель: {config.model or 'не задана'}")
    index = None
    if needs_index:
        paths = [root / "output" / (s + ".sqlite") for s in ("structural", "fixed") if (root / "output" / (s + ".sqlite")).exists()]
        if not paths:
            st.info("Нет готового индекса. Постройте его на экране «Индексация» или скопируйте индекс Дня 21.")
        else:
            path = st.selectbox("Сохранённый индекс", paths, format_func=lambda p: LABELS[p.stem], key=prefix + "_index")
            index = load_index(path)
            readiness = check_index(index, root / "data/raw")
            st.caption(f"{len(index.chunks)} чанков · версия {index.config['build_id'][:8]} · {index.config['created_at'][:19]}")
            for warning in readiness["warnings"]:
                st.warning(warning)
    top_k = st.slider("Top-k: сколько чанков искать", 1, 20, defaults.top_k, key=prefix + "_topk") if needs_index else defaults.top_k
    with st.expander("Общие настройки генерации"):
        st.caption("Одна модель и одинаковые параметры для обоих режимов. История диалога не передаётся.")
        temperature = st.number_input("Температура", 0., 2., defaults.temperature, .1, key=prefix + "_temp")
        output = st.number_input("Максимум токенов ответа (отдельный резерв)", 1, 32768, defaults.max_output_tokens, key=prefix + "_output")
        context = st.number_input("Бюджет контекста", 1, 100000, defaults.max_context_tokens, key=prefix + "_context")
        st.caption(f"Окно модели: {defaults.context_window_tokens}. Укажите достоверное значение LLM_CONTEXT_WINDOW_TOKENS в .env. Без известного токенизатора используется консервативная оценка по UTF-8 байтам.")
    settings = replace(defaults, temperature=temperature, max_output_tokens=int(output), top_k=top_k, max_context_tokens=int(context))
    settings.validate()
    return config, settings, index, configured and (not needs_index or index is not None)

@st.fragment(run_every="1s")
def poll(prefix):
    jid = st.session_state.get(prefix + "_job")
    job = MANAGER.snapshot(jid) if jid else None
    if not job:
        return
    if job["status"] in {"queued", "running"}:
        p = job["progress"]
        st.info("В очереди" if job["status"] == "queued" else p.get("stage", "Подготовка"))
        if p.get("total") is not None:
            st.write(f"Выполнено {p['done']} из {p['total']}" + (" вызовов" if prefix == "rag_evaluation" else ""))
    elif job["status"] == "failed":
        st.error(job["error"])
    elif st.session_state.get(prefix + "_consumed") != jid:
        st.session_state[prefix + "_run"] = job["result"]["run_id"]
        st.session_state[prefix + "_consumed"] = jid
        st.rerun(scope="app")

def run_picker(storage, prefix, kind=None):
    rows = [r for r in storage.list_runs() if kind is None or r["kind"] == kind]
    if not rows:
        st.info("Пока нет сохранённых экспериментов")
        return None
    ids = [r["run_id"] for r in rows]
    lookup = {r["run_id"]: r for r in rows}
    key = prefix + "_run"
    if st.session_state.get(key) not in ids:
        st.session_state[key] = ids[0]
    selected = st.selectbox("Сохранённый эксперимент — открывается без вызова LLM", ids,
        format_func=lambda rid: f"{lookup[rid]['created_at'][:19]} · {lookup[rid]['status']} · {rid[:8]}", key=key)
    return storage.get_run(selected)

def render_answer(answer, storage, run, key):
    st.caption(f"{answer['model']} · {answer['provider']} · статус: {answer['status']}")
    if answer["error"]:
        st.error(answer["error"]["message"])
    if answer["answer"]:
        st.markdown(answer["answer"])
    for warning in answer["warnings"]:
        st.warning(warning)
    st.caption(f"Всего: {answer['total_latency_ms']:.0f} мс · поиск: {answer['retrieval_latency_ms']:.0f} мс · генерация: {answer['generation_latency_ms']:.0f} мс")
    st.caption("Usage: " + str(answer["usage"] or "не возвращён провайдером") + " · Стоимость не рассчитана")
    if answer["invalid_citations"]:
        st.error("Неизвестные ссылки: " + ", ".join(answer["invalid_citations"]))
    with st.expander("Запрос к API"):
        render_request(answer, key)
    with st.expander("Точный prompt и настройки"):
        st.json({"prompt_version": answer["prompt_version"], "settings": answer["generation_settings"], "index": answer["index_version"], "messages": answer["messages"]})
    if answer["mode"] == "no-rag":
        st.caption("Источники: N/A — документы в этот запрос не передавались")
        return
    st.caption(f"Найдено (Retrieved): {len(answer['retrieved_chunks'])} · включено (Included): {len(answer['context_items'])} · процитировано (Cited): {len(answer['citations'])}")
    if answer["index_version"]:
        st.caption(f"Использован индекс: {answer['index_version']['strategy']} · {answer['index_version']['build_id']}")
    st.caption("Валидная ссылка не доказывает, что текст подтверждает утверждение. Similarity — не уверенность в ответе.")
    for item in answer["context_items"]:
        if item["citation_id"] not in answer["citations"]:
            continue
        if st.button(f"[{item['citation_id']}] {item['source']} · открыть источник", key=key + item["citation_id"]):
            st.session_state[key + "_source"] = item["citation_id"]
    selected = st.session_state.get(key + "_source")
    item = next((i for i in answer["context_items"] if i["citation_id"] == selected), None)
    if item:
        snapshot = storage.snapshot(run["metadata"]["build_id"])
        doc = Document(**snapshot["documents"][item["source"]])
        st.write(item["title"])
        st.caption(f"Раздел: {item['section']} · строки {item['line_start']}–{item['line_end']} · страницы {item['page_start']}–{item['page_end']} · версия {run['metadata']['build_id'][:8]}")
        with st.container(height=350):
            st.markdown(source_html(doc, ContextItem(**item)), unsafe_allow_html=True)
    with st.expander("Найденные чанки (Retrieved)"):
        for chunk in answer["retrieved_chunks"]:
            st.caption(f"#{chunk['rank']} · {chunk['source']} · similarity {chunk['retrieval_score']:.4f}")
            st.text(chunk["text"])
    with st.expander("Фактический контекст (Included)"):
        st.caption(f"Бюджет использован: {answer['context_token_count']} · {answer['token_count_method']}")
        for context in answer["context_items"]:
            st.caption(f"[{context['citation_id']}] {context['source']} · [{context['start_char']}, {context['end_char']}) · обрезан: {context['truncated']}")
            st.text(context["included_text"])

def downloads(storage, run, key):
    snapshot = storage.snapshot(run["metadata"]["build_id"]) if run["metadata"].get("build_id") else None
    st.caption("Экспорт включает сохранённые ответы, ошибки и ручные оценки; повторных вызовов LLM нет.")
    for col, (ext, text) in zip(st.columns(3), report_formats(run, snapshot).items()):
        col.download_button(ext.upper(), text, f"{run['run_id']}.{ext}", key=key + ext)
