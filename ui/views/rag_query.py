import streamlit as st
from document_indexer.llm_client import ChatCompletionsClient
from document_indexer.rag import RAGService
from document_indexer.experiment_storage import ExperimentStorage
from document_indexer.jobs import MANAGER
from document_indexer.rag_evaluation import export_run
from .rag_shared import setup, poll, run_picker, render_answer, downloads
from .api_requests import render_comparison

def render(root):
    version = st.radio("Версия RAG", ["День 22", "Реранкинг"], horizontal=True)
    if version == "Реранкинг":
        from .day23 import render as render_day23
        render_day23(root)
        return
    st.title("RAG-запрос")
    st.write("Сравните знания модели с ответом по выбранной версии ваших документов.")
    labels = {"compare": "Сравнить оба", "no-rag": "Без RAG", "rag": "С RAG"}
    mode = st.radio("Режим", list(labels), format_func=labels.get, horizontal=True, key="rag_mode")
    config, settings, index, ready = setup(root, "rag_query", mode != "no-rag")
    question = st.text_area("Вопрос", placeholder="Как выполнить фоновую задачу после перезагрузки устройства?", key="rag_question")
    storage = ExperimentStorage(root / "output/rag_experiments")
    st.caption("Выбранные фрагменты будут отправлены настроенному LLM-провайдеру только после нажатия кнопки.")
    if st.button("Сравнить ответы" if mode == "compare" else "Получить ответ", type="primary", disabled=not ready or MANAGER.active("rag_query")):
        if not question.strip():
            st.error("Введите непустой вопрос")
        else:
            def action(progress):
                service = RAGService(ChatCompletionsClient(config), storage)
                run = service.run_query(question, mode, settings, index, progress)
                return export_run(storage, run["run_id"])
            st.session_state["rag_query_job"] = MANAGER.submit("rag_query", action)
    poll("rag_query")
    run = run_picker(storage, "rag_query")
    if not run:
        return
    if run["answers"]:
        st.subheader(run["answers"][0]["question"])
    render_comparison(run["answers"], "query_compare_" + run["run_id"])
    cols = st.columns(2) if len(run["answers"]) == 2 else [st.container()]
    for col, answer in zip(cols, sorted(run["answers"], key=lambda a: a["mode"] != "no-rag")):
        with col:
            st.subheader("Без RAG" if answer["mode"] == "no-rag" else "С RAG")
            render_answer(answer, storage, run, "query_" + answer["request_id"])
    downloads(storage, run, "query_download_")
