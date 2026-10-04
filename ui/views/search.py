import streamlit as st
from document_indexer.search import search_both
from document_indexer.storage import compatibility
from .shared import LABELS, indexes, result_cards

def render(root):
    st.title("Один вопрос. Два индекса.")
    st.write("Сравните найденные фрагменты и откройте точное место в сохранённом документе.")
    available = indexes(root)
    if len(available) < 2:
        st.info("Постройте индекс, чтобы начать поиск")
        return
    reasons = compatibility(available["fixed"], available["structural"])
    if reasons:
        st.error("Индексы несовместимы: " + "; ".join(reasons))
        return
    with st.form("search"):
        query = st.text_input("Ваш вопрос", placeholder="Как выполнять фоновые задачи?", key="query")
        a, b = st.columns(2)
        top_k = a.slider("Число результатов", 1, 10, 5)
        source = b.selectbox("Искать в документе", ["Все документы"] + sorted(available["fixed"].documents))
        submitted = st.form_submit_button("Найти в обоих индексах", type="primary")
    builds = tuple(i.config["build_id"] for i in available.values())
    if submitted:
        with st.spinner("Ищем по смыслу…"):
            results = search_both(available["fixed"], available["structural"], query, top_k, None if source == "Все документы" else source)
        st.session_state["search_output"] = {"builds": builds, "query": query, "results": results}
    output = st.session_state.get("search_output")
    st.caption("Similarity — близость векторов, а не процент уверенности.")
    if output and output["builds"] == builds:
        st.write(f"Результаты для: {output['query']}")
        for col, strategy in zip(st.columns(2), LABELS):
            with col:
                st.subheader(LABELS[strategy])
                result = output["results"][strategy]
                st.caption(f"Поиск: {result['seconds'] * 1000:.1f} мс · версия {available[strategy].config['created_at'][:19]}")
                result_cards(available[strategy], result["results"], "search_" + strategy)
    elif output:
        st.info("Индекс обновлён. Повторите запрос для новой версии.")
