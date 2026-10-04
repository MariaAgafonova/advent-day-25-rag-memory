import streamlit as st
from document_indexer.request_inspection import inspect_request, request_json, request_diff, parameter_comparison

def render_request(answer, key):
    inspection = inspect_request(answer)
    request = inspection["request"]
    if inspection["captured"]:
        st.caption(inspection["notice"])
        st.code(f"{request['method']} {request['url']}", language=None, wrap_lines=True)
        st.caption(f"Таймаут: {request['timeout_seconds']} с · попыток отправки: {answer.get('provider_attempts', 0)}")
    else:
        st.info(inspection["notice"])
    messages_tab, json_tab = st.tabs(["Сообщения модели", "JSON запроса"])
    with messages_tab:
        for number, message in enumerate(request["body"].get("messages", []), 1):
            st.caption(f"{number}. {message.get('role', 'unknown')}")
            st.code(str(message.get("content", "")), language=None, wrap_lines=True)
        if not request["body"].get("messages"):
            st.caption("Сообщения не сформированы")
    with json_tab:
        st.code(request_json(answer), language="json", wrap_lines=True)
        st.download_button("Скачать запрос JSON", request_json(answer), f"request-{answer['request_id']}.json",
                           mime="application/json", key=key + "_api_json")

def render_comparison(answers, key):
    if len(answers) != 2 or {a["mode"] for a in answers} != {"no-rag", "rag"}:
        return
    by_mode = {a["mode"]: a for a in answers}
    without_rag, with_rag = by_mode["no-rag"], by_mode["rag"]
    with st.expander("Сравнить запросы к API", expanded=True):
        st.write("Без RAG модель получает общую инструкцию и вопрос. С RAG к ним добавляются правила работы с источниками и фактически отобранные фрагменты.")
        st.caption("Просмотр сохранённых запросов не обращается к API и не меняет ответы.")
        st.dataframe(parameter_comparison(without_rag, with_rag), hide_index=True, width="stretch")
        st.caption(f"Вопросы {'совпадают' if without_rag['question'] == with_rag['question'] else 'различаются'} · в RAG-контекст включено фрагментов: {len(with_rag['context_items'])}")
        side_by_side, changes = st.tabs(["Два запроса рядом", "Различия"])
        with side_by_side:
            for col, answer in zip(st.columns(2), (without_rag, with_rag)):
                with col:
                    st.subheader("Без RAG" if answer["mode"] == "no-rag" else "С RAG")
                    render_request(answer, key + answer["request_id"])
        with changes:
            st.caption("Строки с − относятся к запросу без RAG; строки с + — к запросу с RAG.")
            st.code(request_diff(without_rag, with_rag) or "Сохранённые запросы одинаковы", language="diff", wrap_lines=True)
