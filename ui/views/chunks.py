import numpy as np
import streamlit as st
from document_indexer.models import to_dict
from .shared import LABELS, indexes, show_source, location

def render(root):
    st.title("Где проходят границы")
    st.write("Выберите фрагмент и сравните покрытие одного и того же участка двумя стратегиями.")
    available = indexes(root)
    if not available:
        st.info("Постройте индекс, чтобы посмотреть чанки")
        return
    sources = sorted({source for idx in available.values() for source in idx.documents})
    desired = st.session_state.pop("focus_source", None)
    if desired in sources:
        st.session_state["chunk_source"] = desired
    source = st.selectbox("Документ", sources, key="chunk_source")
    focus = st.session_state.pop("focus_chunk", None)
    selected = {}
    for col, (strategy, index) in zip(st.columns(2), available.items()):
        with col:
            st.subheader(LABELS[strategy])
            chunks = [c for c in index.chunks if c.source == source]
            if not chunks:
                st.info("Нет чанков этого документа")
                continue
            ids = [c.chunk_id for c in chunks]
            key = "selected_" + strategy
            if focus in ids:
                st.session_state[key] = focus
            if st.session_state.get(key) not in ids:
                st.session_state[key] = ids[0]
            lookup = {c.chunk_id: c for c in chunks}
            selected_id = st.selectbox("Чанк", ids, format_func=lambda cid, lookup=lookup: f"{lookup[cid].ordinal}. {lookup[cid].token_count} токенов · {location(lookup[cid])}", key=key)
            chunk = lookup[selected_id]
            selected[strategy] = chunk
            st.caption(" · ".join(" / ".join(s) for s in chunk.sections))
            st.text(chunk.text)
            st.caption(f"Диапазон [{chunk.start_char}, {chunk.end_char}) · всего чанков: {len(chunks)}")
            neighbors = [c for c in chunks if abs(c.ordinal - chunk.ordinal) == 1]
            with st.expander("Соседние чанки"):
                for neighbor in neighbors:
                    st.caption(f"#{neighbor.ordinal} · {location(neighbor)}")
                    st.text(neighbor.text)
            with st.expander("Метаданные"):
                st.json(to_dict(chunk))
                if st.checkbox("Показать вектор", key="vector_" + strategy):
                    vector = index.matrix[next(i for i, c in enumerate(index.chunks) if c.chunk_id == selected_id)]
                    st.json({"dimension": len(vector), "dtype": index.config["dtype"], "norm": float(np.linalg.norm(vector)), "first_values": vector[:8].tolist()})
            show_source(index, chunk)
    if len(selected) == 2:
        st.subheader("Пересечение диапазонов")
        reference = st.radio("Диапазон выбранного чанка", list(selected), format_func=lambda k: LABELS[k], horizontal=True)
        other = "structural" if reference == "fixed" else "fixed"
        if available[reference].documents[source].content_hash != available[other].documents[source].content_hash:
            st.warning("Версии документа различаются: пересечение диапазонов не сопоставимо")
        else:
            chunk = selected[reference]
            matches = [c for c in available[other].chunks if c.source == source and c.start_char < chunk.end_char and c.end_char > chunk.start_char]
            st.dataframe([{"Чанк другой стратегии": c.ordinal, "Начало": c.start_char, "Конец": c.end_char, "Токенов": c.token_count} for c in matches], hide_index=True, width="stretch")
