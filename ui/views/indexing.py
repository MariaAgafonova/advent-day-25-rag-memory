import streamlit as st
from document_indexer.jobs import MANAGER
from document_indexer.indexing import build_index, is_stale
from document_indexer.loaders import load_corpus
from .shared import LABELS, indexes

@st.fragment(run_every="1s")
def status_panel(root):
    for col, strategy in zip(st.columns(2), LABELS):
        with col:
            st.subheader(LABELS[strategy])
            job_id = st.session_state.get("job_" + strategy)
            job = MANAGER.snapshot(job_id) if job_id else None
            if job:
                labels = {"queued": "В очереди", "running": "Выполняется", "completed": "Индекс построен", "failed": "Ошибка сборки"}
                st.write(labels[job["status"]])
                progress = job["progress"]
                if progress:
                    st.caption(progress["stage"])
                    if progress["total"] is not None:
                        st.write(f"Обработано {progress['done']} из {progress['total']}")
                if job["error"]:
                    st.error(job["error"])
                if job["status"] == "completed":
                    st.success(f"{job['result']['config']['chunk_count']} чанков · {job['result']['total_seconds']:.1f} с")
            else:
                st.caption("Готов к сборке" if (root / "output" / (strategy + ".sqlite")).exists() else "Ещё не построен")

def render(root):
    st.title("Два способа разделить текст")
    st.write("Одна модель и один корпус. Различается только способ определения границ.")
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Фиксированные окна")
        st.write("Последовательные токенные окна с overlap; окно может пересекать разделы.")
    with cols[1]:
        st.subheader("По структуре")
        st.write("Разделы и абзацы. Overlap появляется только при разбиении слишком большого блока.")
    a, b = st.columns(2)
    size = a.number_input("Размер чанка, токенов", min_value=2, max_value=500, value=300, key="size")
    overlap = b.number_input("Overlap, токенов", min_value=0, max_value=499, value=50, key="overlap")
    st.caption("Модель: intfloat/multilingual-e5-small · устройство: CPU · batch size: 16")
    docs, errors = load_corpus(root / "data/raw")
    for key, index in indexes(root).items():
        st.caption(f"{LABELS[key]} · версия {index.config['created_at'][:19]} · {index.config['chunk_count']} чанков")
        if errors or is_stale(index, docs, size, overlap):
            st.warning(f"{LABELS[key]}: Индекс требует обновления")
    active = any(MANAGER.active("build_" + s) for s in LABELS)
    if st.button("Построить оба индекса", type="primary", disabled=active or overlap >= size):
        for strategy in LABELS:
            def action(progress, strategy=strategy):
                return build_index(root / "data/raw", strategy, root / "output" / (strategy + ".sqlite"), int(size), int(overlap), progress=progress)
            st.session_state["job_" + strategy] = MANAGER.submit("build_" + strategy, action)
    if overlap >= size:
        st.error("Overlap должен быть меньше размера чанка")
    status_panel(root)
