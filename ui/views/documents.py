import streamlit as st
from document_indexer.loaders import load_corpus, corpus_stats, save_upload

def render(root):
    st.title("Ваш корпус знаний")
    st.write("Добавьте документы, проверьте извлечённый текст и подготовьте корпус к поиску.")
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    with st.expander("Добавить документы", expanded=True):
        with st.form("uploads", clear_on_submit=True):
            files = st.file_uploader("Markdown, TXT или текстовый PDF · до 20 МБ на файл", type=["md", "txt", "pdf"], accept_multiple_files=True)
            submitted = st.form_submit_button("Добавить в корпус", type="primary")
        if submitted:
            for file in files:
                try:
                    name = save_upload(raw, file.name, file.getvalue())
                    st.success(f"Добавлен {name}")
                except Exception as exc:
                    st.error(f"{file.name}: {exc}")
    docs, errors = load_corpus(raw)
    stats = corpus_stats(docs, errors)
    a, b, c = st.columns(3)
    a.metric("Файлов", stats["files"])
    b.metric("Слов до overlap", f"{stats['words']:,}".replace(",", " "))
    c.metric("Условных страниц без дубликатов", f"{stats['unique_equivalent_pages']:.1f}")
    if stats["volume_requirement_met"]:
        st.success("Требование по объёму выполнено: не менее 25 страниц / 10 000 слов.")
    else:
        st.info("Для задания нужно 25–30 содержательных PDF-страниц или около 10–12 тысяч слов. 400 слов = 1 условная страница.")
    for error in errors:
        st.error(f"{error['source']}: {error['error']}")
    if stats["duplicates"]:
        st.warning("Найдены дубликаты по хешу; они исключены из проверки достаточности объёма.")
    if docs:
        st.dataframe([{"Файл": d.source, "Формат": d.format.upper(), "Символов": len(d.text),
                       "Состояние": "; ".join(d.warnings) if d.warnings else "Текст извлечён"} for d in docs], hide_index=True, width="stretch")
        doc = st.selectbox("Посмотреть документ", docs, format_func=lambda d: d.source)
        for warning in doc.warnings:
            st.warning(warning)
        st.text_area("Извлечённый нормализованный текст", doc.text, height=360)
        if doc.source_url:
            st.link_button("Источник материала", doc.source_url)
    with st.expander("Подробная статистика"):
        st.json(stats)
        st.caption("Точное число токенов доступно после индексации или через inspect --tokens.")
