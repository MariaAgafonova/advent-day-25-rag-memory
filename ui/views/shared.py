import html
import streamlit as st
from document_indexer.storage import load_index
from document_indexer.sections import extract_sections
from document_indexer.models import to_dict

LABELS = {"fixed": "Фиксированные окна", "structural": "По структуре"}

def indexes(root):
    result = {}
    for key in LABELS:
        path = root / "output" / (key + ".sqlite")
        if path.exists():
            try:
                result[key] = load_index(path)
            except Exception as exc:
                st.error(f"{LABELS[key]}: {exc}")
    return result

def location(chunk):
    if isinstance(chunk, dict):
        return f"Страницы {chunk['page_start']}–{chunk['page_end']}" if chunk["page_start"] else f"Строки {chunk['line_start']}–{chunk['line_end']}"
    return location(to_dict(chunk))

def source_html(doc, selected, siblings=()):
    a, b = selected.start_char, selected.end_char
    overlaps = [(max(a, c.start_char), min(b, c.end_char)) for c in siblings
                if c.chunk_id != selected.chunk_id and c.start_char < b and c.end_char > a]
    sections = extract_sections(doc)
    boundaries = {s.start_char: " / ".join(s.heading_path) for s in sections}
    cuts = sorted({0, len(doc.text), a, b, *boundaries, *(x for pair in overlaps for x in pair)})
    parts = []
    for start, end in zip(cuts, cuts[1:]):
        if start in boundaries:
            parts.append('<div style="border-top:1px dashed #94a3b8;color:#64748b;font:12px sans-serif;margin:12px 0 6px">' + html.escape(boundaries[start]) + '</div>')
        text = html.escape(doc.text[start:end])
        if a <= start < b:
            overlap = any(x <= start < y for x, y in overlaps)
            style = "background:#bfdbfe;border-bottom:2px dashed #2563eb" if overlap else "background:#dbeafe"
            text = f'<mark style="{style};color:#172033">{text}</mark>'
        parts.append(text)
    return '<div style="white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.65 monospace">' + "".join(parts) + '</div>'

def show_source(index, chunk):
    doc = index.documents[chunk.source]
    st.caption(f"Сохранённая версия · {index.config['created_at'][:19]} · {doc.content_hash[:10]} · {location(chunk)}")
    st.caption("Голубой — выбранный чанк; штриховая линия — overlap; пунктир — граница раздела.")
    with st.container(height=460):
        st.markdown(source_html(doc, chunk, [c for c in index.chunks if c.source == chunk.source]), unsafe_allow_html=True)

def result_cards(index, results, prefix):
    if not results:
        st.info("Фрагменты не найдены")
    lookup = {c.chunk_id: c for c in index.chunks}
    for item in results:
        with st.container(border=True):
            st.caption(f"#{item['rank']} · similarity {item['score']:.4f} · {location(item)}")
            st.write(item["title"])
            st.caption(" · ".join(" / ".join(s) for s in item["sections"]))
            st.text(item["text"][:1100] + ("…" if len(item["text"]) > 1100 else ""))
            if st.button("Открыть источник", key=f"{prefix}_source_{item['chunk_id']}"):
                st.session_state["source_preview"] = (prefix, item["chunk_id"])
            if st.session_state.get("source_preview") == (prefix, item["chunk_id"]):
                show_source(index, lookup[item["chunk_id"]])
                if item.get("source_url"):
                    st.link_button("Оригинал на сайте", item["source_url"])
            if st.button("Посмотреть чанк", key=f"{prefix}_chunk_{item['chunk_id']}"):
                st.session_state["focus_chunk"] = item["chunk_id"]
                st.session_state["focus_source"] = item["source"]
                st.session_state["pending_page"] = "Чанки"
                st.rerun()
