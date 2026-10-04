from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui"))
from views import documents, indexing, chunks, search, comparison, rag_query, rag_evaluation, day23, day24, chat

st.set_page_config(page_title="День 25 · Мини-чат", page_icon="📚", layout="wide")
st.markdown("""<style>
.block-container{max-width:1480px;padding-top:2rem;padding-bottom:3rem}
h1{letter-spacing:-.045em!important} h2,h3{letter-spacing:-.025em!important}
div[data-testid="stMetric"]{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:16px}
pre{white-space:pre-wrap!important;overflow-wrap:anywhere!important}
div[data-testid="stMarkdownContainer"]{overflow-wrap:anywhere}
</style>""", unsafe_allow_html=True)
if "pending_page" in st.session_state:
    st.session_state["page"] = st.session_state.pop("pending_page")
if st.session_state.get("page") == "День 23":
    st.session_state["page"] = "Реранкинг"
with st.sidebar:
    st.caption("AI ADVENT / ДЕНЬ 25")
    st.title("Document\nIndexer")
    st.caption("Локальный поиск по смыслу")
    st.divider()
    page = st.radio("Рабочее пространство", ["Документы", "Индексация", "Чанки", "Поиск", "Сравнение", "RAG-запрос", "Оценка RAG", "Реранкинг", "Проверяемые цитаты", "Мини-чат"], index=None if "page" in st.session_state else 9, key="page")
    st.divider()
    st.caption("multilingual-e5-small · CPU\n\nДокументы и индексы хранятся локально.")
try:
    {"Документы": documents, "Индексация": indexing, "Чанки": chunks,
     "Поиск": search, "Сравнение": comparison, "RAG-запрос": rag_query, "Оценка RAG": rag_evaluation, "Реранкинг": day23, "Проверяемые цитаты": day24, "Мини-чат": chat}[page].render(ROOT)
except Exception as exc:
    st.error(str(exc))
    st.caption("Исправьте проблему и повторите действие. Предыдущие сохранённые индексы доступны.")
