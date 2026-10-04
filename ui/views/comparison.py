import json
import streamlit as st
from document_indexer.evaluate import evaluate, validate_questions, attach_file_statistics
from document_indexer.storage import compatibility
from document_indexer.jobs import MANAGER
from document_indexer.reporting import report_formats, export_report
from .shared import LABELS, indexes, result_cards

@st.fragment(run_every="1s")
def poll_job():
    job_id = st.session_state.get("evaluation_job")
    job = MANAGER.snapshot(job_id) if job_id else None
    if job and job["status"] in {"queued", "running"}:
        progress = job["progress"]
        st.info(f"Оценка выполняется: {progress.get('done', 0)} из {progress.get('total') or '…'} вопросов")
    elif job and job["status"] == "failed":
        st.error(job["error"])
    elif job and job["status"] == "completed" and st.session_state.get("evaluation_consumed") != job_id:
        st.session_state["report"] = job["result"]
        st.session_state["evaluation_consumed"] = job_id
        st.rerun(scope="app")

def render(root):
    st.title("Сравнение на размеченных вопросах")
    st.write("Ответ засчитывается, только если один чанк целиком содержит достаточный фрагмент эталона.")
    available = indexes(root)
    if len(available) < 2:
        st.info("Нет измерений. Сначала постройте оба индекса.")
        return
    reasons = compatibility(available["fixed"], available["structural"])
    if reasons:
        st.error("Сравнение заблокировано: " + "; ".join(reasons))
        return
    uploaded = st.file_uploader("Файл evaluation.json (или используйте разметку проекта)", type=["json"])
    local = root / "data/evaluation.json"
    data = json.loads(uploaded.getvalue().decode("utf-8-sig")) if uploaded else json.loads(local.read_text(encoding="utf-8")) if local.exists() else None
    valid = False
    if data:
        questions = validate_questions(data, available["fixed"].documents)
        validate_questions(data, available["structural"].documents)
        st.caption(f"Проверены хеши и диапазоны: {len(questions)} вопросов")
        valid = True
    if st.button("Запустить оценку", type="primary", disabled=not valid or MANAGER.active("evaluation")):
        def action(progress):
            result = evaluate(available["fixed"], available["structural"], data, progress=progress)
            attach_file_statistics(result, {s: root / "output" / (s + ".sqlite") for s in LABELS})
            export_report(result, root / "output/comparison")
            return result
        st.session_state["evaluation_job"] = MANAGER.submit("evaluation", action)
    poll_job()
    report = st.session_state.get("report")
    saved = root / "output/comparison.json"
    if report is None and saved.exists():
        report = json.loads(saved.read_text(encoding="utf-8"))
    if report is None:
        st.info("Нет измерений")
        return
    if any(report["strategies"][s]["config"]["build_id"] != available[s].config["build_id"] for s in LABELS):
        st.warning("Измерения относятся к старой сборке. Запустите оценку заново.")
        return
    st.caption(f"Вопросов: {report['question_count']} · загрузка модели: {report['model_load_seconds']:.2f} с · прогрев и 3 повторения")
    for col, strategy in zip(st.columns(2), LABELS):
        with col:
            r = report["strategies"][strategy]
            st.subheader(LABELS[strategy])
            for metric, value in r["metrics"].items():
                st.metric(metric, f"{value:.3f}")
            st.caption(f"{r['chunk_count']} чанков · база {r.get('database_bytes', 0)/1024:.0f} КБ")
            st.write(f"Поиск median / p95: {r['timing']['median_seconds']*1000:.1f} / {r['timing']['p95_seconds']*1000:.1f} мс")
            seconds = r.get("indexing_seconds")
            st.write(f"Индексация: {seconds:.1f} с" if seconds is not None else "Время индексации: нет измерений")
            st.caption("Распределение размеров чанков, токенов")
            st.bar_chart({"Токены": r["chunk_sizes"]}, height=150)
            st.json(r["size_distribution"], expanded=False)
    first = report["strategies"]["fixed"]["questions"]
    second = report["strategies"]["structural"]["questions"]
    st.dataframe([{"Вопрос": a["question"], "Fixed: первая релевантная позиция": a["first_relevant_rank"],
                   "Structural: первая релевантная позиция": b["first_relevant_rank"]} for a, b in zip(first, second)], hide_index=True, width="stretch")
    qindex = st.selectbox("Разобрать вопрос", range(len(first)), format_func=lambda i: first[i]["question"])
    with st.expander("Ожидаемый ответ", expanded=True):
        for answer in first[qindex]["answers"]:
            st.caption(f"{answer['source']} · [{answer['start_char']}, {answer['end_char']})")
            st.text(answer["answer_excerpt"])
    for col, strategy in zip(st.columns(2), LABELS):
        with col:
            q = report["strategies"][strategy]["questions"][qindex]
            st.subheader(LABELS[strategy])
            st.write(f"Первый релевантный: {q['first_relevant_rank'] or 'нет в top-5'}")
            for result in q["results"]:
                st.caption(f"#{result['rank']}: " + ("засчитан — полностью содержит эталон" if result["matched_answers"] else "не засчитан — не содержит целый диапазон ответа"))
            with st.expander("Найденные фрагменты"):
                result_cards(available[strategy], q["results"], f"eval_{qindex}_{strategy}")
    st.subheader("Экспорт отчёта")
    for col, (extension, content) in zip(st.columns(3), report_formats(report).items()):
        col.download_button(extension.upper(), content, "comparison." + extension, mime={"json": "application/json", "csv": "text/csv", "md": "text/markdown"}[extension])
