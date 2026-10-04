import json
import streamlit as st
from document_indexer.llm_client import ChatCompletionsClient
from document_indexer.rag import RAGService
from document_indexer.experiment_storage import ExperimentStorage
from document_indexer.jobs import MANAGER
from document_indexer.rag_evaluation import validate_questions, evaluate_rag, source_checks, summarize, export_run
from .rag_shared import setup, poll, run_picker, render_answer, downloads
from .api_requests import render_comparison

def review_form(answer, question, storage):
    if answer["status"] not in {"completed", "citation_error"}:
        st.info("Неуспешный вызов виден в отчёте; оценка качества недоступна")
        return
    existing = answer["review"]
    with st.form("review_" + answer["request_id"]):
        st.caption("Ручная рубрика. Выставляйте оценки после чтения ответа и фактического контекста.")
        correctness = st.selectbox("Корректность: 0 — неверно; 1 — частично; 2 — правильно", [0, 1, 2], index=existing.get("correctness", 0))
        facts = {f["id"]: st.checkbox(f["description"] + f" (вес {f['weight']})", value=existing.get("facts", {}).get(f["id"], False), key=answer["request_id"] + f["id"]) for f in question["expected_facts"]}
        contradictions = st.number_input("Существенных противоречий эталону", min_value=0, value=existing.get("contradictions", 0))
        unsupported = st.number_input("Утверждений, не подтверждённых переданным контекстом", min_value=0, value=existing.get("unsupported_claims") or 0) if answer["mode"] == "rag" else None
        def boolean_field(label, field, enabled):
            if not enabled:
                return None
            return st.selectbox(label, [None, True, False], index=[None, True, False].index(existing.get(field)), format_func=lambda value: {None: "Не оценено", True: "Да", False: "Нет"}[value])
        support = boolean_field("Цитаты действительно поддерживают утверждения?", "citation_support", answer["mode"] == "rag" and bool(question["expected_sources"]))
        insufficient = boolean_field("Нехватка информации распознана без выдуманного ответа?", "recognized_insufficiency", question["category"] == "insufficient")
        premise = boolean_field("Ложная предпосылка отвергнута?", "rejected_false_premise", question["category"] == "false_premise")
        comment = st.text_area("Краткое обоснование оценки", value=existing.get("comment", ""))
        if st.form_submit_button("Сохранить ручную оценку"):
            storage.save_review(answer["request_id"], {"correctness": correctness, "facts": facts, "contradictions": int(contradictions),
                "unsupported_claims": int(unsupported) if unsupported is not None else None, "citation_support": support,
                "recognized_insufficiency": insufficient, "rejected_false_premise": premise, "comment": comment})
            st.rerun()

def render(root):
    st.title("Оценка RAG")
    st.write("10 независимых вопросов. Автоматически проверяются диапазоны и ссылки; качество оценивается вручную.")
    config, settings, index, ready = setup(root, "rag_evaluation")
    path = root / "data/rag_questions.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if data is None:
        st.info("Нет data/rag_questions.json")
    elif index:
        questions = validate_questions(data, index)
        with st.expander("Все 10 вопросов и ожидания", expanded=True):
            st.dataframe([{"ID": q["id"], "Вопрос": q["question"], "Категория": q["category"], "Ожидание": q["expected_behavior"]} for q in questions], hide_index=True, width="stretch")
            st.caption("Эталоны проверены по хешам и диапазонам. В prompt они не передаются.")
    storage = ExperimentStorage(root / "output/rag_experiments")
    st.info("Запланировано 20 вызовов LLM: 10 вопросов × 2 режима. Временные ошибки могут вызвать один повтор запроса. Стоимость не рассчитана.")
    if st.button("Запустить сравнение: 20 вызовов", type="primary", disabled=not ready or data is None or MANAGER.active("rag_evaluation")):
        def action(progress):
            service = RAGService(ChatCompletionsClient(config), storage)
            run = evaluate_rag(service, index, data, settings, progress)
            return export_run(storage, run["run_id"])
        st.session_state["rag_evaluation_job"] = MANAGER.submit("rag_evaluation", action)
    poll("rag_evaluation")
    run = run_picker(storage, "rag_evaluation", "evaluation")
    if not run:
        return
    summary = summarize(run)
    st.write(f"Завершено: {summary['completed_calls']} · ошибок: {summary['failed_calls']} · ещё нет результатов: {summary['planned_calls'] - summary['recorded_calls']}")
    st.caption(f"Полностью проверено вопросов: {summary['fully_reviewed_questions']}; хотя бы один ответ проверен: {summary['questions_with_review']}")
    for col, mode in zip(st.columns(2), ("no-rag", "rag")):
        with col:
            m = summary["modes"][mode]
            st.subheader("Без RAG" if mode == "no-rag" else "С RAG")
            st.write("Нет ручных оценок" if m["correctness"] is None else f"Средняя корректность: {m['correctness']:.2f}; полнота: {m['completeness']}")
            st.caption(f"Проверено ответов: {m['reviewed_answers']} · источники: {'N/A' if mode == 'no-rag' else m['source_support']}")
    questions = run["metadata"]["questions"]
    rows = []
    for q in questions:
        row = {"Вопрос": q["question"]}
        for mode in ("no-rag", "rag"):
            answer = next((a for a in run["answers"] if a["question_id"] == q["id"] and a["mode"] == mode), None)
            review = answer["review"] if answer else {}
            row[mode + " качество"] = review.get("correctness")
            row[mode + " полнота"] = review.get("completeness")
            row[mode + " проверка"] = review.get("status", "unreviewed")
            row[mode + " проблемы"] = str(answer["error"] or {"противоречия": review.get("contradictions"), "без опоры": review.get("unsupported_claims"), "ссылки": answer["invalid_citations"]}) if answer else "Не выполнен"
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")
    qid = st.selectbox("Разобрать вопрос", [q["id"] for q in questions], key="review_question")
    question = next(q for q in questions if q["id"] == qid)
    st.write(question["question"])
    with st.expander("Ожидания, запретные утверждения и доказательства", expanded=True):
        st.json(question)
    render_comparison([a for a in run["answers"] if a["question_id"] == qid], "eval_compare_" + run["run_id"])
    for col, mode in zip(st.columns(2), ("no-rag", "rag")):
        with col:
            st.subheader("Без RAG" if mode == "no-rag" else "С RAG")
            answer = next((a for a in run["answers"] if a["question_id"] == qid and a["mode"] == mode), None)
            if answer:
                render_answer(answer, storage, run, "eval_" + answer["request_id"])
                checks = source_checks(answer, question)
                if checks is not None:
                    st.json({"Автоматическая проверка диапазонов (не смысловая)": checks}, expanded=False)
                review_form(answer, question, storage)
    downloads(storage, run, "eval_download_")
