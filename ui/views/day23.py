from dataclasses import replace
from pathlib import Path
import json
import streamlit as st
from document_indexer.post_retrieval import MODES, PipelineSettings
from document_indexer.llm_client import load_configuration, ChatCompletionsClient
from document_indexer.storage import load_index
from document_indexer.experiment_storage import ExperimentStorage
from document_indexer.rag import RAGService, check_index
from document_indexer.jobs import MANAGER
from document_indexer.day23_evaluation import (prepare_calibration, calibrate_filter, label_pair, evaluate_day23,
    export_day23_report, report_formats, summarize_day23, evidence_checks, validate_calibration)
from document_indexer.models import Document
from document_indexer.rag_models import ContextItem
from .shared import source_html
from .rag_shared import poll, run_picker, render_answer

LABELS = {"baseline": "Baseline", "rewrite": "Rewrite", "filter": "Filter", "rewrite_filter": "Rewrite + Filter", "reranker": "Реранкер"}

def setup(root, storage, mode):
    config, defaults = load_configuration(root)
    paths = [p for p in (root / "output/structural.sqlite", root / "output/fixed.sqlite") if p.exists()]
    if not paths:
        st.error("Нет готового индекса")
        return config, defaults, None, PipelineSettings(mode=mode), False
    path = st.selectbox("Сохранённый индекс", paths, format_func=lambda p: p.stem, key="d23_index")
    index = load_index(path)
    check_index(index)
    st.caption(f"{len(index.chunks)} чанков · build {index.config['build_id']} · {index.config['model_id']}")
    with st.expander("Настройки поиска и генерации", expanded=True):
        cols = st.columns(3)
        rk = cols[0].number_input("retrieval_k", min_value=1, value=15, step=1)
        fk = cols[1].number_input("final_k", min_value=1, value=5, step=1)
        cols[2].selectbox("Метод фильтрации", ["similarity_threshold"])
        st.caption("Фильтр удаляет чанки, сохраняя порядок. Cross-encoder доступен на основном экране в режиме «Реранкер». Score не является вероятностью правильности.")
        cals = storage.list_artifacts("calibration")
        lookup = {c["id"]: c for c in cals}
        selected = st.selectbox("Калибровка", [None, *lookup], format_func=lambda c: "Ручной эксперимент — порог не откалиброван" if c is None else c[:8])
        threshold_text = st.text_input("Cosine threshold (обязателен для Filter)", value="", disabled=selected is not None,
                                       help="Введите число от −1 до 1. До калибровки значение экспериментальное.")
        threshold, valid = None, True
        if selected:
            threshold = lookup[selected]["threshold"]
        elif threshold_text.strip():
            try:
                threshold = float(threshold_text)
            except ValueError:
                st.error("Порог должен быть числом")
                valid = False
        temperature = st.number_input("Температура", 0., 2., defaults.temperature, .1)
        output = st.number_input("Максимум токенов ответа", 1, 32768, defaults.max_output_tokens)
        context = st.number_input("Бюджет контекста", 1, 100000, defaults.max_context_tokens)
        settings = replace(defaults, temperature=temperature, max_output_tokens=int(output), max_context_tokens=int(context))
        pipeline = PipelineSettings(mode=mode, retrieval_k=int(rk), final_k=int(fk), similarity_threshold=threshold, calibration_id=selected)
        try:
            pipeline.validate()
            settings.validate()
            if selected:
                validate_calibration(lookup[selected], index, pipeline, ChatCompletionsClient(config), settings)
                st.success(f"Порог откалиброван: {threshold:.6f}")
            else:
                st.warning("Порог не откалиброван")
        except ValueError as exc:
            st.warning(str(exc))
            valid = False
    st.caption(f"Провайдер {config.provider} · модель {config.model or 'не настроена'} · один prompt генерации для четырёх режимов")
    if config.missing():
        st.info("Для новых запросов настройте .env. Просмотр сохранённых результатов доступен.")
    return config, settings, index, pipeline, valid and not config.missing()

def render_candidate(answer, storage, run):
    rows = answer.get("candidates", [])
    if not rows:
        return
    table = [{"Источник": c["source"], "Retrieval rank": c["retrieval_rank"], "Retrieval score": c["retrieval_score"],
        "Решение": "отсеян" if not c["passed_filter"] else "выбран" if c["selected"] else "вне final_k",
        "Причина исключения": c["rejection_reason"], "Включён в контекст": c["included_in_context"], "Цитируется": c["cited"]} for c in rows]
    if any(c.get("reranker_score") is not None for c in rows):
        for row, candidate in zip(table, rows):
            row.update({"Reranker rank": candidate.get("reranker_rank"), "Reranker score (raw logit)": candidate.get("reranker_score"),
                        "Обрезан для реранкера": candidate.get("reranker_truncated", False),
                        "Токенов до / после": f"{candidate.get('reranker_input_tokens')} / {candidate.get('reranker_used_tokens')}"})
    event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="candidates_" + answer["request_id"])
    selected = event.selection.rows
    # A selectbox is also keyboard-friendly and supported by Streamlit's test runner.
    number = st.selectbox("Открыть чанк и источник", range(len(rows)), index=selected[0] if selected else 0,
                          format_func=lambda i: f"#{rows[i]['retrieval_rank']} · {rows[i]['source']}", key="chunk_" + answer["request_id"])
    c = rows[selected[0] if selected else number]
    with st.expander("Текст выбранного чанка и источник", expanded=bool(selected)):
        st.text(c["text"])
        snapshot = storage.snapshot(run["metadata"]["build_id"])
        doc = Document(**snapshot["documents"][c["source"]])
        item = ContextItem("", c["chunk_id"], c["text"], c["source"], c["title"], c["section"], c["line_start"], c["line_end"],
                           c["page_start"], c["page_end"], c["start_char"], c["end_char"], c["document_hash"], 0, "", False, c.get("source_url"))
        st.markdown(source_html(doc, item), unsafe_allow_html=True)

def show_answer(answer, storage, run):
    st.subheader(LABELS[answer["mode"]])
    left, right = st.columns(2)
    left.write("Исходный: " + answer["original_query"])
    right.write("Поисковый: " + answer["rewritten_query"])
    rewrite = answer.get("rewrite", {})
    if rewrite.get("status") == "fallback":
        st.warning("Fallback rewrite: " + str(rewrite.get("fallback_reason")))
    if answer["status"] == "no_relevant_context":
        st.info("После фильтрации подходящие фрагменты не найдены")
    elif answer["status"] == "no_results":
        st.info("Поиск не вернул кандидатов")
    elif answer["status"] == "empty_context":
        st.info("Кандидаты найдены, но контекст не сформирован: проверьте бюджет")
    counts = answer.get("counts", {})
    st.caption("Найдено {retrieved} → прошли фильтр {passed_filter} → выбрано {selected} → включено в контекст {included} → процитировано {cited}".format(**{k: counts.get(k, 0) for k in ("retrieved", "passed_filter", "selected", "included", "cited")}))
    st.caption("При выключенном фильтре все найденные кандидаты считаются прошедшими.")
    st.json({"Время этапов (мс)": answer["stage_latency_ms"], "Rewrite usage": rewrite.get("usage"),
             "Generation usage": answer["usage"], "generation_skipped": answer["generation_skipped"],
             "rewrite_reused": rewrite.get("reused"), "rewrite_original_latency_ms": rewrite.get("latency_ms")}, expanded=False)
    render_answer(answer, storage, run, "day23_" + answer["request_id"])
    render_candidate(answer, storage, run)

def review_form(answer, question, storage):
    if answer["status"] == "failed":
        return
    old = answer.get("review", {})
    with st.form("d23_review_" + answer["request_id"]):
        st.caption("Ручная оценка после чтения ответа и контекста; это не автоматический LLM-судья.")
        correctness = st.selectbox("Корректность 0–2", [0, 1, 2], index=old.get("correctness", 0))
        facts = {f["id"]: st.checkbox(f["description"], old.get("facts", {}).get(f["id"], False), key=answer["request_id"] + "_" + f["id"]) for f in question["expected_facts"]}
        contradictions = st.number_input("Противоречий", min_value=0, value=old.get("contradictions", 0))
        unsupported = st.number_input("Неподтверждённых утверждений", min_value=0, value=old.get("unsupported_claims", 0))
        def boolean(label, field, enabled=True):
            return st.selectbox(label, [None, True, False], index=[None, True, False].index(old.get(field)), format_func=lambda v: {None: "Не оценено", True: "Да", False: "Нет"}[v]) if enabled else None
        support = boolean("Цитирования подтверждены?", "citation_support", bool(question["expected_sources"]))
        insufficient = boolean("Недостаточность распознана?", "recognized_insufficiency", question["category"] == "insufficient")
        premise = boolean("Ложная предпосылка отвергнута?", "rejected_false_premise", question["category"] == "false_premise")
        irrelevant = boolean("Остался нерелевантный контекст?", "irrelevant_context", not question["answerable_from_corpus"])
        meaning = effect = None
        if answer["mode"] in {"rewrite", "rewrite_filter"} and answer["rewrite"].get("status") != "fallback":
            meaning = st.selectbox("Смысл rewrite", [None, "preserved", "changed"], index=[None, "preserved", "changed"].index(old.get("rewrite_meaning")))
            effect = st.selectbox("Поиск после rewrite", [None, "better", "same", "worse"], index=[None, "better", "same", "worse"].index(old.get("rewrite_search")))
        comment = st.text_area("Обоснование по фактическим данным", value=old.get("comment", ""))
        if st.form_submit_button("Сохранить оценку"):
            try:
                storage.save_review(answer["request_id"], dict(correctness=correctness, facts=facts, contradictions=int(contradictions),
                    unsupported_claims=int(unsupported), citation_support=support, recognized_insufficiency=insufficient,
                    rejected_false_premise=premise, irrelevant_context=irrelevant, rewrite_meaning=meaning, rewrite_search=effect, comment=comment))
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

def render_calibration(root, store, config, settings, index, pipeline, ready):
    st.subheader("Калибровка cosine-порога")
    st.caption("8 дополнительных вопросов × 2 поисковые формулировки. Подготовка: до 8 rewrite-вызовов, без генерации ответов. Частично релевантный = отрицательный для F1.")
    if st.button("Подготовить calibration-пары", disabled=not ready or MANAGER.active("day23_calibration")):
        dataset = json.loads((root / "data/day23_calibration_questions.json").read_text(encoding="utf-8"))
        test = json.loads((root / "data/rag_questions.json").read_text(encoding="utf-8"))
        def action(progress):
            return prepare_calibration(RAGService(ChatCompletionsClient(config), store), index, settings,
                replace(pipeline, mode="baseline", calibration_id=None), dataset, test, progress)
        st.session_state["day23_calibration_job"] = MANAGER.submit("day23_calibration", action)
    poll("day23_calibration")
    artifacts = store.list_artifacts("calibration_pairs")
    if not artifacts:
        st.info("Сначала подготовьте пары. Основной эксперимент ожидает ручной калибровки.")
        return
    aid = st.selectbox("Набор пар", [a["id"] for a in artifacts])
    data = store.get_artifact(aid)
    if data.get("status") != "awaiting_human_labels":
        st.info("Подготовка набора не завершена: " + data.get("status", "unknown"))
        if data.get("error"):
            st.error(str(data["error"]))
        return
    pairs = data["pairs"]
    st.write(f"Размечено {sum(p['reviewed_by'] == 'human' for p in pairs)} / {len(pairs)} пар")
    if pairs:
        pid = st.selectbox("Пара для разметки", range(len(pairs)), format_func=lambda i: f"{'✓' if pairs[i]['reviewed_by'] else '○'} {pairs[i]['question_id']} · {pairs[i]['source']} · {pairs[i]['chunk_id'][:8]}")
        pair = pairs[pid]
        st.write(pair["question"])
        st.text(pair["text"])
        st.json(pair["observations"], expanded=False)
        with st.form("label_pair_" + pair["pair_id"]):
            labels = [None, "relevant", "partial", "irrelevant"]
            label = st.selectbox("Ручная метка", labels, index=labels.index(pair["label"]), format_func=lambda x: {None: "Выберите метку", "relevant": "Релевантный", "partial": "Частично релевантный", "irrelevant": "Нерелевантный"}[x])
            comment = st.text_input("Обоснование метки", value=pair["comment"])
            confirmed = st.checkbox("Я прочитал(а) вопрос и чанк и проверил(а) метку")
            if st.form_submit_button("Сохранить метку"):
                if label is None or not confirmed:
                    st.error("Выберите и подтвердите метку")
                else:
                    label_pair(store, aid, pair["pair_id"], label, comment)
                    st.rerun()
    if st.button("Рассчитать и зафиксировать порог"):
        try:
            result = calibrate_filter(store, aid)
            st.success(f"Порог {result['threshold']:.6f}. Выберите калибровку {result['calibration_id']} в настройках.")
            st.dataframe(result["table"], hide_index=True)
        except ValueError as exc:
            st.error(str(exc))
    for c in store.list_artifacts("calibration"):
        with st.expander("Калибровка " + c["id"][:8]):
            st.write(c["rule"])
            st.dataframe(c["table"], hide_index=True)
            st.download_button("Скачать калибровку JSON", json.dumps(c, ensure_ascii=False, indent=2), c["id"] + ".json", key="cal_export_" + c["id"])

def render(root):
    if not st.checkbox("Расширенный режим: rewrite и калибровка", key="day23_advanced"):
        from .day23_simple import render as render_simple
        render_simple(root)
        return
    render_advanced(root)

def render_advanced(root):
    st.title("Реранкинг · дополнительные режимы")
    store = ExperimentStorage(root / "output/day23_experiments")
    section = st.radio("Действие", ["Запрос", "Калибровка", "Эксперимент"], horizontal=True)
    mode = st.radio("Режим", list(MODES), format_func=LABELS.get, horizontal=True) if section == "Запрос" else "baseline"
    config, settings, index, pipeline, ready = setup(root, store, mode)
    if section == "Калибровка":
        render_calibration(root, store, config, settings, index, pipeline, ready)
        return
    if section == "Запрос":
        question = st.text_area("Исходный вопрос", key="d23_question")
        compare_baseline = st.checkbox("Сравнить с Baseline (до двух генераций)", disabled=mode == "baseline")
        if st.button("Получить ответ", disabled=not ready or MANAGER.active("day23")):
            if not question.strip():
                st.error("Введите вопрос")
            else:
                def action(progress):
                    service = RAGService(ChatCompletionsClient(config), store)
                    if compare_baseline and mode != "baseline":
                        from document_indexer.day23_pipeline import compare
                        run = compare(service, question, index, settings, pipeline, progress)
                    else:
                        run = service.run_query(question, mode, settings, index, progress, pipeline)
                    return export_day23_report(store, run["run_id"])
                st.session_state["day23_job"] = MANAGER.submit("day23", action)
    else:
        st.info("10 неизменённых вопросов × 4 режима: до 40 генераций + 10 rewrite. HTTP-повторы учитываются отдельно. Запуск только кнопкой.")
        if not pipeline.calibration_id:
            st.warning("Основной эксперимент заблокирован: сначала завершите ручную калибровку")
        if st.button("Запустить 4 режима на 10 вопросах", disabled=not ready or not pipeline.calibration_id or MANAGER.active("day23")):
            test = json.loads((root / "data/rag_questions.json").read_text(encoding="utf-8"))
            def action(progress):
                run = evaluate_day23(RAGService(ChatCompletionsClient(config), store), index, test, settings, pipeline, progress)
                return export_day23_report(store, run["run_id"])
            st.session_state["day23_job"] = MANAGER.submit("day23", action)
    poll("day23")
    run = run_picker(store, "day23")
    if not run:
        return
    stats = summarize_day23(run)
    st.dataframe([{"Режим": LABELS[m], **{k: v for k, v in stats["modes"][m].items() if not isinstance(v, dict)}} for m in stats["modes"]], hide_index=True)
    st.caption(f"Вызовы: generation {stats['generation_calls']}, rewrite {stats['rewrite_calls']}; HTTP-попытки: {stats['generation_attempts'] + stats['rewrite_attempts']}")
    rows = run["answers"]
    if rows:
        qid = st.selectbox("Вопрос для просмотра", list(dict.fromkeys(a["question_id"] for a in rows)))
        rows = [a for a in rows if a["question_id"] == qid]
        improved_modes = [a["mode"] for a in rows if a["mode"] != "baseline"]
        chosen = st.selectbox("Улучшенный режим для сравнения", improved_modes, format_func=LABELS.get) if improved_modes else None
        visible = [a for a in rows if a["mode"] in {"baseline", chosen}]
        questions = {q["id"]: q for q in run["metadata"].get("questions", [])}
        for col, a in zip(st.columns(len(visible)), visible):
            with col:
                show_answer(a, store, run)
                if qid in questions:
                    st.json({"Ожидания (не отправлялись LLM)": questions[qid], "Проверка диапазонов": evidence_checks(a, questions[qid])}, expanded=False)
                    review_form(a, questions[qid], store)
    snapshot = store.snapshot(run["metadata"]["build_id"])
    for ext, content in report_formats(run, snapshot).items():
        st.download_button("Скачать " + ext.upper(), content, run["run_id"] + "." + ext, key="d23_download_" + ext)
