"""The default Day 23 flow: one question, three controls, two answers."""
import json
import streamlit as st
from document_indexer.llm_client import load_configuration, ChatCompletionsClient
from document_indexer.storage import load_index
from document_indexer.experiment_storage import ExperimentStorage
from document_indexer.rag import RAGService
from document_indexer.day23_pipeline import compare
from document_indexer.post_retrieval import PipelineSettings
from document_indexer.reranker import DEFAULT_MODEL, DEFAULT_REVISION
from document_indexer.day23_evaluation import export_day23_report, report_formats
from document_indexer.jobs import MANAGER
from .rag_shared import poll

CHOICES = {None: "Выберите после чтения ответов", "baseline": "Без фильтра лучше", "filter": "С фильтром лучше",
           "equal": "Одинаковое качество", "neither": "Оба ответа неудовлетворительны"}

def render_answer(answer, title):
    st.subheader(title)
    if answer.get("error"):
        st.error(answer["error"]["message"])
    if answer["answer"]:
        st.markdown(answer["answer"])
    if answer.get("invalid_citations"):
        st.warning("Неизвестные ссылки: " + ", ".join(answer["invalid_citations"]))
    for warning in answer.get("warnings", []):
        st.warning(warning)
    counts = answer.get("counts", {})
    st.caption(f"Найдено {counts.get('retrieved', 0)} → осталось {counts.get('passed_filter', 0)} → выбрано {counts.get('selected', 0)} → в контексте {counts.get('included', 0)}")
    if answer.get("generation_skipped"):
        st.caption("Генерация пропущена: контекста недостаточно.")
    else:
        tokens = (answer.get("usage") or {}).get("total_tokens", "неизвестно")
        st.caption(f"{answer['total_latency_ms'] / 1000:.1f} с · токены: {tokens}")
    with st.expander("Источники ответа"):
        if not answer["context_items"]:
            st.write("Фрагменты не включены в контекст.")
        for item in answer["context_items"]:
            st.caption(f"[{item['citation_id']}] {item['source']} · строки {item['line_start']}–{item['line_end']}")
            st.text(item["included_text"])

def render(root):
    st.title("Реранкинг")
    st.write("Слева — обычный поиск. Справа — выбранная обработка тех же кандидатов.")
    mode = st.radio("Обработка", ["filter", "reranker"], format_func=lambda m: {"filter": "Фильтр", "reranker": "Реранкер"}[m], horizontal=True, key="simple_processing_mode")
    config, settings = load_configuration(root)
    storage = ExperimentStorage(root / "output/day23_experiments")
    question = st.text_area("Вопрос", placeholder="Например: как выполнить фоновую задачу после перезапуска приложения?", key="simple_filter_question")
    cols = st.columns(3)
    retrieval_k = cols[0].number_input("Top-K до обработки", min_value=1, value=15, step=1, key="simple_filter_before")
    final_k = cols[1].number_input("Top-K после обработки", min_value=1, value=5, step=1, key="simple_filter_after")
    if mode == "filter":
        threshold = cols[2].number_input("Порог similarity", min_value=-1., max_value=1., value=.85, step=.01, format="%.2f", key="simple_filter_threshold")
        st.caption("Фильтр: similarity ≥ порога, порядок сохраняется. 0.85 — ручной стартовый порог, не оптимум.")
        pipeline = PipelineSettings(mode=mode, retrieval_k=int(retrieval_k), final_k=int(final_k), similarity_threshold=threshold)
    else:
        use_threshold = cols[2].checkbox("Отсекать по score реранкера", value=False, key="simple_reranker_cutoff")
        threshold = cols[2].number_input("Порог реранкера (raw score)", value=0., step=.5, format="%.2f", key="simple_reranker_threshold") if use_threshold else None
        st.caption("Реранкер заново оценивает пары «исходный вопрос + чанк», пересортировывает кандидатов и выбирает Top-K. По умолчанию отсечение выключено: откалиброванного порога этой модели пока нет.")
        if use_threshold:
            st.caption("Проходит raw score ≥ порога. Это отдельный ручной порог, не cosine similarity и не вероятность.")
        pipeline = PipelineSettings(mode=mode, retrieval_k=int(retrieval_k), final_k=int(final_k), filter_method="cross_encoder",
            reranker_model=DEFAULT_MODEL, reranker_revision=DEFAULT_REVISION, reranker_threshold=threshold)
    st.caption("Rewrite выключен. В обоих ответах одинаковые индекс, модель генерации, Top-K и бюджет контекста.")
    ready = True
    try:
        pipeline.validate()
    except ValueError:
        st.error("Top-K после обработки должен быть не больше Top-K до обработки.")
        ready = False
    paths = [p for p in (root / "output/structural.sqlite", root / "output/fixed.sqlite") if p.exists()]
    with st.expander("Дополнительно"):
        path = st.selectbox("Индекс", paths, format_func=lambda p: p.stem, key="simple_filter_index") if paths else None
        st.caption(f"Модель: {config.model or 'не настроена'}. Параметры генерации и бюджет контекста одинаковы.")
        if mode == "reranker":
            st.caption(f"Реранкер: {DEFAULT_MODEL} · revision {DEFAULT_REVISION} · CPU · до 512 токенов в паре; обрезка текста чанка отмечается в результатах.")
    if not paths:
        st.error("Нет готового индекса")
        ready = False
    if config.missing():
        st.info("Настройте LLM в .env для новых запросов. История доступна без ключа.")
        ready = False
    if st.button("Сравнить ответы", type="primary", disabled=not ready or MANAGER.active("simple_filter")):
        if not question.strip():
            st.error("Введите вопрос")
        else:
            def action(progress):
                service = RAGService(ChatCompletionsClient(config), storage)
                run = compare(service, question, load_index(path), settings, pipeline, progress)
                return export_day23_report(storage, run["run_id"])
            st.session_state["simple_filter_job"] = MANAGER.submit("simple_filter", action)
    poll("simple_filter")

    runs = [storage.get_run(row["run_id"]) for row in storage.list_runs() if row["kind"] == "day23_comparison"]
    runs = [run for run in runs if {a["mode"] for a in run["answers"]} == {"baseline", mode}]
    if not runs:
        return
    lookup = {run["run_id"]: run for run in runs}
    if st.session_state.get("simple_filter_run") not in lookup:
        st.session_state["simple_filter_run"] = runs[0]["run_id"]
    with st.expander("Предыдущие сравнения"):
        selected = st.selectbox("Сохранённый вопрос", list(lookup), key="simple_filter_run",
            format_func=lambda rid: lookup[rid]["created_at"][:16] + " · " + lookup[rid]["answers"][0]["question"])
    run = lookup[selected]
    answers = {a["mode"]: a for a in run["answers"]}
    baseline, filtered = answers["baseline"], answers[mode]
    saved_settings = run["metadata"]["pipeline"]
    st.divider()
    st.write(baseline["question"])
    saved_threshold = saved_settings["reranker_threshold"] if mode == "reranker" else saved_settings["similarity_threshold"]
    threshold_label = "отключён" if saved_threshold is None else str(saved_threshold)
    st.caption(f"Сохранённый результат: Top-K {saved_settings['retrieval_k']} → {saved_settings['final_k']}, порог {threshold_label}. Изменение настроек выше не перезапускает запрос.")
    for col, answer, title in zip(st.columns(2), (baseline, filtered), ("Без обработки", "Реранкер" if mode == "reranker" else "Фильтр")):
        with col:
            render_answer(answer, title)
    removed = sum(not c["passed_filter"] for c in filtered.get("candidates", []))
    st.write(f"Порог отсеял {removed} из {len(filtered.get('candidates', []))} найденных фрагментов.")
    if mode == "reranker":
        changed = sum(c.get("reranker_rank") != c["retrieval_rank"] for c in filtered.get("candidates", []))
        st.caption(f"Реранкер изменил позицию {changed} кандидатов · этап реранкинга: {filtered['stage_latency_ms'].get('reranker', 0) / 1000:.2f} с")
    if baseline["context_text"] and baseline["context_text"] == filtered["context_text"]:
        st.info("Контекст одинаковый: фильтрация не изменила данные для ответа. Различия текста могут быть связаны с генерацией.")
    with st.expander("Чанки: scores, порядок и отсечение"):
        from .day23 import render_candidate
        render_candidate(filtered, storage, run)
    with st.expander("Оценить качество ответов"):
        review = run.get("comparison_review") or {}
        choices = {None: CHOICES[None], "baseline": "Без обработки лучше", mode: "Реранкер лучше" if mode == "reranker" else "Фильтр лучше", "equal": CHOICES["equal"], "neither": CHOICES["neither"]}
        with st.form("simple_review_" + run["run_id"]):
            st.caption("Сравните правильность, полноту и опору на источники. Число оставшихся фрагментов само по себе не оценивает качество.")
            winner = st.radio("Какой ответ лучше?", list(choices), index=list(choices).index(review.get("winner")), format_func=choices.get)
            comment = st.text_area("Почему?", value=review.get("comment", ""))
            if st.form_submit_button("Сохранить оценку"):
                try:
                    storage.save_comparison_review(run["run_id"], winner, comment)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    with st.expander("Скачать сравнение"):
        for ext, content in report_formats(run).items():
            st.download_button(ext.upper(), content, run["run_id"] + "." + ext, key="simple_export_" + ext)
