from dataclasses import asdict
from time import perf_counter
from uuid import uuid4
from .rag_models import AnswerResult, RetrievedChunk
from .query_rewrite import QueryRewriter, RewriteResult
from .context_builder import ContextBuilder
from .citations import validate_citations
from .prompts import PROMPT_VERSION, messages_with_rag
from .search import search
from .embeddings import get_embedding_service
from .post_retrieval import process_candidates

def compare(service, question, index, settings, pipeline, progress=None):
    from dataclasses import replace
    if pipeline.mode == "baseline":
        raise ValueError("Выберите улучшенный режим для сравнения")
    pipeline.validate()
    run_id = service.storage.create_run("day23_comparison", {"planned_calls": 2, "planned_answers": 2,
        "settings": asdict(settings), "pipeline": pipeline.snapshot(), "report_modes": ["baseline", pipeline.mode]}, index)
    answer(service, question, index, settings, replace(pipeline, mode="baseline"), run_id, "q01", progress)
    answer(service, question, index, settings, pipeline, run_id, "q01", progress)
    service.storage.finish_run(run_id)
    return service.storage.get_run(run_id)

def answer(service, question, index, settings, pipeline, run_id, question_id, progress=None, rewrite=None,
           structured_evidence=False, paired=False):
    from .rag import check_index
    start = perf_counter()
    emit = progress or (lambda *args: None)
    result = AnswerResult(str(uuid4()), question.strip(), pipeline.mode,
        model=service.client.model, provider=service.client.provider, prompt_version=PROMPT_VERSION,
        generation_settings={**asdict(settings), **getattr(service.client, "generation_options", {})},
        run_id=run_id, question_id=question_id, original_query=question.strip(),
        pipeline_settings=pipeline.snapshot())
    if structured_evidence:
        result.mode, result.schema_version = 'day24', 3
        from .answer_policy import PROMPT_VERSION as evidence_prompt_version
        result.prompt_version = evidence_prompt_version
    stage = "validation"
    timings = {key: 0. for key in ("rewrite", "retrieval", "post_retrieval", "reranker", "context", "generation")}
    stage_start = None
    try:
        pipeline.validate()
        settings.validate()
        if not result.question:
            raise ValueError("Введите непустой вопрос")
        result.index_version = check_index(index)
        if pipeline.calibration_id:
            from .day23_evaluation import validate_calibration
            validate_calibration(service.storage.get_artifact(pipeline.calibration_id), index, pipeline, service.client, settings)
        stage = "retrieval"
        embedding = service.embedding_service or get_embedding_service(index.config["model_id"], index.config["revision"])
        rewritten = RewriteResult(result.question, result.question)
        if pipeline.rewrite_enabled:
            stage = "rewrite"
            emit(stage, 0, None)
            if rewrite is not None:
                if rewrite.original_query != result.question or rewrite.model != service.client.model:
                    raise ValueError("Rewrite не соответствует исходному вопросу или модели")
                rewritten = rewrite
            else:
                rewritten = QueryRewriter(service.client).rewrite(result.question, settings, embedding)
            timings["rewrite"] = 0 if rewrite is not None else rewritten.latency_ms
        result.rewrite = {**asdict(rewritten), "reused": rewrite is not None and pipeline.rewrite_enabled}
        result.rewritten_query = rewritten.rewritten_query
        stage = "retrieval"
        emit(stage, 0, None)
        stage_start = perf_counter()
        found = search(index, result.rewritten_query, pipeline.retrieval_k, service=embedding)
        timings[stage] = (perf_counter() - stage_start) * 1000
        for r in found["results"]:
            result.retrieved_chunks.append(RetrievedChunk(r["chunk_id"], r["source"], r["title"], r["sections"],
                r["line_start"], r["line_end"], r["page_start"], r["page_end"], r["text"], r["score"], r["rank"],
                r["start_char"], r["end_char"], index.documents[r["source"]].content_hash, r["source_url"]))
        stage = "reranker" if pipeline.rerank_enabled else "post_retrieval"
        emit(stage, len(result.retrieved_chunks), None)
        stage_start = perf_counter()
        if pipeline.rerank_enabled and not result.retrieved_chunks:
            selected, result.candidates = [], []
        else:
            selected, result.candidates = process_candidates(result.retrieved_chunks, pipeline, result.question, getattr(service, "reranker_service", None))
        decisions = {c["chunk_id"]: c for c in result.candidates}
        result.processed_chunks = [dict(decisions[c.chunk_id]) for c in selected]
        truncated = sum(c.get("reranker_truncated", False) for c in result.candidates)
        if truncated:
            result.warnings.append(f"Реранкер: обрезаны тексты {truncated} кандидатов до лимита пары {pipeline.reranker_max_length} токенов; исходный вопрос сохранён полностью")
        timings[stage] = (perf_counter() - stage_start) * 1000
        stage = "context"
        emit(stage, len(selected), None)
        stage_start = perf_counter()
        if structured_evidence:
            from .answer_policy import generation_messages_for_budget
            bundle = ContextBuilder(service.counter).build(selected, result.question, settings, index.documents,
                                                          message_factory=generation_messages_for_budget)
        else:
            bundle = ContextBuilder(service.counter).build(selected, result.question, settings, index.documents)
        result.context_items, result.context_text = bundle.items, bundle.text
        result.context_token_count, result.token_count_method = bundle.token_count, bundle.token_count_method
        timings[stage] = (perf_counter() - stage_start) * 1000
        result.messages = messages_with_rag(result.question, bundle.text)
        if structured_evidence:
            from .answer_policy import messages_with_evidence
            result.messages = messages_with_evidence(result.question, bundle.text)
        included = {i.chunk_id for i in bundle.items}
        for c in result.candidates:
            c["included_in_context"] = c["chunk_id"] in included
            if c["selected"] and not c["included_in_context"]:
                # ContextBuilder records the exact exclusion decision below.
                c["rejection_reason"] = bundle.exclusions.get(c["chunk_id"], "context_budget")
        result.counts = {"retrieved": len(result.candidates), "passed_filter": sum(c["passed_filter"] for c in result.candidates),
                         "selected": len(selected), "included": len(included), "cited": 0}
        if paired:
            from .day24_evaluation import generate_baseline
            timings['preparation_overhead'] = max(0., (perf_counter() - start) * 1000 - sum(timings.values()))
            generate_baseline(service, result, bundle, settings, timings, run_id, question_id)
        if not bundle.items:
            result.generation_skipped = True
            result.status = "no_relevant_context" if result.retrieved_chunks and not selected else "no_results" if not result.retrieved_chunks else "empty_context"
            prefix = "После фильтрации подходящие фрагменты не найдены. " if result.status == "no_relevant_context" else ""
            result.answer = prefix + "Недостаточно предоставленного контекста для ответа на вопрос."
            if structured_evidence:
                from .answer_policy import refuse
                reason = 'no_retrieval_results' if not result.retrieved_chunks else 'below_relevance_threshold' if not selected else 'insufficient_evidence'
                refuse(result, reason)
        elif structured_evidence:
            from .answer_policy import generate
            stage = 'generation'
            emit(stage, 0, None)
            stage_start = perf_counter()
            generate(service, result, bundle, settings, index)
            timings[stage] = (perf_counter() - stage_start) * 1000
        else:
            stage = "generation"
            emit(stage, 0, None)
            stage_start = perf_counter()
            result.generation_calls = 1
            response = service.client.complete(result.messages, settings)
            timings[stage] = (perf_counter() - stage_start) * 1000
            result.answer, result.usage = response.text, response.usage
            result.api_request, result.provider_attempts = response.api_request, response.attempts
            result.finish_reason, result.error = response.finish_reason, response.error
            if result.error:
                result.error = {**result.error, "stage": "generation"}
            result.status = "failed" if response.error else "completed"
            if not response.error:
                result.citations, result.invalid_citations = validate_citations(result.answer, bundle.items)
                if result.invalid_citations:
                    result.status = "citation_error"
                if not result.citations:
                    result.warnings.append("Нет валидных ссылок: требуется ручная проверка")
                if response.finish_reason == "length":
                    result.warnings.append("Ответ остановлен по лимиту токенов")
                cited = {i.chunk_id for i in bundle.items if i.citation_id in result.citations}
                for c in result.candidates:
                    c["cited"] = c["chunk_id"] in cited
                result.counts["cited"] = len(cited)
    except Exception as exc:
        from .reranker import RerankerError
        if stage in timings and stage_start is not None:
            timings[stage] = (perf_counter() - stage_start) * 1000
        result.status = "failed"
        result.error = {"code": stage + "_error", "stage": stage,
                        "message": str(exc) if isinstance(exc, RerankerError) or (stage == "validation" and isinstance(exc, ValueError)) else "Ошибка этапа " + stage}
    result.stage_latency_ms = timings
    result.retrieval_latency_ms, result.generation_latency_ms = timings["retrieval"], timings["generation"]
    result.total_latency_ms = sum(timings.values()) if paired else (perf_counter() - start) * 1000
    service.storage.save_answer(run_id, question_id, result)
    emit("готово: " + pipeline.mode, 1, 1)
    return result
