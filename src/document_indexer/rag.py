from dataclasses import asdict
from uuid import uuid4
from time import perf_counter
from .rag_models import AnswerResult, RetrievedChunk
from .prompts import PROMPT_VERSION, messages_with_rag, messages_without_rag
from .context_builder import ContextBuilder, TokenCounter
from .citations import validate_citations
from .search import search
from .loaders import corpus_hash, load_corpus
from .embeddings import validate_vectors

def check_index(index, raw_directory=None):
    required = ["build_id", "corpus_hash", "model_id", "revision", "dimension"]
    if any(not index.config.get(k) for k in required):
        raise ValueError("В конфигурации индекса не хватает модели или версии корпуса")
    if not index.chunks:
        raise ValueError("Индекс пуст: нет чанков")
    validate_vectors(index.matrix, len(index.chunks), index.config["dimension"])
    for c in index.chunks:
        doc = index.documents.get(c.source)
        if doc is None or not c.text.strip() or not 0 <= c.start_char < c.end_char <= len(doc.text) or doc.text[c.start_char:c.end_char] != c.text:
            raise ValueError("Индекс не содержит корректный текст и координаты источника")
    warnings = []
    if raw_directory is not None:
        docs, errors = load_corpus(raw_directory)
        if errors or corpus_hash(docs) != index.config["corpus_hash"]:
            warnings.append("Индекс требует обновления. Ответ использует сохранённую версию базы.")
    return {"build_id": index.config["build_id"], "corpus_hash": index.config["corpus_hash"],
            "strategy": index.config["strategy"], "model_id": index.config["model_id"], "revision": index.config["revision"], "warnings": warnings}

class RAGService:
    def __init__(self, client, storage, embedding_service=None, counter=None, reranker_service=None):
        self.client, self.storage = client, storage
        self.embedding_service = embedding_service
        self.reranker_service = reranker_service
        self.counter = counter or TokenCounter(client.model)

    def answer_with_evidence(self, question, index, settings, pipeline, progress=None):
        from .day23_pipeline import answer
        from .day24_evaluation import run_metadata
        run_id = self.storage.create_run('day24_single', run_metadata(index, settings, pipeline, self.client, 1), index)
        answer(self, question, index, settings, pipeline, run_id, 'q01', progress, structured_evidence=True)
        self.storage.finish_run(run_id)
        return self.storage.get_run(run_id)

    def _answer(self, question, index, settings, mode, run_id, question_id, progress):
        started = perf_counter()
        result = AnswerResult(str(uuid4()), question.strip(), mode, model=self.client.model, provider=self.client.provider,
                              generation_settings={**asdict(settings), **getattr(self.client, "generation_options", {})}, prompt_version=PROMPT_VERSION)
        emit = progress or (lambda *args: None)
        try:
            settings.validate()
            if not result.question:
                raise ValueError("Введите непустой вопрос")
            if mode == "rag":
                result.index_version = check_index(index)
                emit("поиск", 0, None)
                retrieval_start = perf_counter()
                output = search(index, result.question, settings.top_k, service=self.embedding_service)
                result.retrieval_latency_ms = (perf_counter() - retrieval_start) * 1000
                for r in output["results"]:
                    result.retrieved_chunks.append(RetrievedChunk(r["chunk_id"], r["source"], r["title"], r["sections"],
                        r["line_start"], r["line_end"], r["page_start"], r["page_end"], r["text"], r["score"], r["rank"],
                        r["start_char"], r["end_char"], index.documents[r["source"]].content_hash, r["source_url"]))
                emit("подготовка контекста", len(result.retrieved_chunks), None)
                bundle = ContextBuilder(self.counter).build(result.retrieved_chunks, result.question, settings, index.documents)
                result.context_items, result.context_token_count, result.token_count_method = bundle.items, bundle.token_count, bundle.token_count_method
                result.messages = messages_with_rag(result.question, bundle.text)
                if not bundle.items:
                    result.warnings.append("Контекст пуст: в prompt явно указана недостаточность информации")
            else:
                result.messages = messages_without_rag(result.question)
            if self.counter.messages(result.messages) + settings.max_output_tokens > settings.context_window_tokens:
                raise ValueError("Prompt не помещается в окно модели с резервом ответа")
            emit("генерация: " + mode, 0, None)
            response = self.client.complete(result.messages, settings)
            result.provider_attempts = response.attempts
            result.api_request = response.api_request
            result.answer, result.generation_latency_ms = response.text, response.latency_ms
            result.usage, result.finish_reason, result.error = response.usage, response.finish_reason, response.error
            result.status = "failed" if response.error else "completed"
            if not response.error:
                result.citations, result.invalid_citations = validate_citations(result.answer, result.context_items)
                if result.invalid_citations:
                    result.status = "citation_error"
                if mode == "rag" and result.context_items and not result.citations:
                    result.warnings.append("Нет валидных ссылок: требуется ручная проверка ответа")
                if response.finish_reason == "length":
                    result.warnings.append("Ответ остановлен по лимиту токенов")
        except ValueError as exc:
            result.status, result.error = "failed", {"code": "validation", "message": str(exc)}
        except Exception:
            # Do not persist exception text: provider exceptions can contain headers/secrets.
            result.status, result.error = "failed", {"code": "pipeline_error", "message": "Ошибка поиска или генерации; проверьте доступность модели и индекса"}
        result.total_latency_ms = (perf_counter() - started) * 1000
        self.storage.save_answer(run_id, question_id, result)
        emit("готово: " + mode, 1, 1)
        return result

    def _single(self, question, index, settings, mode, run_id=None, question_id="q01", progress=None):
        owns_run = run_id is None
        if owns_run:
            run_id = self.storage.create_run("single", {"planned_calls": 1, "settings": asdict(settings)}, index)
        result = self._answer(question, index, settings, mode, run_id, question_id, progress)
        if owns_run:
            self.storage.finish_run(run_id)
        return result

    def answer_without_rag(self, question, settings, **kwargs):
        return self._single(question, None, settings, "no-rag", **kwargs)

    def answer_with_rag(self, question, index, settings, **kwargs):
        return self._single(question, index, settings, "rag", **kwargs)

    def compare_answers(self, question, index, settings, progress=None):
        run_id = self.storage.create_run("comparison", {"planned_calls": 2, "settings": asdict(settings)}, index)
        self.answer_without_rag(question, settings, run_id=run_id, progress=progress)
        self.answer_with_rag(question, index, settings, run_id=run_id, progress=progress)
        self.storage.finish_run(run_id)
        return self.storage.get_run(run_id)

    def run_query(self, question, mode, settings, index=None, progress=None, pipeline=None):
        from .post_retrieval import ALL_MODES, PipelineSettings
        if mode in ALL_MODES:
            from dataclasses import replace
            from .day23_pipeline import answer
            pipeline = replace(pipeline or PipelineSettings(), mode=mode)
            pipeline.validate()
            run_id = self.storage.create_run("day23_single", {"planned_calls": 1, "planned_answers": 1,
                "settings": asdict(settings), "pipeline": pipeline.snapshot(), "report_modes": [mode]}, index)
            answer(self, question, index, settings, pipeline, run_id, "q01", progress)
            self.storage.finish_run(run_id)
            return self.storage.get_run(run_id)
        if mode == "compare":
            return self.compare_answers(question, index, settings, progress)
        if mode not in {"rag", "no-rag"}:
            raise ValueError("Неизвестный режим")
        run_id = self.storage.create_run("single", {"planned_calls": 1, "settings": asdict(settings)}, index if mode == "rag" else None)
        self._single(question, index if mode == "rag" else None, settings, mode, run_id, progress=progress)
        self.storage.finish_run(run_id)
        return self.storage.get_run(run_id)
