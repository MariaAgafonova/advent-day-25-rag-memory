"""Day 25 orchestration. History/task state never enter the knowledge index."""
from dataclasses import dataclass, asdict, replace
from time import perf_counter
from uuid import uuid4
from pathlib import Path
import os
from .chat_memory import MemoryPlanner, active_memory, FIELDS
from .context_builder import ContextBuilder, TokenCounter
from .rag_models import AnswerResult, RetrievedChunk
from .post_retrieval import PipelineSettings, process_candidates
from .answer_policy import generate, generation_messages_for_budget
from .rag import check_index
from .search import search


@dataclass(frozen=True)
class ChatSettings:
    recent_messages: int = 8
    max_message_chars: int = 4000
    history_budget: int = 2500
    query_language: str = 'original'

    @classmethod
    def from_env(cls):
        result = cls(int(os.getenv('CHAT_RECENT_MESSAGES', '8')), int(os.getenv('CHAT_MAX_MESSAGE_CHARS', '4000')),
                     int(os.getenv('CHAT_HISTORY_BUDGET', '2500')), os.getenv('CHAT_QUERY_LANGUAGE', 'en'))
        if min(result.recent_messages, result.max_message_chars, result.history_budget) < 1 or result.query_language not in ('en', 'original'):
            raise ValueError('Настройки чата должны быть положительными')
        return result


class Knowledge:
    def __init__(self, index, client, embedding=None, pipeline=None, counter=None, reranker=None):
        self.index, self.client, self.embedding_service = index, client, embedding
        self.pipeline = pipeline or PipelineSettings(mode='filter', similarity_threshold=.85)
        self.counter = counter or TokenCounter(client.model)
        self.reranker = reranker

    @property
    def index(self):
        if isinstance(self._index, (str, Path)):
            from .storage import load_index
            return load_index(self._index)
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    def retrieve(self, query):
        from .embeddings import get_embedding_service
        started = perf_counter()
        version = check_index(self.index)
        embedding = self.embedding_service or get_embedding_service(self.index.config['model_id'], self.index.config['revision'])
        found = search(self.index, query, self.pipeline.retrieval_k, service=embedding)
        chunks = [RetrievedChunk(r['chunk_id'], r['source'], r['title'], r['sections'], r['line_start'],
            r['line_end'], r['page_start'], r['page_end'], r['text'], r['score'], r['rank'],
            r['start_char'], r['end_char'], self.index.documents[r['source']].content_hash, r['source_url'])
            for r in found['results']]
        retrieval_ms = (perf_counter() - started) * 1000
        start_filter = perf_counter()
        selected, candidates = process_candidates(chunks, self.pipeline, query, self.reranker)
        return selected, dict(index_version=version, pipeline=self.pipeline.snapshot(), candidates=candidates,
            retrieved=[dict(chunk_id=c.chunk_id, score=c.retrieval_score) for c in chunks],
            selected=[c.chunk_id for c in selected], stage_ms=dict(retrieval=retrieval_ms,
            filtering=(perf_counter() - start_filter) * 1000))

    def answer(self, question, selected, diagnostics, state, history, settings, config):
        conversation = dict(task_state=active_memory(state), history=[dict(role=m['role'], content=m['content'])
            for m in history if m['status'] == 'completed'][-config.recent_messages:])
        while conversation['history'] and self.counter.messages(conversation['history']) > config.history_budget:
            conversation['history'].pop(0)
        factory = lambda q, c: generation_messages_for_budget(q, c, conversation)
        while self.counter.messages(factory(question, '')) + settings.max_output_tokens + min(1000, settings.max_context_tokens) > settings.context_window_tokens:
            if conversation['history']:
                conversation['history'].pop(0)
            elif conversation['task_state']['summary']:
                conversation['task_state']['summary'] = ''
            else:
                raise ValueError('context_budget')
        start = perf_counter()
        bundle = ContextBuilder(self.counter).build(selected, question, settings, self.index.documents, message_factory=factory, allow_truncation=False)
        diagnostics['stage_ms']['context'] = (perf_counter() - start) * 1000
        diagnostics['context'] = dict(tokens=bundle.token_count, method=bundle.token_count_method,
            chunk_ids=[i.chunk_id for i in bundle.items], history_messages=len(conversation['history']), exclusions=bundle.exclusions)
        included = {item.chunk_id for item in bundle.items}
        for candidate in diagnostics['candidates']:
            candidate['included_in_context'] = candidate['chunk_id'] in included
            if candidate['selected'] and candidate['chunk_id'] not in included:
                candidate['rejection_reason'] = bundle.exclusions.get(candidate['chunk_id'], 'context_budget')
        if not bundle.items:
            return failure('context', 'no_complete_chunk_fits')
        result = AnswerResult(str(uuid4()), question, 'day25', index_version=diagnostics['index_version'])
        result.candidates = diagnostics['candidates']
        start = perf_counter()
        generate(self, result, bundle, settings, self.index, conversation)
        diagnostics['stage_ms']['generation'] = (perf_counter() - start) * 1000
        diagnostics['validation'] = result.validation
        diagnostics['generation_calls'] = result.generation_calls
        diagnostics['usage'] = result.usage
        if result.error or result.status != 'completed':
            return failure('generation', (result.error or {}).get('code', 'citation_validation'))
        valid_ids = {c.chunk_id for c in selected} & {r['chunk_id'] for r in diagnostics['retrieved']}
        sources = []
        for citation in result.structured_answer['citations']:
            source = next(s for s in result.resolved_sources if s['source_id'] == citation['context_id'])
            chunk = next(c for c in selected if c.chunk_id == source['chunk_id'])
            if source['chunk_id'] not in valid_ids or source['source'] != chunk.source or source['section'] != chunk.section or citation['quote'] not in chunk.text:
                return failure('validation', 'invalid_source')
            sources.append({**source, 'quote': citation['quote'], 'citation_id': citation['citation_id']})
        if result.answer_status == 'insufficient_context':
            # Ambiguous references are handled before generation. Lack of evidence
            # must not reopen a topic the user has already specified explicitly.
            return refusal(question)
        return dict(status=result.answer_status, answer=result.answer, sources=sources,
            claims=result.structured_answer['claims'], clarification_question=result.structured_answer['clarification_question'])


def refusal(topic=None):
    subject = (' по запросу «' + ' '.join(topic.split())[:300] + '»') if topic else ''
    return dict(status='insufficient_context', answer='Не знаю на основании доступных документов. '
                'В найденном контексте недостаточно доказательств' + subject + '. '
                'Добавьте документ, который раскрывает эту тему, и обновите индекс.', sources=[])


def failure(stage, code):
    return dict(status='error', answer='Ошибка обработки: ' + stage + '. Повторите запрос или сбросьте память текущего диалога.',
                sources=[], error=dict(stage=stage, code=code))


class ChatService:
    def __init__(self, store, knowledge, settings, planner=None, config=None):
        self.store, self.knowledge, self.settings = store, knowledge, settings
        self.config = config or ChatSettings.from_env()
        self.planner = planner or MemoryPlanner(knowledge.client, knowledge.counter, self.config.recent_messages, self.config.query_language)

    def send_message(self, cid, rid, content, retry=False):
        if not isinstance(content, str) or not content.strip() or len(content) > self.config.max_message_chars or '\x00' in content:
            raise ValueError('Сообщение должно содержать от 1 до ' + str(self.config.max_message_chars) + ' символов')
        if not isinstance(rid, str) or not 1 <= len(rid) <= 128:
            raise ValueError('Некорректный request_id')
        self.store.conversation(cid)
        reset_command = content.strip().casefold() == '/reset'
        with self.store.serial(cid):
            # Don't accept more turns while a failed memory update remains unresolved.
            if self.store.memory_blocked(cid, rid) and not self.store.request(cid, rid) and not reset_command:
                raise ValueError('Сначала повторите сообщение с ошибкой памяти или сбросьте память')
            mid, cached = self.store.begin(cid, rid, content, retry)
            if cached is not None:
                return cached
            start = perf_counter()
            state = self.store.state(cid)
            history = self.store.history(cid, True)
            current = next(m for m in history if m['id'] == mid)
            prior = [m for m in history if m['request_id'] != rid]
            diagnostics = dict(original_query=content, rewritten_query=content, task_revision=state['revision'],
                               stage_ms={}, ambiguity='', retrieval_attempted=False)
            memory_error, stage = False, 'memory'
            try:
                request = self.store.request(cid, rid)
                if request['memory_done']:
                    preparation = request['preparation']
                else:
                    tick = perf_counter()
                    try:
                        if reset_command:
                            from .chat_storage import empty_state
                            state = empty_state(cid, state['revision'] + 1)
                            prior = []
                            preparation = dict(rewritten_query=content, ambiguity='', intent='task_update', reset_memory=True)
                        else:
                            state, preparation = self.planner.prepare(state, prior, current, self.settings)
                        self.store.save_memory(cid, rid, mid, state, preparation)
                    except Exception:
                        memory_error = True
                        raise
                    finally:
                        diagnostics['stage_ms']['memory'] = (perf_counter() - tick) * 1000
                diagnostics.update(rewritten_query=preparation['rewritten_query'], ambiguity=preparation['ambiguity'],
                                   task_revision=state['revision'])
                stage = 'retrieval'
                diagnostics['retrieval_attempted'] = True
                selected, trace = self.knowledge.retrieve(preparation['rewritten_query'])
                diagnostics['stage_ms'].update(trace.pop('stage_ms'))
                diagnostics.update(trace)
                if preparation['ambiguity']:
                    result = dict(status='clarification', answer=preparation['ambiguity'], sources=[])
                elif preparation['intent'] in ('task_update', 'memory_recap'):
                    labels = dict(goal='Цель', constraints='Ограничения', clarifications='Уточнения', terms='Термины', decisions='Решения', open_questions='Открытые вопросы')
                    lines = [labels[f] + ': ' + '; '.join((i['key'] + ' — ' if f == 'terms' else '') + i['value'] + (' (предположение)' if i['origin'] == 'inferred' else '')
                             for i in state[f] if i['active']) for f in FIELDS if any(i['active'] for i in state[f])]
                    result = dict(status='clarification', answer='Память задачи обновлена.\n' + '\n'.join(lines), sources=[])
                elif not selected:
                    result = refusal(content)
                else:
                    stage = 'generation'
                    result = self.knowledge.answer(content, selected, diagnostics, state, prior, self.settings, self.config)
            except Exception:
                result = failure(stage, stage + '_error')
                if not diagnostics['retrieval_attempted']:
                    diagnostics['retrieval_attempted'] = True
                    try:
                        _, trace = self.knowledge.retrieve(content)
                        diagnostics['stage_ms'].update(trace.pop('stage_ms'))
                        diagnostics.update(trace)
                    except Exception:
                        diagnostics['retrieval_error'] = 'retrieval_error'
            diagnostics['stage_ms']['total'] = (perf_counter() - start) * 1000
            result.update(diagnostics=diagnostics, task_state=state, request_id=rid)
            self.store.finish(cid, rid, result, memory_error)
            return result


def build_chat(root, store=None, retrieval_mode=None):
    from pathlib import Path
    from .chat_storage import ChatStore
    from .llm_client import load_configuration, ChatCompletionsClient
    from .storage import load_index
    root = Path(root)
    config, settings = load_configuration(root)
    # Conservative UTF-8 counting for unknown provider tokenizers needs enough room
    # for schema + memory + evidence, still strictly bounded at every model call.
    settings = replace(settings, context_window_tokens=int(os.getenv('CHAT_CONTEXT_WINDOW', '20000')),
                       max_output_tokens=int(os.getenv('CHAT_OUTPUT_TOKENS', '1800')),
                       max_context_tokens=int(os.getenv('CHAT_DOCUMENT_BUDGET', '6000')))
    client = ChatCompletionsClient(config)
    index = root / os.getenv('CHAT_INDEX', 'output/structural.sqlite')
    mode = retrieval_mode or os.getenv('CHAT_RETRIEVAL_MODE', 'reranker')
    if mode == 'reranker':
        from .reranker import DEFAULT_MODEL, DEFAULT_REVISION
        pipeline = PipelineSettings(mode='reranker', filter_method='cross_encoder', reranker_model=DEFAULT_MODEL,
                                    reranker_revision=DEFAULT_REVISION)
    elif mode == 'filter':
        pipeline = PipelineSettings(mode='filter', similarity_threshold=float(os.getenv('CHAT_SIMILARITY_THRESHOLD', '.85')))
    else:
        raise ValueError('CHAT_RETRIEVAL_MODE должен быть filter или reranker')
    return ChatService(store or ChatStore(root / 'output/chat.sqlite'), Knowledge(index, client, pipeline=pipeline), settings)
