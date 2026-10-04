from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4
import json
import time
import pytest
from document_indexer.chat import ChatService, ChatSettings, Knowledge
from document_indexer.chat_storage import ChatStore, empty_state
from document_indexer.chat_memory import apply_preparation, MemoryPlanner
from document_indexer.rag_models import GenerationSettings
from document_indexer.context_builder import TokenCounter
from document_indexer.post_retrieval import PipelineSettings
from document_indexer.storage import load_index
from document_indexer.llm_client import LLMResponse


def prep(text='alpha', operations=None, ambiguity='', intent='question'):
    return dict(operations=operations or [], summary='', rewritten_query=text, ambiguity=ambiguity, intent=intent)


def op(field, key, value, evidence, action='set', origin='user_explicit'):
    return dict(action=action, field=field, key=key, value=value, evidence=evidence, origin=origin)


class Planner:
    def __init__(self):
        self.payload = prep()
        self.calls = 0
        self.error = False
        self.histories = []

    def prepare(self, state, history, message, settings):
        self.calls += 1
        self.histories.append(history)
        if self.error:
            raise ValueError('malformed JSON')
        return apply_preparation(state, self.payload, message), self.payload


class Retrieval:
    def __init__(self):
        self.calls = []
        self.error = False
        self.counter = TokenCounter('mock')
        self.client = SimpleNamespace(model='mock')

    def retrieve(self, query):
        self.calls.append(query)
        if self.error:
            raise RuntimeError('secret must not leak')
        return [], dict(stage_ms={'retrieval': 1}, retrieved=[], selected=[], candidates=[])


@pytest.fixture
def chat(tmp_path):
    store, retrieval, planner = ChatStore(tmp_path / 'chat.sqlite'), Retrieval(), Planner()
    service = ChatService(store, retrieval, GenerationSettings(), planner)
    cid = store.create()
    return service, store, retrieval, planner, cid


def send(chat, text='alpha', rid=None, retry=False):
    return chat[0].send_message(chat[4], rid or str(uuid4()), text, retry)


def test_restart_and_isolation(chat):
    service, store, retrieval, planner, cid = chat
    planner.payload = prep(operations=[op('goal', 'main', 'alpha', 'alpha')])
    send(chat)
    restored = ChatStore(store.path)
    assert len(restored.history(cid)) == 2
    assert restored.state(cid)['goal'][0]['value'] == 'alpha'
    other = restored.create()
    assert not restored.state(other)['goal']
    assert not restored.history(other)
    assert len(retrieval.calls) == 1


def test_replace_forget_and_side_question(chat):
    _, store, _, planner, cid = chat
    planner.payload = prep(operations=[op('goal', 'main', 'project', 'project'), op('constraints', 'language', 'Kotlin', 'Kotlin'), op('constraints', 'ui', 'web', 'web')])
    send(chat, 'project Kotlin web')
    planner.payload = prep(operations=[op('constraints', 'ui', 'CLI', 'CLI')])
    send(chat, 'Use CLI instead')
    state = store.state(cid)
    assert [(i['value'], i['active']) for i in state['constraints']] == [('Kotlin', True), ('web', False), ('CLI', True)]
    assert state['constraints'][1]['superseded_by'] == state['constraints'][2]['id']
    planner.payload = prep('BROWSABLE')
    send(chat, 'Side question BROWSABLE?')
    assert store.state(cid)['goal'][0]['value'] == 'project'
    planner.payload = prep(operations=[op('constraints', 'ui', '', 'forget UI', action='remove')])
    send(chat, 'forget UI')
    assert [i['value'] for i in store.state(cid)['constraints'] if i['active']] == ['Kotlin']


def test_followup_every_turn_and_ambiguity(chat):
    _, _, retrieval, planner, _ = chat
    send(chat, 'Compare options')
    planner.payload = prep('JSON persistence after restart')
    send(chat, 'And second option?')
    assert retrieval.calls[-1] == 'JSON persistence after restart'
    planner.payload = prep('available topic', ambiguity='Which option?')
    assert send(chat, 'And that?')['status'] == 'clarification'
    planner.payload = prep(intent='task_update')
    assert send(chat, 'only Kotlin')['sources'] == []
    assert len(retrieval.calls) == 4


def test_memory_error_retains_message_and_blocks_stale_generation(chat):
    _, store, retrieval, planner, cid = chat
    planner.error = True
    result = send(chat, rid='stable')
    assert result['status'] == 'error' and result['error']['stage'] == 'memory'
    assert store.state(cid)['revision'] == 0 and len(store.history(cid)) == 2
    assert len(retrieval.calls) == 1
    with pytest.raises(ValueError, match='Сначала'):
        send(chat, 'new text')
    planner.error = False
    assert send(chat, rid='stable', retry=True)['status'] == 'insufficient_context'
    assert len(store.history(cid)) == 2 and store.state(cid)['revision'] == 1


def test_retrieval_error_is_not_absence_of_knowledge_and_retry_reuses_memory(chat):
    _, store, retrieval, planner, cid = chat
    retrieval.error = True
    result = send(chat, rid='stable')
    assert result['status'] == 'error' and result['error']['stage'] == 'retrieval'
    assert 'secret' not in json.dumps(result)
    retrieval.error = False
    result = send(chat, rid='stable', retry=True)
    assert result['status'] == 'insufficient_context' and result['sources'] == []
    assert planner.calls == 1 and store.state(cid)['revision'] == 1


def test_idempotency_and_input_validation(chat):
    first = send(chat, rid='same')
    assert send(chat, rid='same') == first
    assert len(chat[2].calls) == 1
    with pytest.raises(ValueError, match='другого'):
        send(chat, 'different', rid='same')
    for invalid in ('', '  ', 'x' * 4001, '\x00'):
        with pytest.raises(ValueError):
            send(chat, invalid)
    with pytest.raises(ValueError):
        chat[0].send_message(str(uuid4()), 'id', 'alpha')


def test_reset_preserves_history_but_cuts_memory_context(chat):
    _, store, _, planner, cid = chat
    planner.payload = prep(operations=[op('goal', 'main', 'alpha', 'alpha')])
    send(chat)
    other = store.create()
    before = store.state(other)
    store.reset(cid)
    assert not store.state(cid)['goal'] and len(store.history(cid)) == 2
    assert store.state(other) == before
    planner.payload = prep()
    send(chat, 'new task')
    assert planner.histories[-1] == []
    assert len(store.revisions(cid)) == 3


def test_interrupted_request_recovers_without_duplicates(chat):
    _, store, _, planner, cid = chat
    with store.serial(cid):
        store.begin(cid, 'crash', 'alpha')
    restarted = ChatStore(store.path)
    with restarted.serial(cid):
        pass
    assert restarted.request(cid, 'crash')['result']['error']['stage'] == 'interrupted'
    send(chat, rid='crash', retry=True)
    assert len(restarted.history(cid)) == 2


def test_reset_command_runs_retrieval_and_cuts_failed_memory(chat):
    _, store, retrieval, planner, cid = chat
    planner.error = True
    send(chat)
    result = send(chat, '/reset')
    assert result['status'] == 'clarification'
    assert len(retrieval.calls) == 2 and not store.state(cid)['goal']
    assert len(store.history(cid)) == 4
    planner.error = False
    send(chat, 'new question')
    assert all(m['content'] != 'alpha' for m in planner.histories[-1])


def test_both_long_scenarios_and_restart(tmp_path):
    from document_indexer.day25_evaluation import run
    from pathlib import Path
    result = run(Path(__file__).resolve().parents[1], mock=True, output=tmp_path / 'report')
    assert result['scenarios'] == {'A': 12, 'B': 12}
    report = json.loads((tmp_path / 'report/results.json').read_text(encoding='utf-8'))
    for scenario in report['scenarios'].values():
        assert len(scenario['history']) == 24 and scenario['restart_verified']
        assert all(step['actual']['diagnostics']['retrieval_attempted'] for step in scenario['steps'])


def test_small_context_errors_without_losing_constraints(built, service):
    from document_indexer.day25_mock import MockClient
    index = load_index(built[1]['structural'])
    knowledge = Knowledge(index, MockClient(), service, PipelineSettings(mode='filter', similarity_threshold=.8))
    selected, diagnostics = knowledge.retrieve('alpha')
    state = empty_state('cid')
    state['constraints'] = [dict(id='x', key='important', value='Kotlin ' * 80, origin='user_explicit', message_id='m', active=True)]
    with pytest.raises(ValueError, match='context_budget'):
        knowledge.answer('alpha', selected, diagnostics, state, [], GenerationSettings(context_window_tokens=600), ChatSettings())
    assert state['constraints'][0]['active']


def test_missing_index_persists_user_and_reports_retrieval_error(tmp_path):
    from document_indexer.day25_mock import MockClient
    store = ChatStore(tmp_path / 'chat.sqlite')
    chat = ChatService(store, Knowledge(tmp_path / 'missing.sqlite', MockClient()), GenerationSettings(), Planner())
    cid = store.create()
    result = chat.send_message(cid, 'id', 'alpha')
    assert result['status'] == 'error' and result['error']['stage'] == 'retrieval'
    assert len(store.history(cid)) == 2


def test_storage_survives_a_new_python_process(chat):
    import subprocess
    import sys
    send(chat)
    program = 'import sys,json; from document_indexer.chat_storage import ChatStore; s=ChatStore(sys.argv[1]); print(json.dumps([len(s.history(sys.argv[2])),s.state(sys.argv[2])["revision"]]))'
    output = subprocess.check_output([sys.executable, '-c', program, str(chat[1].path), chat[4]], text=True)
    assert json.loads(output) == [2, 1]


def test_diagnostics_identify_only_actual_context_chunks():
    from document_indexer.day25_mock import mock_index, MockClient, MockEmbedding
    knowledge = Knowledge(mock_index(), MockClient(), MockEmbedding())
    selected, diagnostics = knowledge.retrieve('WorkManager')
    result = knowledge.answer('WorkManager', selected, diagnostics, empty_state('cid'), [], GenerationSettings(context_window_tokens=20000), ChatSettings())
    assert result['status'] == 'answered'
    included = {c['chunk_id'] for c in diagnostics['candidates'] if c['included_in_context']}
    assert included == set(diagnostics['context']['chunk_ids'])
    assert {s['chunk_id'] for s in result['sources']} <= included


def test_concurrent_same_request_is_processed_once(chat):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: send(chat, rid='same'), range(2)))
    assert results[0] == results[1]
    assert chat[3].calls == 1
    assert len(chat[1].history(chat[4])) == 2


def test_concurrent_distinct_messages_are_serial(chat):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: send(chat, rid=str(i)), range(3)))
    assert sorted(r['task_state']['revision'] for r in results) == [1, 2, 3]
    assert len(chat[2].calls) == 3


@pytest.mark.parametrize('mutation', ['evidence', 'extra', 'inferred_replace', 'unknown_remove'])
def test_memory_schema_rejects_invalid_mutations(mutation):
    state = empty_state('cid')
    msg = dict(id='m1', content='Kotlin')
    state = apply_preparation(state, prep(operations=[op('constraints', 'language', 'Kotlin', 'Kotlin')]), msg)
    original = json.dumps(state)
    data = prep(operations=[op('constraints', 'language', 'Java', 'Kotlin')])
    if mutation == 'evidence': data['operations'][0]['evidence'] = 'invented'
    if mutation == 'extra': data['secret'] = 'unexpected'
    if mutation == 'inferred_replace': data['operations'][0]['origin'] = 'inferred'
    if mutation == 'unknown_remove': data['operations'][0].update(action='remove', key='missing')
    with pytest.raises(ValueError):
        apply_preparation(state, data, msg)
    assert json.dumps(state) == original


def test_real_memory_planner_validates_json_and_bounds_history():
    class Client:
        def complete_structured(self, messages, settings, schema):
            self.messages = messages
            return LLMResponse(json.dumps(prep()), 'mock', 'mock', 0, finish_reason='stop')
    client = Client()
    planner = MemoryPlanner(client, TokenCounter('mock'), 2)
    history = [dict(id=str(i), role='user', content='test ' + str(i), status='completed') for i in range(20)]
    planner.prepare(empty_state('cid'), history, dict(id='new', content='alpha'), GenerationSettings(context_window_tokens=16000))
    assert len(json.loads(client.messages[1]['content'])['history']) == 2


def test_plan_request_is_not_swallowed_as_memory_update():
    from document_indexer.chat_memory import requested_intent
    for text in ('Составь план', 'Сократи до трёх пунктов', 'Подведи итог', 'Продолжай'):
        assert requested_intent(text, 'task_update') == 'question'
    assert requested_intent('Без платных сервисов', 'task_update') == 'task_update'


@pytest.mark.parametrize('bad', ['id', 'quote', 'filtered'])
def test_invalid_evidence_rejected_by_real_rag_path(built, service, bad):
    class Client:
        model, provider = 'mock', 'mock'
        def complete_structured(self, messages, settings, schema):
            payload = dict(status='answered', answer='alpha', claims=[dict(claim_id='c', text='alpha', citation_ids=['q'])],
                citations=[dict(citation_id='q', context_id='S999' if bad in ('id', 'filtered') else 'S1', quote='never in corpus' if bad == 'quote' else 'alpha')],
                clarification_question=None)
            return LLMResponse(json.dumps(payload), 'mock', 'mock', 0, finish_reason='stop')
    knowledge = Knowledge(load_index(built[1]['structural']), Client(), service,
                          pipeline=PipelineSettings(mode='filter', similarity_threshold=.8))
    selected, diagnostics = knowledge.retrieve('alpha')
    result = knowledge.answer('alpha', selected, diagnostics, empty_state('cid'), [], GenerationSettings(context_window_tokens=20000), ChatSettings())
    assert result['status'] == 'error' and result['sources'] == []
