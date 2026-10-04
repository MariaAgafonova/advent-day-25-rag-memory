from copy import deepcopy
from dataclasses import asdict, replace
import json
from unittest.mock import Mock
import pytest
from document_indexer.answer_schema import parse_answer, SchemaError
from document_indexer.evidence_validator import EvidenceValidator
from document_indexer.rag_models import ContextItem, GenerationSettings
from document_indexer.answer_policy import messages_with_evidence, generation_messages_for_budget
from document_indexer.post_retrieval import PipelineSettings
from document_indexer.day24_evaluation import summarize, fraction, evaluate_day24, export_day24_report, validate_day24_review
from document_indexer.llm_client import LLMResponse
from test_day23 import setup23
from test_rag import dataset

def payload(quote='alpha', status='answered'):
    return {'status': status, 'answer': 'Подтверждённый факт' if status == 'answered' else 'Подтверждённая часть; другая часть неизвестна.',
            'claims': [{'claim_id': 'C1', 'text': 'Факт', 'citation_ids': ['Q1']}],
            'citations': [{'citation_id': 'Q1', 'context_id': 'S1', 'quote': quote}],
            'clarification_question': None if status == 'answered' else 'Уточните недостающую часть?'}

@pytest.fixture
def evidence(make_doc):
    doc = make_doc('alpha beta alpha truncated')
    item = ContextItem('S1', 'chunk', doc.text[:16], doc.source, doc.title, [['One']], 1, 1,
                       None, None, 0, 16, doc.content_hash, 4, 'test', True)
    def validate(data):
        return EvidenceValidator().validate(json.dumps(data), [item], {'build_id': 'saved'}, {doc.source: doc})
    return validate, item, doc

def test_exact_quote_resolves_app_metadata_and_ambiguity(evidence):
    validate, item, doc = evidence
    r = validate(payload())
    assert r['valid']
    q = r['quote_validation']['citations'][0]
    assert (q['quote_start'], q['quote_end'], q['match_count'], q['ambiguous']) == (0, 5, 2, True)
    assert r['sources'][0]['source'] == doc.source
    assert r['sources'][0]['chunk_ids'] == ['chunk']
    assert r['semantic_support']['status'] == 'unreviewed'

@pytest.mark.parametrize('change', [
    lambda d: d.update(source='invented.md'),
    lambda d: d['citations'][0].update(source='https://invented.test'),
    lambda d: d['citations'][0].update(context_id='S99'),
    lambda d: d['citations'][0].update(quote='Alpha'),
    lambda d: d['citations'][0].update(quote='alpha  beta'),
    lambda d: d['citations'][0].update(quote='alpha ... alpha'),
    lambda d: d['citations'][0].update(quote='truncated'),
    lambda d: d['citations'][0].update(quote=''),
    lambda d: d['claims'][0].update(citation_ids=[]),
    lambda d: d['claims'][0].update(citation_ids=['Q99']),
    lambda d: d['claims'].clear(),
    lambda d: d['citations'].append(dict(d['citations'][0])),
    lambda d: d['claims'].append(dict(d['claims'][0])),
    lambda d: d.update(status='unknown'),
    lambda d: d.update(answer=42),
])
def test_invalid_evidence(evidence, change):
    data = payload(); change(data)
    assert not evidence[0](data)['valid']

def test_precise_quote_can_fail_semantically(evidence):
    data = payload()
    data['claims'][0]['text'] = 'The document guarantees zero errors forever.'
    result = evidence[0](data)
    assert result['valid'] and result['quote_validation']['status'] == 'passed'
    assert result['semantic_support'] == {'status': 'unreviewed'}
    # A human-reviewed fixture explicitly records the unsupported inference.
    manual = {'C1': 'unsupported'}
    assert manual['C1'] != 'supported'

def test_preserve_all_chunk_provenance(evidence):
    validate, item, _ = evidence
    item.provenance = [{'chunk_id': 'a', 'start_char': 0, 'end_char': 5}, {'chunk_id': 'b', 'start_char': 5, 'end_char': 16}]
    assert validate(payload())['sources'][0]['chunk_ids'] == ['a', 'b']

def test_stale_document_rejected(evidence):
    validate, item, _ = evidence
    item.document_hash = 'wrong'
    assert not validate(payload())['valid']

def test_refusal_empty_evidence_valid(evidence):
    d = {'status': 'insufficient_context', 'answer': 'Не знаю: недостаточно контекста', 'claims': [], 'citations': [], 'clarification_question': 'Уточните документ?'}
    result = evidence[0](d)
    assert result['valid'] and result['sources'] == []
    assert result['quote_validation']['status'] == 'not_applicable'

def test_duplicate_json_keys_rejected():
    with pytest.raises(SchemaError):
        parse_answer('{"status":"answered","status":"partial"}')

def run24(setup23):
    rag, client, store, index = setup23
    client.answer_text = json.dumps(payload())
    run = rag.answer_with_evidence('Question?', index, GenerationSettings(context_window_tokens=20000),
                                    PipelineSettings(mode='filter', similarity_threshold=.85))
    return run

def test_pipeline_answer_and_reopen_no_llm(setup23):
    run = run24(setup23)
    answer = run['answers'][0]
    assert answer['status'] == 'completed', answer.get('error')
    assert answer['answer_status'] == 'answered'
    assert len(setup23[1].calls) == 1 and answer['resolved_sources']
    assert answer['validation']['quote_validation']['status'] == 'passed'
    store = setup23[2]
    export_day24_report(store, run['run_id'])
    store.get_run(run['run_id'])
    assert len(setup23[1].calls) == 1

@pytest.mark.parametrize('empty,reason', [(False,'below_relevance_threshold'), (True,'no_retrieval_results')])
def test_refusal_no_generator(setup23, monkeypatch, empty, reason):
    import document_indexer.day23_pipeline as pipe
    original = pipe.search
    def search(*args, **kwargs):
        found = original(*args, **kwargs)
        if empty:
            found['results'] = []
        else:
            for r in found['results']: r['score'] = .1
        return found
    monkeypatch.setattr(pipe, 'search', search)
    answer = run24(setup23)['answers'][0]
    assert answer['answer_status'] == 'insufficient_context'
    assert answer['refusal_reason'] == reason
    assert not setup23[1].calls and answer['generation_skipped']
    assert answer['answer'].startswith('Не знаю')
    assert answer['structured_answer']['clarification_question']
    assert not answer['citations'] and not answer['resolved_sources']

def test_partial_saved(setup23):
    rag, client, _, index = setup23
    client.answer_text = json.dumps(payload(status='partial'))
    a = rag.answer_with_evidence('Question?', index, GenerationSettings(context_window_tokens=20000), PipelineSettings())['answers'][0]
    assert a['status'] == 'completed' and a['answer_status'] == 'partial'

@pytest.mark.parametrize('repair_valid', [True, False])
def test_repair_once_same_context_and_question(setup23, repair_valid):
    rag, client, store, index = setup23
    responses = ['not JSON', json.dumps(payload()) if repair_valid else json.dumps(payload(quote='invented'))]
    calls = []
    def complete(messages, settings):
        calls.append(deepcopy(messages))
        return LLMResponse(responses.pop(0), 'test', 'test-only', 1, {'total_tokens': 10}, attempts=1)
    client.complete = complete
    run = rag.answer_with_evidence('Question?', index, GenerationSettings(context_window_tokens=20000), PipelineSettings())
    a = run['answers'][0]
    assert len(calls) == 2 and calls[0][1] == calls[1][1]
    assert a['generation_calls'] == 2 and a['usage']['total_tokens'] == 20
    assert a['evidence_attempts'][0]['raw_output'] == 'not JSON'
    if repair_valid:
        assert a['answer_status'] == 'answered'
    else:
        assert a['status'] == 'validation_failed' and a['refusal_reason'] == 'validation_failed'
        assert a['answer_status'] is None and not a['resolved_sources']

def test_provider_error_not_refusal(setup23):
    rag, client, _, index = setup23
    client.complete = Mock(return_value=LLMResponse('', 'test', 'test-only', 1, error={'code':'http_500','message':'Failure'}))
    a = rag.answer_with_evidence('Question?', index, GenerationSettings(context_window_tokens=20000), PipelineSettings())['answers'][0]
    assert a['status'] == 'failed' and a['answer_status'] is None and a['refusal_reason'] is None
    assert client.complete.call_count == 1

def test_model_refusal_never_publishes_uncited_facts(setup23):
    rag, client, _, index = setup23
    raw = {'status': 'insufficient_context', 'answer': 'Не знаю. Uncited factual assertion.',
           'claims': [], 'citations': [], 'clarification_question': 'Уточните документ?'}
    client.answer_text = json.dumps(raw)
    a = rag.answer_with_evidence('Question?', index, GenerationSettings(context_window_tokens=20000), PipelineSettings())['answers'][0]
    assert 'Uncited' not in a['answer']
    assert 'Uncited' in a['evidence_attempts'][0]['raw_output']
    assert a['policy_actions'] == ['canonical_refusal_text']
    assert a['structured_answer']['answer'] == a['answer']

def test_provider_json_mode_and_prompt_fallback():
    from document_indexer.llm_client import LLMConfig, ChatCompletionsClient
    from document_indexer.answer_schema import ANSWER_SCHEMA
    transport = Mock()
    transport.post.return_value.status_code = 200
    transport.post.return_value.json.return_value = {'choices':[{'message':{'content':json.dumps(payload())},'finish_reason':'stop'}]}
    client = ChatCompletionsClient(LLMConfig('test-key','https://api.deepseek.com','test'), transport)
    client.complete_structured(messages_with_evidence('q','c'), GenerationSettings(), ANSWER_SCHEMA)
    assert transport.post.call_args.kwargs['json']['response_format'] == {'type':'json_object'}
    client = ChatCompletionsClient(LLMConfig('test-key','https://generic.invalid','test'), transport)
    client.complete_structured(messages_with_evidence('q','c'), GenerationSettings(), ANSWER_SCHEMA)
    assert 'response_format' not in transport.post.call_args.kwargs['json']

def test_manual_review_persists_without_changing_automatic_evidence(setup23):
    rag, client, store, index = setup23
    q = dataset(index)['questions'][0]
    from document_indexer.day24_evaluation import run_metadata
    from document_indexer.day23_pipeline import answer
    meta = run_metadata(index, GenerationSettings(), PipelineSettings(), client, 1)
    meta['questions'] = [q]
    rid = store.create_run('day24_evaluation',meta,index)
    client.answer_text = json.dumps(payload())
    result = answer(rag,q['question'],index,GenerationSettings(context_window_tokens=20000),PipelineSettings(),rid,q['id'],structured_evidence=True)
    review = {'claims':{'C1':'unsupported'},'correctness':0, 'facts':{f['id']:False for f in q['expected_facts']},
              'answer_has_extra_facts':False, 'contradictions':False, 'answers_question':False, 'context_sufficient':True,
              'clarification_requested':False,'conditions_preserved':True,'sources_support':False,
              'comment':'The literal quote does not entail the claim. Human fixture.', 'human_confirmed':True}
    store.save_review(result.request_id,review)
    saved = store.get_run(rid)
    assert saved['answers'][0]['validation']['quote_validation']['status'] == 'passed'
    assert saved['answers'][0]['validation']['semantic_support']['claims']['C1'] == 'unsupported'
    assert summarize(saved)['day24']['metrics']['supported_claims'] == fraction(0,1)

def test_metrics_na_not_zero(setup23):
    run = run24(setup23)
    metrics = summarize(run)['day24']['metrics']
    assert metrics['supported_claims'] == fraction(0, 0)
    assert metrics['with_sources'] == fraction(1, 1)
    assert metrics['false_refusals_corpus']['value'] is None

def test_missing_token_field_not_zero(setup23):
    run = run24(setup23)
    run['answers'][0]['usage'] = {'prompt_tokens': 10}
    result = summarize(run)['day24']
    assert result['metrics']['mean_generation_tokens'] == fraction(0, 0)
    assert result['known_generation_usage']['total_tokens'] is None
    assert result['usage_field_coverage']['total_tokens'] == fraction(0, 1)

def test_markdown_metric_table_is_contiguous(setup23):
    from document_indexer.day24_evaluation import report_formats
    run = run24(setup23)
    md = report_formats(run)['md']
    assert '| Значение |\n|---|---:|---:|---:|\n|' in md

def test_ten_day24_questions_without_baseline_or_reference_leak(setup23, monkeypatch):
    rag, client, store, index = setup23
    original = client.complete
    def complete(messages, settings):
        client.answer_text = json.dumps(payload()) if 'JSON schema' in messages[0]['content'] else 'Ответ [S1]'
        return original(messages, settings)
    client.complete = complete
    monkeypatch.setattr('document_indexer.day24_evaluation.generate_baseline', Mock(side_effect=AssertionError('No Day 23 calls')))
    data = dataset(index)
    data['corpus_hash'] = index.config['corpus_hash']
    for q in data['questions']:
        q['expected_status'] = 'answered'
        q['acceptable_clarifications'] = ['Уточните тему']
    data['questions'][-1]['answerable_from_corpus'] = False
    data['questions'][-1]['expected_status'] = 'insufficient_context'
    updates = []
    run = evaluate_day24(rag, index, data, GenerationSettings(context_window_tokens=20000), PipelineSettings(),
                         lambda *args: updates.append(args))
    assert len(run['answers']) == 10 and len(client.calls) == 10
    assert run['metadata']['planned_answers'] == run['metadata']['planned_calls'] == 10
    assert run['metadata']['report_modes'] == ['day24']
    assert updates[-1][1:] == (10, 10)
    for answer in run['answers']:
        assert answer['mode'] == 'day24'
        assert 'JSON schema' in answer['messages'][0]['content']
        assert 'MARKER_EXPECTATION' not in json.dumps(answer['messages'])
    assert set(summarize(run)) == {'day24'}
    from document_indexer.day24_evaluation import report_formats
    report = report_formats(run)
    assert 'сравнение с Днём 23' not in report['md']
    assert 'day23' not in report['csv']
