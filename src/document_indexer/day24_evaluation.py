"""Ten-question Day 24 evaluation; legacy paired reports remain readable."""
from copy import deepcopy
from dataclasses import asdict
from time import perf_counter
from uuid import uuid4
import csv
import io
import json
from .models import digest, canonical
from .rag import check_index
from .rag_evaluation import validate_questions, contains
from .prompts import messages_with_rag, PROMPT_VERSION
from .citations import validate_citations

LABELS = ('supported', 'partially_supported', 'unsupported', 'contradicted')

def run_metadata(index, settings, pipeline, client, count):
    pipeline.validate()
    return {'schema_version': 3, 'planned_calls': count, 'planned_answers': count,
            'settings': asdict(settings), 'pipeline': pipeline.snapshot(), 'model': client.model,
            'provider': client.provider, 'generation_options': getattr(client, 'generation_options', {}),
            'threshold_provenance': {'value': pipeline.reranker_threshold if pipeline.rerank_enabled else pipeline.similarity_threshold,
                'score_type': 'raw_logit' if pipeline.rerank_enabled else 'cosine_similarity',
                'model_id': pipeline.reranker_model if pipeline.rerank_enabled else index.config['model_id'],
                'revision': pipeline.reranker_revision if pipeline.rerank_enabled else index.config['revision'],
                'index_build_id': index.config['build_id'], 'corpus_hash': index.config['corpus_hash'],
                'calibration_id': pipeline.calibration_id,
                'origin': 'day23_calibration' if pipeline.calibration_id else 'day23_manual_threshold',
                'calibrated': bool(pipeline.calibration_id)},
            'semantic_evaluation': 'human_only', 'report_modes': ['day24']}

def generate_baseline(service, prepared, bundle, settings, timings, run_id, question_id):
    """Same retrieved/filtered/included evidence; original Day 23 generation prompt."""
    result = deepcopy(prepared)
    result.request_id, result.mode, result.prompt_version = str(uuid4()), 'day23', PROMPT_VERSION
    result.messages = messages_with_rag(result.question, bundle.text)
    started = perf_counter()
    if not bundle.items:
        result.generation_skipped = True
        result.status = 'no_results' if not result.retrieved_chunks else 'no_relevant_context' if not result.processed_chunks else 'empty_context'
        prefix = 'После фильтрации подходящие фрагменты не найдены. ' if result.status == 'no_relevant_context' else ''
        result.answer = prefix + 'Недостаточно предоставленного контекста для ответа на вопрос.'
    else:
        response = service.client.complete(result.messages, settings)
        result.answer, result.usage = response.text, response.usage
        result.api_request, result.provider_attempts = response.api_request, response.attempts
        result.error, result.finish_reason = response.error, response.finish_reason
        result.generation_calls = 1
        result.status = 'failed' if response.error else 'completed'
        if not response.error:
            result.citations, result.invalid_citations = validate_citations(result.answer, bundle.items)
            if result.invalid_citations:
                result.status = 'citation_error'
            if not result.citations:
                result.warnings.append('Нет валидных ссылок: требуется ручная проверка')
            if response.finish_reason == 'length':
                result.warnings.append('Ответ остановлен по лимиту токенов')
            cited = {i.chunk_id for i in bundle.items if i.citation_id in result.citations}
            result.counts['cited'] = len(cited)
            for c in result.candidates:
                c['cited'] = c['chunk_id'] in cited
    result.generation_latency_ms = (perf_counter() - started) * 1000
    result.stage_latency_ms = {**timings, 'generation': result.generation_latency_ms}
    result.retrieval_latency_ms = timings['retrieval']
    result.total_latency_ms = sum(result.stage_latency_ms.values())
    service.storage.save_answer(run_id, question_id, result)

def evaluate_day24(service, index, dataset, settings, pipeline, progress=None):
    check_index(index)
    if dataset.get('corpus_hash') != index.config['corpus_hash']:
        raise ValueError('Хеш контрольного корпуса не совпадает с индексом')
    questions = validate_questions(dataset, index)
    for q in questions:
        if q.get('expected_status') not in {'answered', 'partial', 'insufficient_context'} or not q.get('acceptable_clarifications'):
            raise ValueError('Сначала зафиксируйте рубрику Дня 24')
    if not any(not q['answerable_from_corpus'] for q in questions):
        raise ValueError('Нужен контрольный вопрос без ответа')
    metadata = {**run_metadata(index, settings, pipeline, service.client, len(questions)), 'questions': deepcopy(questions),
                'question_set_hash': digest(canonical(dataset)), 'shared_context': False,
                'timing_policy': 'Each Day 24 answer includes its own preparation and generation time.'}
    run_id = service.storage.create_run('day24_evaluation', metadata, index)
    from .day23_pipeline import answer
    for n, q in enumerate(questions):
        if progress:
            progress('День 24: ' + q['id'], n, len(questions))
        result = answer(service, q['question'], index, settings, pipeline, run_id, q['id'],
                        structured_evidence=True)
        if progress:
            progress('День 24: ' + q['id'] + ' завершён', n + 1, len(questions))
        if result.error and result.error.get('code') in {'configuration', 'http_401', 'http_403', 'temporary_connection'}:
            # Preserve the partial run; do not repeat a known unavailable provider across the dataset.
            break
    service.storage.finish_run(run_id)
    return service.storage.get_run(run_id)

def validate_day24_review(review, answer, question):
    if answer['status'] in {'failed', 'validation_failed'}:
        raise ValueError('Технический сбой не оценивается как качество ответа')
    kind = answer.get('answer_status') if answer['mode'] == 'day24' else review.get('answer_kind')
    if kind not in {'answered', 'partial', 'insufficient_context'}:
        raise ValueError('Укажите тип ответа')
    claims = review.get('claims', {})
    if not isinstance(claims, dict) or any(v not in LABELS for v in claims.values()):
        raise ValueError('Неверные оценки claims')
    if answer['mode'] == 'day24':
        expected = {c['claim_id'] for c in (answer.get('structured_answer') or {}).get('claims', [])}
        if set(claims) != expected:
            raise ValueError('Оцените каждое утверждение')
    elif kind != 'insufficient_context' and not claims:
        raise ValueError('Разметьте фактические утверждения старого ответа вручную')
    elif kind == 'insufficient_context' and claims:
        raise ValueError('Для отказа без фактов список утверждений пуст')
    if type(review.get('correctness')) is not int or review['correctness'] not in {0, 1, 2}:
        raise ValueError('Корректность: 0, 1, 2')
    facts = review.get('facts', {})
    if set(facts) != {f['id'] for f in question['expected_facts']} or any(type(v) is not bool for v in facts.values()):
        raise ValueError('Оцените каждый ожидаемый факт')
    for field in ('answer_has_extra_facts', 'contradictions', 'answers_question', 'context_sufficient',
                  'clarification_requested', 'conditions_preserved', 'sources_support'):
        if type(review.get(field)) is not bool:
            raise ValueError('Заполните поле: ' + field)
    if not str(review.get('comment', '')).strip():
        raise ValueError('Добавьте обоснование оценки')
    if review.get('human_confirmed') is not True:
        raise ValueError('Подтвердите ручную проверку')
    total = sum(f['weight'] for f in question['expected_facts'])
    completeness = sum(f['weight'] for f in question['expected_facts'] if facts[f['id']]) / total if total else None
    return {**review, 'status': 'reviewed', 'reviewed_by': 'human', 'answer_kind': kind, 'completeness': completeness}

def fraction(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'value': numerator / denominator if denominator else None}

def summarize(run):
    questions = {q['id']: q for q in run['metadata'].get('questions', [])}
    summary = {}
    for mode in run['metadata']['report_modes']:
        rows = [a for a in run['answers'] if a['mode'] == mode]
        reviewed = [a for a in rows if a.get('review', {}).get('status') == 'reviewed']
        def kind(a):
            return a.get('answer_status') if mode == 'day24' else a.get('review', {}).get('answer_kind')
        substantive = [a for a in rows if kind(a) in {'answered', 'partial'}]
        final_quotes = [q for a in rows if a.get('status') == 'completed' for q in a.get('validation', {}).get('quote_validation', {}).get('citations', [])]
        attempted_quotes = [q for a in rows for attempt in a.get('evidence_attempts', []) for q in attempt.get('validation', {}).get('quote_validation', {}).get('citations', [])]
        marks = [m for a in reviewed for m in a['review'].get('claims', {}).values()]
        no_answer = [a for a in reviewed if a['question_id'] in questions and not questions[a['question_id']]['answerable_from_corpus']]
        answerable = [a for a in reviewed if a['question_id'] in questions and questions[a['question_id']]['answerable_from_corpus']]
        sufficient = [a for a in reviewed if a['review']['context_sufficient']]
        refusals = [a for a in rows if kind(a) == 'insufficient_context']
        known_usage = [a for a in rows if a.get('generation_calls') and a.get('usage') is not None]
        known_total = [a for a in known_usage if 'total_tokens' in a['usage']]
        fact_n = sum(sum(f['weight'] for f in questions[a['question_id']]['expected_facts'] if a['review']['facts'].get(f['id'])) for a in reviewed if a['question_id'] in questions)
        fact_d = sum(sum(f['weight'] for f in questions[a['question_id']]['expected_facts']) for a in reviewed if a['question_id'] in questions)
        metric = {
            'review_coverage': fraction(len(reviewed), len(rows)),
            'classified_answers': fraction(sum(kind(a) is not None for a in rows), len(rows)),
            'with_sources': fraction(sum(bool(a.get('resolved_sources') if mode == 'day24' else a.get('citations')) for a in substantive), len(substantive)),
            'with_exact_quotes': fraction(sum(bool((a.get('structured_answer') or {}).get('citations')) for a in substantive), len(substantive)) if mode == 'day24' else fraction(0, 0),
            'valid_quotes': fraction(sum(q['validation_status'] == 'passed' for q in final_quotes), len(final_quotes)),
            'valid_quotes_all_attempts': fraction(sum(q['validation_status'] == 'passed' for q in attempted_quotes), len(attempted_quotes)),
            'supported_claims': fraction(marks.count('supported'), len(marks)),
            'correctness': fraction(sum(a['review']['correctness'] for a in reviewed), 2 * len(reviewed)),
            'completeness': fraction(fact_n, fact_d),
            'correct_refusals': fraction(sum(kind(a) == 'insufficient_context' and a['review']['correctness'] == 2 for a in no_answer), len(no_answer)),
            'false_refusals_corpus': fraction(sum(kind(a) == 'insufficient_context' for a in answerable), len(answerable)),
            'false_refusals_context': fraction(sum(kind(a) == 'insufficient_context' for a in sufficient), len(sufficient)),
            'refusals_with_clarification': fraction(sum(bool((a.get('structured_answer') or {}).get('clarification_question')) if mode == 'day24' else a['review']['clarification_requested'] for a in refusals), len(refusals)),
            'initial_validation_errors': fraction(sum(bool(a.get('evidence_attempts')) and a['evidence_attempts'][0].get('validation', {}).get('valid') is False for a in rows), sum(bool(a.get('evidence_attempts')) and 'validation' in a['evidence_attempts'][0] for a in rows)),
            'final_validation_failures': fraction(sum(a['status'] == 'validation_failed' for a in rows), len(rows)),
            'technical_errors': fraction(sum(a['status'] == 'failed' for a in rows), len(rows)),
            'repair_calls': fraction(sum(max(0, a.get('generation_calls', 0) - 1) for a in rows), len(rows)),
            'mean_latency_ms': fraction(sum(a['total_latency_ms'] for a in rows), len(rows)),
            'usage_coverage': fraction(len(known_usage), sum(bool(a.get('generation_calls')) for a in rows)),
            'mean_generation_tokens': fraction(sum(a['usage']['total_tokens'] for a in known_total), len(known_total)),
        }
        summary[mode] = {'answers': len(rows), 'substantive_answers': len(substantive),
            'claim_labels': {label: marks.count(label) for label in LABELS}, 'metrics': metric,
            'generation_calls': sum(a.get('generation_calls', 0) for a in rows),
            'provider_attempts': sum(a.get('provider_attempts', 0) for a in rows),
            'known_generation_usage': {k: sum(a['usage'][k] for a in known_usage if k in a['usage']) if any(k in a['usage'] for a in known_usage) else None for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
            'usage_field_coverage': {k: fraction(sum(k in a['usage'] for a in known_usage), sum(bool(a.get('generation_calls')) for a in rows)) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
            'rewrite_usage': [a.get('rewrite', {}).get('usage') for a in rows if a.get('rewrite', {}).get('logical_calls')]}
    return summary

def evidence_checks(answer, question):
    rows = []
    valid_context = set(answer.get('citations', []))
    for e in question['expected_sources']:
        quoted = [q for q in answer.get('validation', {}).get('quote_validation', {}).get('citations', [])
                  if q['validation_status'] == 'passed' and q['context_id'] in valid_context]
        items = {i['citation_id']: i for i in answer['context_items']}
        rows.append({'source': e['source'], 'start_char': e['start_char'], 'end_char': e['end_char'],
            'retrieved': any(contains(c, e) for c in answer['retrieved_chunks']),
            'included': any(contains(i, e) for i in answer['context_items']),
            'cited_context': any(contains(i, e) and i['citation_id'] in valid_context for i in answer['context_items']),
            'quote_overlaps_expected_range': any(items[q['context_id']]['source'] == e['source'] and
                q['document_quote_start'] < e['end_char'] and q['document_quote_end'] > e['start_char'] for q in quoted),
            'semantic_support': None})
    return rows

def report_formats(run, snapshot=None):
    stats = summarize(run)
    questions = {q['id']: q for q in run['metadata'].get('questions', [])}
    full = {**run, 'summary': stats, 'source_snapshot': snapshot,
        'expected_evidence_checks': {a['request_id']: evidence_checks(a, questions[a['question_id']])
            for a in run['answers'] if a['question_id'] in questions}}
    out = io.StringIO(newline='')
    writer = csv.writer(out)
    writer.writerow(['run_id', 'mode', 'metric', 'numerator', 'denominator', 'value'])
    paired = 'day23' in run['metadata']['report_modes']
    lines = ['# День 24: сравнение с Днём 23' if paired else '# День 24: проверка контрольных вопросов', '', 'Run: ' + run['run_id'],
        'Точность цитат не доказывает смысловую поддержку. Неоценённая семантика — N/A.',
        'Порог и происхождение: `' + json.dumps(run['metadata']['threshold_provenance'], ensure_ascii=False) + '`',
        'Время подготовки общего контекста учитывается в обеих ветках; генерация измеряется отдельно.' if paired else
        'Проверяется только режим Дня 24. Время включает подготовку и генерацию каждого ответа.', '']
    if paired:
        lines.append('Отсутствие нового JSON-формата у Дня 23 не считается ошибкой.')
    for mode, stat in stats.items():
        lines += ['## ' + mode, '| Метрика | Числитель | Знаменатель | Значение |', '|---|---:|---:|---:|']
        for name, m in stat['metrics'].items():
            val = 'N/A' if m['value'] is None else f"{m['value']:.4f}"
            writer.writerow([run['run_id'], mode, name, m['numerator'], m['denominator'], val])
            lines.append(f"| {name} | {m['numerator']} | {m['denominator']} | {val} |")
    for a in run['answers']:
        lines += ['', f"## {a['question_id']} · {a['mode']}", a['question'],
                  f"Статус: {a.get('answer_status') or a['status']}; причина: {a.get('refusal_reason') or '—'}", a['answer']]
        structured = a.get('structured_answer') or {}
        for claim in structured.get('claims', []):
            lines.append(f"- {claim['claim_id']}: {claim['text']} [{', '.join(claim['citation_ids'])}]")
        sources = {s['source_id']: s for s in a.get('resolved_sources', [])}
        for q in structured.get('citations', []):
            s = sources[q['context_id']]
            lines += [f"### {q['citation_id']} · {s['title']} · {s['source']}",
                      f"Раздел: {s['section']}; строки {s['line_start']}–{s['line_end']}; чанки: {s['chunk_ids']}",
                      '\n'.join('> ' + line for line in q['quote'].split('\n'))]
        if structured.get('clarification_question') or a.get('failure_clarification'):
            lines.append(structured.get('clarification_question') or a['failure_clarification'])
        lines += ['Ручная оценка: ' + json.dumps(a.get('review', {'status': 'unreviewed'}), ensure_ascii=False)]
    lines += ['', '## Выводы и ограничения',
              'Смысловая оценка выполняется человеком через интерфейс. Пока она не завершена, выводы о корректности, полноте, подтверждённости и правильности отказов не делаются.',
              'Порог 0.85 унаследован из Дня 23 и не откалиброван. 10 вопросов не дают статистической гарантии.',
              'Полная трассировка, исходные выводы, ошибки и обе попытки сохранены в JSON.']
    return {'json': json.dumps(full, ensure_ascii=False, indent=2), 'csv': '\ufeff' + out.getvalue(), 'md': '\n'.join(lines)}

def export_day24_report(storage, run_id):
    run = storage.get_run(run_id)
    snapshot = storage.snapshot(run['metadata']['build_id'])
    paths = {}
    # Unique export folder also preserves successive versions of manual reviews.
    folder = storage.directory / 'exports' / (run_id + '-' + uuid4().hex[:8])
    folder.mkdir(parents=True)
    for ext, content in report_formats(run, snapshot).items():
        path = folder / ('report.' + ext)
        path.write_text(content, encoding='utf-8')
        paths[ext] = str(path)
    return {'run_id': run_id, 'status': run['status'], 'summary': summarize(run), 'paths': paths}
