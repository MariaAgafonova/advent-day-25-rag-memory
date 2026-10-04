"""Exact matching against included_text, not retrieved/full document text.

Offsets are zero-based Python Unicode character offsets, end exclusive.
No normalization is applied. First match is selected; repeated matches are flagged.
"""
from dataclasses import asdict
from .answer_schema import parse_answer, SchemaError

class EvidenceValidator:
    def validate(self, raw, items, index_version, documents):
        structural, quotes, sources = [], [], {}
        try:
            answer = parse_answer(raw)
        except SchemaError as exc:
            return {'valid': False, 'structured_answer': None,
                    'structural_validation': {'status': 'failed', 'errors': [str(exc)]},
                    'quote_validation': {'status': 'not_evaluated', 'citations': []},
                    'semantic_support': {'status': 'unreviewed'}, 'sources': [], 'errors': [str(exc)]}
        context = {i.citation_id: i for i in items}
        citation_ids = [c.citation_id for c in answer.citations]
        claim_ids = [c.claim_id for c in answer.claims]
        if len(set(citation_ids)) != len(citation_ids):
            structural.append('citation_id должны быть уникальны')
        if len(set(claim_ids)) != len(claim_ids):
            structural.append('claim_id должны быть уникальны')
        used = set()
        for claim in answer.claims:
            if not claim.citation_ids:
                structural.append(f'{claim.claim_id}: нет доказательств')
            if len(set(claim.citation_ids)) != len(claim.citation_ids):
                structural.append(f'{claim.claim_id}: повторные ссылки')
            for cid in claim.citation_ids:
                if cid not in citation_ids:
                    structural.append(f'{claim.claim_id}: неизвестный citation_id {cid}')
                used.add(cid)
        for cid in set(citation_ids) - used:
            structural.append(f'{cid}: неиспользуемая цитата')
        if answer.status in {'answered', 'partial'} and (not answer.claims or not answer.citations):
            structural.append('Содержательный ответ требует claims и citations')
        if answer.status == 'answered' and answer.clarification_question is not None:
            structural.append('answered: clarification_question должен быть null')
        if answer.status in {'partial', 'insufficient_context'} and not answer.clarification_question:
            structural.append('Требуется вопрос для уточнения')
        if answer.status == 'insufficient_context':
            if not answer.answer.casefold().startswith('не знаю'):
                structural.append('insufficient_context: ответ должен начинаться с «Не знаю»')
            if answer.claims or answer.citations:
                structural.append('Для неподтверждённого отказа используйте пустые claims и citations; подтверждённая часть — partial')
        for citation in answer.citations:
            row = {**asdict(citation), 'quote_start': None, 'quote_end': None,
                   'document_quote_start': None, 'document_quote_end': None,
                   'ambiguous': False, 'match_count': 0, 'validation_status': 'failed', 'validation_error': None}
            item = context.get(citation.context_id)
            if item is None:
                row['validation_error'] = 'Неизвестный context_id'
            else:
                doc = documents.get(item.source)
                matches, start = [], 0
                while True:
                    pos = item.included_text.find(citation.quote, start)
                    if pos < 0:
                        break
                    matches.append(pos)
                    start = pos + 1
                if not matches:
                    row['validation_error'] = 'Цитата не найдена дословно в переданном контексте'
                elif not doc or doc.content_hash != item.document_hash or doc.text[item.start_char:item.end_char] != item.included_text or not index_version:
                    row['validation_error'] = 'Источник не разрешается в сохранённые метаданные'
                else:
                    pos = matches[0]
                    row.update(quote_start=pos, quote_end=pos + len(citation.quote),
                               document_quote_start=item.start_char + pos,
                               document_quote_end=item.start_char + pos + len(citation.quote),
                               ambiguous=len(matches) > 1, match_count=len(matches), validation_status='passed')
                    if citation.citation_id in used:
                        # Current ContextBuilder emits a separate item for every chunk span.
                        ranges = item.provenance or [{'chunk_id': item.chunk_id, 'start_char': item.start_char,
                            'end_char': item.end_char, 'line_start': item.line_start, 'line_end': item.line_end,
                            'page_start': item.page_start, 'page_end': item.page_end}]
                        sources[item.citation_id] = {'source_id': item.citation_id, 'source': item.source,
                            'title': item.title, 'section': item.section, 'chunk_id': item.chunk_id,
                            'chunk_ids': list(dict.fromkeys(r['chunk_id'] for r in ranges)), 'ranges': ranges,
                            'line_start': item.line_start, 'line_end': item.line_end,
                            'page_start': item.page_start, 'page_end': item.page_end,
                            'document_hash': item.document_hash, 'index_version': index_version}
            quotes.append(row)
        errors = structural + [r['citation_id'] + ': ' + r['validation_error'] for r in quotes if r['validation_error']]
        return {'valid': not errors, 'structured_answer': answer.to_dict(),
                'structural_validation': {'status': 'failed' if structural else 'passed', 'errors': structural},
                'quote_validation': {'status': 'failed' if any(r['validation_error'] for r in quotes) else 'passed' if quotes else 'not_applicable', 'citations': quotes},
                'semantic_support': {'status': 'unreviewed'}, 'sources': list(sources.values()), 'errors': errors}
