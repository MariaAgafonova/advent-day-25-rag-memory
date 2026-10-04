"""Strict provider-facing schema. Source metadata is never accepted from an LLM."""
from dataclasses import dataclass, asdict
from typing import Literal
import json

@dataclass
class Claim:
    claim_id: str
    text: str
    citation_ids: list[str]

@dataclass
class Citation:
    citation_id: str
    context_id: str
    quote: str

@dataclass
class EvidenceAnswer:
    status: Literal['answered', 'partial', 'insufficient_context']
    answer: str
    claims: list[Claim]
    citations: list[Citation]
    clarification_question: str | None

    def to_dict(self):
        return asdict(self)

def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}

STRING = {'type': 'string'}
ANSWER_SCHEMA = obj({
    'status': {'type': 'string', 'enum': ['answered', 'partial', 'insufficient_context']},
    'answer': STRING,
    'claims': {'type': 'array', 'items': obj({'claim_id': STRING, 'text': STRING,
        'citation_ids': {'type': 'array', 'items': STRING}})},
    'citations': {'type': 'array', 'items': obj({'citation_id': STRING, 'context_id': STRING, 'quote': STRING})},
    'clarification_question': {'type': ['string', 'null']},
})

class SchemaError(ValueError):
    pass

def _check(value, schema, path='$'):
    kind = schema['type']
    if isinstance(kind, list):
        if value is None:
            return
        kind = 'string'
    expected = {'object': dict, 'array': list, 'string': str}[kind]
    if type(value) is not expected:
        raise SchemaError(path + ': неверный тип, ожидается ' + kind)
    if kind == 'object':
        if set(value) != set(schema['properties']):
            raise SchemaError(path + ': обязательные поля: ' + ', '.join(schema['properties']) + '; лишние поля запрещены')
        for k, child in schema['properties'].items():
            _check(value[k], child, path + '.' + k)
    elif kind == 'array':
        for i, child in enumerate(value):
            _check(child, schema['items'], f'{path}[{i}]')
    elif not value.strip():
        raise SchemaError(path + ': пустая строка')
    if 'enum' in schema and value not in schema['enum']:
        raise SchemaError(path + ': неизвестное значение')

def parse_answer(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SchemaError('Повторное JSON-поле: ' + key)
            result[key] = value
        return result
    try:
        data = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, TypeError) as exc:
        raise SchemaError('Невалидный JSON' if not isinstance(exc, SchemaError) else str(exc)) from None
    _check(data, ANSWER_SCHEMA)
    return EvidenceAnswer(data['status'], data['answer'], [Claim(**c) for c in data['claims']],
                          [Citation(**c) for c in data['citations']], data['clarification_question'])
