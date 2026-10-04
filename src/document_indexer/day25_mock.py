"""Explicit TEST ONLY fixtures. Not a provider fallback and not a RAG quality test."""
from pathlib import Path
import json
import re
import numpy as np
from .chat_memory import apply_preparation
from .models import Document, Chunk, digest
from .normalize import line_map
from .storage import Index
from .llm_client import LLMResponse

FACTS = [
 ('workmanager', 'WorkManager supports persistent background work across app restarts and device reboots.'),
 ('coroutines', 'Kotlin coroutines support asynchronous work. Ordinary in-process coroutines do not persist across process death.'),
 ('viewmodel', 'ViewModel retains UI state across configuration changes such as screen rotation.'),
 ('savedstate', 'SavedStateHandle lets a ViewModel retain UI state across process death.'),
 ('links', 'The BROWSABLE category allows an Activity to be opened from a web browser through a deep link.'),
]


class MockEmbedding:
    model_id, revision, dimension = 'day25-test-only', 'a' * 40, 6

    def encode_query(self, query):
        q = query.lower()
        vec = np.zeros(6, dtype='<f4')
        if 'sla' in q or 'orion' in q: vec[5] = 1
        elif 'browsable' in q or 'deep link' in q: vec[4] = 1
        elif 'savedstatehandle' in q or 'смерт' in q or 'новым требованием' in q: vec[3] = 1
        elif 'viewmodel' in q or 'поворот' in q or 'состояни' in q: vec[2] = 1
        else: vec[0] = 1
        return vec


def mock_index():
    docs, chunks = {}, []
    for number, (name, fact) in enumerate(FACTS):
        source = 'TEST_ONLY/' + name + '.md'
        text = '# Test corpus: ' + name + '\n' + fact
        doc = Document(source, name, None, 'md', text, digest(text), line_map(text))
        docs[source] = doc
        chunks.append(Chunk('test-' + name, 'structural', source, None, name, [[name]], number,
            text, 0, len(text), 1, 2, None, None, len(text.split()), digest(text)))
    config = dict(build_id='test-only', corpus_hash=digest(''.join(d.text for d in docs.values())),
                  model_id=MockEmbedding.model_id, revision=MockEmbedding.revision, dimension=6, strategy='structural')
    return Index(config, docs, chunks, np.eye(6, dtype='<f4')[:5])


class MockPlanner:
    """Hand-written mock responses keyed to explicit fixture language, not expected results."""
    def prepare(self, state, history, message, settings):
        text, ops = message['content'], []
        low = text.lower()
        def put(field, key, value, action='set'):
            ops.append(dict(action=action, field=field, key=key, value=value, origin='user_explicit', evidence=text))
        intent, query = 'question', text
        if low.startswith('цель:'):
            put('goal', 'main', text[6:]); intent = 'task_update'
        elif low.startswith('уточнение: аудитория'):
            put('clarifications', 'audience', 'начинающие Android-разработчики'); intent = 'task_update'
        elif low.startswith('ограничение бюджета'):
            put('constraints', 'budget', 'без платных сервисов'); intent = 'task_update'
        elif low.startswith('техническое ограничение'):
            put('constraints', 'language', 'Kotlin'); intent = 'task_update'
        elif low.startswith('ограничение:'):
            put('constraints', 'survival', 'Достаточно переживать поворот экрана'); intent = 'task_update'
        elif low.startswith('термин:'):
            put('terms', 'draft', 'черновик — текст поля ввода')
            put('clarifications', 'audience', 'начинающие разработчики'); intent = 'task_update'
        elif low.startswith('меняю прежнее ограничение'):
            put('constraints', 'survival', 'черновик должен переживать смерть процесса'); intent = 'task_update'
        elif low.startswith('забудь только'):
            put('clarifications', 'audience', '', 'remove'); intent = 'task_update'
        elif 'какие мои требования' in low:
            intent = 'memory_recap'
        if 'второй вариант' in low or 'сравни его' in low:
            query = 'WorkManager и корутины: сохранение фоновой работы после перезапуска приложения и устройства'
        if any(t in low for t in ('новым требованием', 'продолжай:', 'итог выбора')):
            query = 'SavedStateHandle и ViewModel для черновика после смерти процесса'
        if 'сократи' in low:
            query = 'WorkManager: план фоновой синхронизации Kotlin без платных сервисов'
        data = dict(operations=ops, summary='Тестовое резюме прогресса, не источник фактов.',
                    rewritten_query=query, ambiguity='', intent=intent)
        return apply_preparation(state, data, message), data


class MockClient:
    model, provider = 'EXPLICIT-MOCK', 'local-test'
    def complete_structured(self, messages, settings, schema):
        envelope = next(m['content'] for m in messages if m['content'].startswith('BEGIN_UNTRUSTED_CONTEXT_JSON'))
        text = json.loads(envelope.split('BEGIN_UNTRUSTED_CONTEXT_JSON\n', 1)[1].split('\nEND_UNTRUSTED_CONTEXT_JSON', 1)[0])
        # Extract only current evidence blocks, not the memory or answer expectations.
        match = re.search(r'\[(S\d+)\]\nTitle:.*?\nText:\n([^\n]+)\n([^\n]+)', text, re.S)
        if not match:
            raise AssertionError('Mock fixture has no evidence')
        cid, _, quote = match.groups()
        data = dict(status='answered', answer=quote, claims=[dict(claim_id='c1', text=quote, citation_ids=['q1'])],
                    citations=[dict(citation_id='q1', context_id=cid, quote=quote)], clarification_question=None)
        return LLMResponse(json.dumps(data), self.provider, self.model, 0, finish_reason='stop')
