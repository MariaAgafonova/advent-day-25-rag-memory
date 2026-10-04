"""Missing evidence must not be presented as unresolved user intent."""
import json
from document_indexer.chat import ChatSettings, Knowledge, refusal
from document_indexer.chat_storage import empty_state
from document_indexer.day25_mock import MockEmbedding, mock_index
from document_indexer.llm_client import LLMResponse
from document_indexer.rag_models import GenerationSettings


def test_insufficient_evidence_does_not_reask_explicit_topic():
    class Client:
        model, provider = 'mock', 'test'
        def complete_structured(self, messages, settings, schema):
            return LLMResponse(json.dumps(dict(status='insufficient_context', answer='Не знаю',
                claims=[], citations=[], clarification_question='Вас интересует Fragment или Activity?')),
                self.provider, self.model, 0, finish_reason='stop')
    knowledge = Knowledge(mock_index(), Client(), MockEmbedding())
    selected, diagnostics = knowledge.retrieve('fragment lifecycle')
    result = knowledge.answer('жизненный цикл Fragment', selected, diagnostics, empty_state('cid'), [],
                              GenerationSettings(context_window_tokens=20000), ChatSettings())
    assert result['status'] == 'insufficient_context'
    assert result['sources'] == []
    assert 'жизненный цикл Fragment' in result['answer']
    assert 'Добавьте документ' in result['answer']
    assert 'Activity' not in result['answer'] and '?' not in result['answer']


def test_empty_retrieval_requests_evidence_instead_of_rephrasing():
    result = refusal('жизненный цико fragment')
    assert 'жизненный цико fragment' in result['answer']
    assert 'обновите индекс' in result['answer']
    assert 'Уточните' not in result['answer'] and result['sources'] == []
