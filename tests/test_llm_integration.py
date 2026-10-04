import pytest
from document_indexer.llm_client import load_configuration, ChatCompletionsClient
from document_indexer.rag import RAGService
from document_indexer.experiment_storage import ExperimentStorage

@pytest.mark.integration
def test_real_provider(tmp_path):
    config, settings = load_configuration()
    if config.missing():
        pytest.skip("Set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL for real provider smoke test")
    config.validate()
    result = RAGService(ChatCompletionsClient(config), ExperimentStorage(tmp_path)).answer_without_rag("Что означает термин Android Activity? Ответь одним предложением.", settings)
    assert result.status == "completed", result.error
    assert result.answer and result.model == config.model
