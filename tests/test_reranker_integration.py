import pytest
from document_indexer.reranker import get_reranker_service, DEFAULT_REVISION, RerankerError

@pytest.mark.integration
def test_real_multilingual_reranker_scores_and_input_limit():
    model = get_reranker_service()
    assert model.revision == DEFAULT_REVISION
    query = "Как выполнять надёжные фоновые задачи в Android?"
    scores = model.score(query, ["WorkManager schedules reliable background tasks that persist across app restarts and device reboots.",
                                "Bananas are yellow fruits grown in warm climates."])
    assert scores[0].score > scores[1].score
    long = model.score(query, ["WorkManager schedules tasks. " * 300])[0]
    assert long.truncated and long.input_tokens > long.used_tokens and long.used_tokens == 512
    with pytest.raises(RerankerError, match="вопрос"):
        model.score("Android " * 600, ["short chunk"])
