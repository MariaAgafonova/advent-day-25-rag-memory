"""Author-controlled evidence anchors, prepared without retrieval or generation."""
from pathlib import Path
from datetime import datetime, timezone
import json
from document_indexer.storage import load_index
from document_indexer.rag_evaluation import validate_questions

SPECS = [
    ("fact", "Какой механизм Android подходит для фоновой задачи, которая должна пережить перезапуск приложения и устройства?", True,
     ["Для такой надёжной фоновой работы подходит WorkManager", "Работа сохраняется через перезапуски приложения и устройства"],
     ["Обычной корутины достаточно для переживания перезагрузки"], [("workmanager.md", "When you want to execute tasks")]),
    ("fact", "Что позволяет сохранить UI state не только при повороте экрана, но и после смерти процесса?", True,
     ["SavedStateHandle предназначен для сохранения состояния через смерть процесса"],
     ["Любое поле обычного ViewModel автоматически сохраняется после смерти процесса"], [("viewmodel.md", "[SavedStateHandle]")]),
    ("fact", "Зачем в intent filter для deep link нужна категория BROWSABLE?", True,
     ["BROWSABLE позволяет открыть Activity через ссылку из браузера"],
     ["BROWSABLE не нужен для открытия ссылки из браузера"], [("deep-links.md", "Include the [`BROWSABLE`]")]),
    ("fact", "Почему нельзя выполнять сохранение данных и сетевые запросы в onPause?", True,
     ["onPause выполняется очень кратко", "Сохранение, сетевые запросы и транзакции могут не завершиться вовремя"],
     ["onPause гарантирует завершение любой транзакции"], [("lifecycle.md", "`onPause` execution is very brief")]),
    ("paraphrase", "Как организовать владение данными, чтобы другие компоненты не могли произвольно менять общее состояние?", True,
     ["Назначить единственный источник истины — владельца данных", "Выдавать неизменяемые данные, а изменения проводить через функции или события владельца"],
     ["Каждый компонент должен напрямую менять общие данные"], [("architecture.md", "When a new data type is defined")]),
    ("paraphrase", "Как сделать функцию сетевого входа безопасной для плавности интерфейса при вызове с главного потока?", True,
     ["Не блокировать обновления UI на главном потоке", "Перенести работу через withContext на другой поток"],
     ["Одна пометка suspend автоматически переносит код на другой поток"], [("coroutines.md", "We consider a function *main-safe*")]),
    ("synthesis", "Чем отличается срок жизни работы в viewModelScope от фоновой задачи WorkManager при уходе с экрана и перезапуске устройства?", True,
     ["viewModelScope отменяется при уничтожении ViewModel", "WorkManager сохраняет задачи через перезапуски приложения и устройства"],
     ["viewModelScope всегда переживает перезагрузку устройства"], [("coroutines.md", "Since this coroutine is started with `viewModelScope`"), ("workmanager.md", "When you want to execute tasks")]),
    ("synthesis", "Как сочетать сохранение состояния при повороте экрана и восстановление после смерти процесса: какую роль играют ViewModel и SavedStateHandle?", True,
     ["ViewModel сохраняет состояние и работу через изменения конфигурации", "Для сохранения UI state через смерть процесса используется SavedStateHandle"],
     ["ViewModel сам сохраняет любые объекты в постоянное хранилище"], [("viewmodel.md", "ViewModel allows persistence through both"), ("viewmodel.md", "[SavedStateHandle]")]),
    ("false_premise", "Почему обычный ViewModel автоматически сохраняет все свои поля после смерти процесса без SavedStateHandle?", True,
     ["Отвергнуть предпосылку об автоматическом сохранении всех полей", "Различить изменения конфигурации и смерть процесса; указать SavedStateHandle"],
     ["Подтвердить автоматическое сохранение всех полей обычного ViewModel через смерть процесса"], [("viewmodel.md", "ViewModel allows persistence through both"), ("viewmodel.md", "[SavedStateHandle]")]),
    ("insufficient", "Какой SLA доставки фоновых задач принят в моей команде и какое точное значение таймаута мы установили в production?", False,
     [], ["Придумать SLA или числовое значение production-таймаута", "Утверждать, что просмотрена вся база или настройки моей команды"], []),
]

def main():
    root = Path(__file__).resolve().parents[1]
    index = load_index(root / "output/structural.sqlite")
    questions = []
    for i, (category, question, answerable, facts, forbidden, anchors) in enumerate(SPECS, 1):
        sources = []
        for source, anchor in anchors:
            doc = index.documents[source]
            if doc.text.count(anchor) != 1:
                raise ValueError(f"Проверьте anchor: {source}: {anchor}")
            start = doc.text.index(anchor)
            end = doc.text.find("\n\n", start)
            if end < 0:
                end = len(doc.text)
            sources.append({"source": source, "document_hash": doc.content_hash, "start_char": start, "end_char": end, "evidence_excerpt": doc.text[start:end]})
        behavior = "Ответить по доказательствам, не добавляя неподтверждённых фактов"
        if category == "false_premise":
            behavior = "Не принять ложную предпосылку; объяснить различие по источникам"
        if not answerable:
            behavior = "Для RAG: сообщить о недостаточности предоставленного контекста, не выдумывать SLA. Без RAG: оценить ответ самостоятельно; отсутствие сведений в корпусе не означает, что модель не может знать ответ."
        questions.append({"id": f"q{i:02}", "question": question, "category": category, "answerable_from_corpus": answerable,
            "expected_behavior": behavior, "expected_facts": [{"id": f"f{j}", "description": text, "weight": 1} for j, text in enumerate(facts, 1)],
            "forbidden_claims": forbidden, "expected_sources": sources})
    data = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "corpus_hash": index.config["corpus_hash"],
            "method": "Manually authored questions and paragraph evidence; prepared before generation; independent of chunk IDs", "questions": questions}
    validate_questions(data, index)
    (root / "data/rag_questions.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Validated 10 questions: 4 facts, 2 paraphrases, 2 synthesis, 1 false premise, 1 insufficient")

if __name__ == "__main__":
    main()
