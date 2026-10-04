"""Curated questions and paragraph anchors, authored before any retrieval run.

Never imports search or chunkers. Does not adapt answers to chunk boundaries.
Run only on the downloaded Android corpus; manually review anchors on updates.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
from document_indexer.loaders import load_corpus
from document_indexer.evaluate import validate_questions

SPECS = [
    ("lifecycle.md", "Почему нельзя сохранять пользовательские данные и выполнять тяжёлые операции в onPause?", "`onPause` execution is very brief"),
    ("lifecycle.md", "Какие ресурсы нужно освобождать в onStop, когда Activity не видна?", "In the `onStop` method, release or adjust resources"),
    ("lifecycle.md", "Когда следует сохранять черновик в постоянное хранилище при остановке Activity?", "Also, use `onStop` to perform relatively CPU-intensive"),
    ("viewmodel.md", "Как ViewModel помогает избежать повторной загрузки данных при повороте экрана?", "ViewModel allows persistence through both"),
    ("viewmodel.md", "Что позволяет сохранить UI state после смерти процесса?", "[SavedStateHandle]"),
    ("viewmodel.md", "Кто обрабатывает бизнес-логику и события в слое UI?", "ViewModel is the right place to handle business logic"),
    ("coroutines.md", "Что происходит с корутинами viewModelScope, когда ViewModel уничтожается?", "Since this coroutine is started with `viewModelScope`"),
    ("coroutines.md", "Что означает main-safe и как withContext помогает не блокировать интерфейс?", "We consider a function *main-safe*"),
    ("coroutines.md", "Какой диспетчер корутин предназначен для операций ввода-вывода?", "* `Dispatchers.IO` indicates"),
    ("workmanager.md", "Как выполнять фоновые задачи, которые должны пережить перезапуск приложения и устройства?", "When you want to execute tasks"),
    ("workmanager.md", "Как ограничить запуск фоновой работы состоянием сети и зарядом батареи?", "Declaratively define the optimal conditions"),
    ("workmanager.md", "Как WorkManager повторяет неудавшуюся задачу с увеличением задержки?", "Sometimes work fails. WorkManager offers"),
    ("deep-links.md", "Какая категория нужна intent filter, чтобы ссылка из браузера открывала приложение?", "Include the [`BROWSABLE`]"),
    ("deep-links.md", "Зачем в intent filter для deep link добавляют категорию DEFAULT?", "Also include the [`DEFAULT`]"),
    ("deep-links.md", "Как различать Activity для ссылок с разными путями URI?", "You can add more attributes to further refine"),
    ("architecture.md", "Кто имеет право изменять данные согласно принципу единственного источника истины?", "When a new data type is defined"),
    ("architecture.md", "Что обычно служит источником истины в offline-first приложении?", "In an offline-first application"),
    ("architecture.md", "В каких направлениях передаются состояние и события при однонаправленном потоке данных?", "The [single source of truth principle]"),
]

def main():
    root = Path(__file__).resolve().parents[1]
    docs, errors = load_corpus(root / "data/raw")
    if errors:
        raise ValueError(errors)
    lookup = {d.source: d for d in docs}
    questions = []
    for i, (source, question, anchor) in enumerate(SPECS, 1):
        doc = lookup[source]
        if doc.text.count(anchor) != 1:
            raise ValueError(f"Проверьте уникальный anchor: {source}: {anchor}")
        start = doc.text.index(anchor)
        end = doc.text.find("\n\n", start)
        if end < 0:
            end = len(doc.text)
        questions.append({"id": f"q{i:02}", "question": question, "answers": [{"source": source,
                          "document_hash": doc.content_hash, "start_char": start, "end_char": end,
                          "answer_excerpt": doc.text[start:end]}]})
    data = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "method": "18 manually authored Russian questions, English answer paragraphs selected before retrieval; no chunk IDs",
            "questions": questions}
    validate_questions(data, lookup)
    target = root / "data/evaluation.json"
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validated {len(questions)} questions before retrieval")

if __name__ == "__main__":
    main()
