from dataclasses import asdict
import csv
import io
import json
from collections import Counter
from .models import digest, canonical
from .rag import check_index

def validate_questions(data, index):
    questions = data.get("questions", [])
    if len(questions) != 10 or len({q.get("id") for q in questions}) != 10:
        raise ValueError("Нужны ровно 10 вопросов с уникальными id")
    for q in questions:
        if not q.get("id") or not q.get("question", "").strip() or not q.get("expected_behavior"):
            raise ValueError("Не хватает текста вопроса или ожидаемого поведения")
        if q.get("category") not in {"fact", "paraphrase", "synthesis", "false_premise", "insufficient"}:
            raise ValueError("Неизвестная категория вопроса")
        if type(q.get("answerable_from_corpus")) is not bool:
            raise ValueError("answerable_from_corpus должен быть boolean")
        facts = q.get("expected_facts", [])
        if len({f.get("id") for f in facts}) != len(facts) or any(not f.get("id") or not f.get("description") or not isinstance(f.get("weight"), (int, float)) or f["weight"] <= 0 for f in facts):
            raise ValueError("Неверные expected_facts")
        if q["answerable_from_corpus"] and not q.get("expected_sources"):
            raise ValueError("Для ответа по корпусу требуются доказательства")
        for evidence in q.get("expected_sources", []):
            doc = index.documents.get(evidence.get("source"))
            if not doc or doc.content_hash != evidence.get("document_hash"):
                raise ValueError(f"Устаревшая разметка: {q['id']}")
            a, b = evidence.get("start_char"), evidence.get("end_char")
            if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(doc.text) or doc.text[a:b] != evidence.get("evidence_excerpt"):
                raise ValueError(f"Неверный evidence excerpt: {q['id']}")
    return questions

def contains(item, evidence):
    return item["source"] == evidence["source"] and item["document_hash"] == evidence["document_hash"] and item["start_char"] <= evidence["start_char"] and item["end_char"] >= evidence["end_char"]

def source_checks(answer, question):
    if answer["mode"] == "no-rag":
        return None
    checks = []
    for evidence in question["expected_sources"]:
        included = [i for i in answer["context_items"] if contains(i, evidence)]
        checks.append({"source": evidence["source"], "start_char": evidence["start_char"], "end_char": evidence["end_char"],
            "retrieved": any(r["source"] == evidence["source"] and r["document_hash"] == evidence["document_hash"] for r in answer["retrieved_chunks"]),
            "evidence_retrieved": any(contains(r, evidence) for r in answer["retrieved_chunks"]),
            "included": bool(included), "cited": any(i["citation_id"] in answer["citations"] for i in included),
            "supports_claim": None})
    return checks

def validate_review(review, answer, question):
    if answer["status"] not in {"completed", "citation_error"}:
        raise ValueError("Неуспешный вызов нельзя оценивать как ответ")
    if review.get("correctness") not in {0, 1, 2}:
        raise ValueError("Корректность: 0, 1 или 2")
    facts = {f["id"] for f in question["expected_facts"]}
    marks = review.get("facts", {})
    if set(marks) != facts or any(type(v) is not bool for v in marks.values()):
        raise ValueError("Отметьте каждый expected_fact")
    for field in ("contradictions", "unsupported_claims"):
        if field == "unsupported_claims" and answer["mode"] == "no-rag":
            continue
        if type(review.get(field)) is not int or review[field] < 0:
            raise ValueError("Количество проблем должно быть неотрицательным целым")
    for field in ("recognized_insufficiency", "rejected_false_premise", "citation_support"):
        if review.get(field) is not None and type(review[field]) is not bool:
            raise ValueError("Отметка рубрики должна быть да/нет/N/A")
    applicable = {"citation_support": answer["mode"] == "rag" and bool(question["expected_sources"]), "recognized_insufficiency": question["category"] == "insufficient",
                  "rejected_false_premise": question["category"] == "false_premise"}
    for field, required in applicable.items():
        if required and type(review.get(field)) is not bool:
            raise ValueError("Заполните применимую отметку: " + field)
        if not required:
            review[field] = None
    weight = sum(f["weight"] for f in question["expected_facts"])
    completeness = sum(f["weight"] for f in question["expected_facts"] if marks[f["id"]]) / weight if weight else None
    return {"status": "reviewed", "correctness": review["correctness"], "facts": marks, "completeness": completeness,
            "contradictions": review["contradictions"], "unsupported_claims": review["unsupported_claims"] if answer["mode"] == "rag" else None,
            **{k: review.get(k) for k in applicable}, "comment": str(review.get("comment", ""))[:5000]}

def summarize(run):
    result = {"planned_calls": run["metadata"]["planned_calls"], "recorded_calls": len(run["answers"]),
              "provider_attempts": sum(a.get("provider_attempts", 0) for a in run["answers"]),
              "completed_calls": sum(a["status"] in {"completed", "citation_error"} for a in run["answers"]),
              "failed_calls": sum(a["status"] == "failed" for a in run["answers"]),
              "citation_error_calls": sum(a["status"] == "citation_error" for a in run["answers"]), "cost": "Стоимость не рассчитана", "modes": {}}
    reviewed_ids = set()
    for mode in ("no-rag", "rag"):
        reviewed = [a for a in run["answers"] if a["mode"] == mode and a.get("review", {}).get("status") == "reviewed"]
        reviewed_ids.update(a["question_id"] for a in reviewed)
        def mean(field):
            values = [a["review"][field] for a in reviewed if a["review"].get(field) is not None]
            return sum(values) / len(values) if values else None
        result["modes"][mode] = {"reviewed_answers": len(reviewed), "correctness": mean("correctness"), "completeness": mean("completeness"),
                                  "source_support": mean("citation_support") if mode == "rag" else None}
    result["questions_with_review"] = len(reviewed_ids)
    result["fully_reviewed_questions"] = sum(all(any(a["question_id"] == qid and a["mode"] == mode and a["review"]["status"] == "reviewed" for a in run["answers"]) for mode in ("no-rag", "rag")) for qid in reviewed_ids)
    return result

def evaluate_rag(service, index, data, settings, progress=None):
    check_index(index)
    questions = validate_questions(data, index)
    run_id = service.storage.create_run("evaluation", {"planned_calls": 20, "settings": asdict(settings),
        "questions": questions, "question_set_hash": digest(canonical(questions))}, index)
    count = 0
    for q in questions:
        for mode in ("no-rag", "rag"):
            if progress:
                progress("генерация: " + q["id"] + " / " + mode, count, 20)
            if mode == "no-rag":
                service.answer_without_rag(q["question"], settings, run_id=run_id, question_id=q["id"])
            else:
                service.answer_with_rag(q["question"], index, settings, run_id=run_id, question_id=q["id"])
            count += 1
            if progress:
                progress("выполнено вызовов", count, 20)
    service.storage.finish_run(run_id)
    return service.storage.get_run(run_id)

def report_formats(run, snapshot=None):
    summary = summarize(run)
    questions = {q["id"]: q for q in run["metadata"].get("questions", [])}
    full = {**run, "summary": summary, "source_checks": {a["request_id"]: source_checks(a, questions[a["question_id"]]) for a in run["answers"] if a["question_id"] in questions}}
    if snapshot:
        full["source_snapshot"] = snapshot
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["question_id", "question", "mode", "status", "review", "correctness", "completeness", "contradictions", "unsupported_claims", "citation_support", "total_latency_ms", "usage", "error"])
    lines = ["# Сравнение RAG / без RAG", "", f"Run: {run['run_id']}", f"Время запуска: {run['created_at']}",
             f"Успешных вызовов: {summary['completed_calls']}; ошибок: {summary['failed_calls']}; запланировано: {summary['planned_calls']}.",
             f"Проверено ответов: {sum(m['reviewed_answers'] for m in summary['modes'].values())}. Стоимость не рассчитана.", "",
             "Синтаксическая валидность ссылки не доказывает подтверждённость утверждения."]
    for a in run["answers"]:
        review = a["review"]
        def cell(value):
            value = "N/A" if value is None else str(value)
            return "'" + value if value.startswith(("=", "+", "-", "@")) else value
        writer.writerow([cell(v) for v in [a["question_id"], a["question"], a["mode"], a["status"], review["status"],
            review.get("correctness"), review.get("completeness"), review.get("contradictions"), review.get("unsupported_claims"), review.get("citation_support"),
            a["total_latency_ms"], json.dumps(a["usage"]), json.dumps(a["error"], ensure_ascii=False)]])
        lines += ["", f"## {a['question_id']} · {a['mode']}", a["question"], "", a["answer"] or "Ответ не получен.", "",
                  f"Статус: {a['status']}; ошибка: {a['error']}; ручная оценка: {json.dumps(review, ensure_ascii=False)}",
                  f"Ссылки: {a['citations']}; неизвестные: {a['invalid_citations']}"]
    lines += ["", "## Выводы", "Средние оценки включают только reviewed; без ручной проверки вывод о качестве не сделан.",
              "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```"]
    return {"json": json.dumps(full, ensure_ascii=False, indent=2), "csv": "\ufeff" + stream.getvalue(), "md": "\n\n".join(lines)}

def export_run(storage, run_id):
    run = storage.get_run(run_id)
    snapshot = storage.snapshot(run["metadata"]["build_id"]) if run["metadata"].get("build_id") else None
    for extension, text in report_formats(run, snapshot).items():
        (storage.directory / f"{run_id}.{extension}").write_text(text, encoding="utf-8")
    return run
