import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .storage import compatibility
from .embeddings import get_embedding_service
from .search import search
from .models import digest, canonical

def validate_questions(data, documents):
    questions = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(questions, list) or not questions:
        raise ValueError("Нужен непустой список questions")
    ids = set()
    for q in questions:
        if not isinstance(q.get("question"), str) or not q["question"].strip() or not q.get("answers"):
            raise ValueError("У каждого вопроса должны быть question и answers")
        if not q.get("id") or q["id"] in ids:
            raise ValueError("Нужны уникальные id вопросов")
        ids.add(q["id"])
        for answer in q["answers"]:
            doc = documents.get(answer.get("source"))
            if doc is None or doc.content_hash != answer.get("document_hash"):
                raise ValueError(f"Устаревшая разметка {q['id']}: источник или document_hash не совпадает")
            a, b = answer.get("start_char"), answer.get("end_char")
            if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(doc.text):
                raise ValueError(f"Неверный диапазон в {q['id']}")
            if doc.text[a:b] != answer.get("answer_excerpt") or not doc.text[a:b].strip():
                raise ValueError(f"answer_excerpt не соответствует диапазону в {q['id']}")
    return questions

def relevance(result, answers):
    return [a for a in answers if result["source"] == a["source"]
            and result["start_char"] <= a["start_char"] and result["end_char"] >= a["end_char"]]

def metrics(ranks):
    if not ranks:
        raise ValueError("Нет вопросов для оценки")
    n = len(ranks)
    return {"Hit@1": sum(r == 1 for r in ranks) / n,
            "Hit@5": sum(r is not None and r <= 5 for r in ranks) / n,
            "MRR@5": sum(1 / r if r is not None and r <= 5 else 0 for r in ranks) / n}

def evaluate(fixed, structural, data, repeats=3, service=None, progress=None):
    reasons = compatibility(fixed, structural)
    if reasons:
        raise ValueError("Сравнение заблокировано: " + "; ".join(reasons))
    if repeats < 2:
        raise ValueError("Для измерений нужны хотя бы два повторения")
    questions = validate_questions(data, fixed.documents)
    validate_questions(data, structural.documents)
    service = service or get_embedding_service(fixed.config["model_id"], fixed.config["revision"])
    # Warm up both paths, then alternate order to reduce order bias.
    search(fixed, questions[0]["question"], service=service)
    search(structural, questions[0]["question"], service=service)
    indexes = {"fixed": fixed, "structural": structural}
    reports = {key: {"ranks": [], "samples": [], "questions": []} for key in indexes}
    for i, q in enumerate(questions):
        outputs = {}
        for repeat in range(repeats):
            order = ("fixed", "structural") if (i + repeat) % 2 == 0 else ("structural", "fixed")
            for key in order:
                output = search(indexes[key], q["question"], 5, service=service)
                reports[key]["samples"].append(output["seconds"])
                outputs[key] = output
        for key in indexes:
            results = outputs[key]["results"]
            for result in results:
                result["matched_answers"] = relevance(result, q["answers"])
            first = next((r["rank"] for r in results if r["matched_answers"]), None)
            reports[key]["ranks"].append(first)
            reports[key]["questions"].append({**q, "first_relevant_rank": first, "results": results})
        if progress:
            progress("оценка", i + 1, len(questions))
    for key, report in reports.items():
        index = indexes[key]
        sizes = [c.token_count for c in index.chunks]
        report.update({"metrics": metrics(report.pop("ranks")), "timing": {
            "median_seconds": float(np.median(report["samples"])), "p95_seconds": float(np.percentile(report["samples"], 95)),
            "samples": len(report["samples"]), "repeats": repeats, "warmup": True,
            "includes": "query encoding + ranking; excludes model/index loading"},
            "chunk_count": len(sizes), "chunk_sizes": sizes,
            "size_distribution": {"min": min(sizes) if sizes else 0, "median": float(np.median(sizes)) if sizes else 0,
                                  "p95": float(np.percentile(sizes, 95)) if sizes else 0, "max": max(sizes) if sizes else 0},
            "config": index.config})
    return {"model_load_seconds": service.load_seconds, "question_count": len(questions),
            "evaluation_hash": digest(canonical(questions)), "strategies": reports}

def compare_files(fixed_path, structural_path, questions_path, repeats=3, progress=None):
    from .storage import load_index
    data = json.loads(Path(questions_path).read_text(encoding="utf-8-sig"))
    result = evaluate(load_index(fixed_path), load_index(structural_path), data, repeats, progress=progress)
    attach_file_statistics(result, {"fixed": fixed_path, "structural": structural_path})
    return result

def attach_file_statistics(result, paths):
    for key, path in paths.items():
        path = Path(path)
        report = result["strategies"][key]
        report["database_bytes"] = path.stat().st_size
        timing_path = path.with_suffix(".timing.json")
        timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        report["indexing_seconds"] = timing.get("total_seconds") if timing.get("build_id") == report["config"]["build_id"] else None
