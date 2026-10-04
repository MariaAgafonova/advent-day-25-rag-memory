"""Calibration, explicitly triggered comparisons and reports; no LLM judge."""
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4
import csv
import io
import json
from .models import canonical, digest
from .rag import check_index
from .rag_evaluation import validate_questions, validate_review, contains
from .query_rewrite import QueryRewriter, PROMPT_VERSION as REWRITE_VERSION
from .post_retrieval import MODES
from .day23_pipeline import answer
from .search import search
from .embeddings import get_embedding_service

LABELS = ("relevant", "partial", "irrelevant")
TERMINAL = {"completed", "citation_error", "no_relevant_context", "no_results", "empty_context"}

def fingerprint(index, pipeline, client, settings):
    return {"index": check_index(index), "retrieval_k": pipeline.retrieval_k, "final_k": pipeline.final_k,
            "filter_method": pipeline.filter_method, "rewrite_model": client.model,
            "provider": client.provider, "rewrite_prompt_version": REWRITE_VERSION,
            "generation_settings": asdict(settings), "provider_options": getattr(client, "generation_options", {})}

def validate_split(calibration, test):
    questions = calibration.get("questions", [])
    if not calibration.get("version") or not 5 <= len(questions) <= 8:
        raise ValueError("Calibration: нужны версия и 5–8 вопросов")
    norm = lambda q: " ".join(q.strip().casefold().split())
    ids, texts = [q.get("id") for q in questions], [norm(q.get("question", "")) for q in questions]
    if len(set(ids)) != len(ids) or len(set(texts)) != len(texts) or not all(ids) or not all(texts):
        raise ValueError("Calibration: пустые или повторные вопросы")
    if set(ids) & {q["id"] for q in test["questions"]} or set(texts) & {norm(q["question"]) for q in test["questions"]}:
        raise ValueError("Calibration и test должны быть разделены")

def prepare_calibration(service, index, settings, pipeline, dataset, test, progress=None):
    if pipeline.filter_method != "similarity_threshold":
        raise ValueError("Эта калибровка предназначена только для cosine; порог реранкера задаётся отдельно")
    validate_split(dataset, test)
    validate_questions(test, index)
    replace(pipeline, mode="baseline").validate()
    embedding = service.embedding_service or get_embedding_service(index.config["model_id"], index.config["revision"])
    document = {"version": dataset["version"], "dataset": dataset, "dataset_hash": digest(canonical(dataset)),
                "test_hash": digest(canonical(test)), "fingerprint": fingerprint(index, pipeline, service.client, settings),
                "partial_policy": "negative", "pairs": [], "searches": [], "rewrites": [], "status": "preparing"}
    artifact_id = service.storage.save_artifact("calibration_pairs", document)
    for q in dataset["questions"]:
        if progress:
            progress("calibration: " + q["id"], len(document["rewrites"]), len(dataset["questions"]))
        rewrite = QueryRewriter(service.client).rewrite(q["question"], settings, embedding)
        document["rewrites"].append(asdict(rewrite))
        # Persist calls before retrieval so an interrupted preparation remains auditable.
        service.storage.save_artifact("calibration_pairs", document, artifact_id)
        pairs = {}
        for variant, query in (("original", q["question"]), ("rewrite", rewrite.rewritten_query)):
            try:
                results = search(index, query, pipeline.retrieval_k, service=embedding)["results"]
            except Exception:
                document.update(status="failed", error={"stage": "retrieval", "question_id": q["id"], "variant": variant})
                service.storage.save_artifact("calibration_pairs", document, artifact_id)
                raise ValueError("Ошибка поиска calibration; частичные данные и вызовы сохранены") from None
            search_id = q["id"] + ":" + variant
            document["searches"].append({"id": search_id, "question_id": q["id"], "variant": variant, "query": query})
            for r in results:
                pair = pairs.setdefault(r["chunk_id"], {"pair_id": q["id"] + ":" + r["chunk_id"],
                    "question_id": q["id"], "question": q["question"], "chunk_id": r["chunk_id"],
                    "source": r["source"], "text": r["text"], "start_char": r["start_char"], "end_char": r["end_char"],
                    "document_hash": index.documents[r["source"]].content_hash, "observations": [],
                    "label": None, "reviewed_by": None, "comment": ""})
                pair["observations"].append({"search_id": search_id, "variant": variant, "query": query,
                    "score": r["score"], "rank": r["rank"]})
        document["pairs"].extend(pairs.values())
        service.storage.save_artifact("calibration_pairs", document, artifact_id)
    document["status"] = "awaiting_human_labels"
    service.storage.save_artifact("calibration_pairs", document, artifact_id)
    return {"run_id": artifact_id, **document}

def label_pair(storage, artifact_id, pair_id, label, comment=""):
    if label not in LABELS:
        raise ValueError("Неизвестная метка")
    data = storage.get_artifact(artifact_id)
    if data.get("status") != "awaiting_human_labels":
        raise ValueError("Разметка доступна только после успешной подготовки всех пар")
    pair = next((p for p in data["pairs"] if p["pair_id"] == pair_id), None)
    if pair is None:
        raise ValueError("Пара не найдена")
    pair.update(label=label, reviewed_by="human", comment=str(comment)[:5000])
    storage.save_artifact("calibration_pairs", data, artifact_id)

def calibrate_filter(storage, artifact_id):
    data = storage.get_artifact(artifact_id)
    if data.get("status") != "awaiting_human_labels" or not data["pairs"] or any(p["label"] not in LABELS or p["reviewed_by"] != "human" for p in data["pairs"]):
        raise ValueError("Нужна полная ручная разметка calibration-пар")
    observations = [(p["label"] == "relevant", o) for p in data["pairs"] for o in p["observations"]]
    if not any(y for y, o in observations) or not any(not y for y, o in observations):
        raise ValueError("Для калибровки нужны положительные и отрицательные пары")
    thresholds = sorted({-1., 1., *(o["score"] for y, o in observations)})
    def metrics(threshold, variant=None):
        rows = [(y, o) for y, o in observations if variant is None or o["variant"] == variant]
        tp = sum(y and o["score"] >= threshold for y, o in rows)
        fp = sum(not y and o["score"] >= threshold for y, o in rows)
        fn = sum(y and o["score"] < threshold for y, o in rows)
        precision, recall = tp / (tp + fp) if tp + fp else 0., tp / (tp + fn) if tp + fn else 0.
        searches = [s["id"] for s in data["searches"] if variant is None or s["variant"] == variant]
        survivors = {o["search_id"] for y, o in rows if o["score"] >= threshold}
        return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
                "empty_context_fraction": sum(s not in survivors for s in searches) / len(searches), "tp": tp, "fp": fp, "fn": fn}
    table = [{"threshold": t, **metrics(t), "original": metrics(t, "original"), "rewrite": metrics(t, "rewrite")} for t in thresholds]
    best = max(table, key=lambda r: (r["f1"], r["recall"], -r["threshold"]))
    result = {"calibration_id": str(uuid4()), "status": "calibrated", "dataset_version": data["version"],
        "dataset_hash": data["dataset_hash"], "test_hash": data["test_hash"], "labels_hash": digest(canonical(data["pairs"])),
        "source_artifact": artifact_id, "fingerprint": data["fingerprint"], "threshold": best["threshold"],
        "filter_method": "similarity_threshold", "partial_policy": "negative", "table": table,
        "rule": "max F1; tie: max recall; tie: min threshold", "reviewed_by": "human",
        "limitations": "Малый calibration-набор; recall только среди найденных кандидатов, не всего корпуса",
        "labeled_pairs_snapshot": data["pairs"]}
    storage.save_artifact("calibration", result, result["calibration_id"])
    return result

def validate_calibration(calibration, index, pipeline, client, settings):
    if calibration.get("status") != "calibrated" or calibration.get("reviewed_by") != "human":
        raise ValueError("Порог не откалиброван человеком")
    if calibration["fingerprint"] != fingerprint(index, pipeline, client, settings):
        raise ValueError("Калибровка несовместима с индексом, моделями или настройками")
    if calibration["threshold"] != pipeline.similarity_threshold or calibration["calibration_id"] != pipeline.calibration_id:
        raise ValueError("Порог или идентификатор не соответствует калибровке")

def evaluate_day23(service, index, data, settings, pipeline, progress=None):
    questions = validate_questions(data, index)
    if not pipeline.calibration_id:
        raise ValueError("Основной эксперимент заблокирован: требуется ручная калибровка")
    calibration = service.storage.get_artifact(pipeline.calibration_id)
    validate_calibration(calibration, index, pipeline, service.client, settings)
    if calibration["test_hash"] != digest(canonical(data)):
        raise ValueError("Контрольный test-набор изменился после калибровки")
    run_id = service.storage.create_run("day23_evaluation", {"planned_calls": 40, "planned_answers": 40,
        "max_generation_calls": 40, "max_rewrite_calls": 10, "settings": asdict(settings),
        "pipeline": pipeline.snapshot(), "questions": questions, "question_set_hash": digest(canonical(data)),
        "calibration": calibration, "rewrites": [], "mode_order": list(MODES)}, index)
    embedding = service.embedding_service or get_embedding_service(index.config["model_id"], index.config["revision"])
    for q in questions:
        rewrite = QueryRewriter(service.client).rewrite(q["question"], settings, embedding)
        service.storage.append_rewrite(run_id, asdict(rewrite))
        for mode in MODES:
            answer(service, q["question"], index, settings, replace(pipeline, mode=mode), run_id, q["id"], progress,
                   rewrite if mode in {"rewrite", "rewrite_filter"} else None)
    service.storage.finish_run(run_id)
    return service.storage.get_run(run_id)

def evidence_checks(result, question):
    checks = []
    for evidence in question.get("expected_sources", []):
        candidates = [c for c in result.get("candidates", []) if contains(c, evidence)]
        included = [c for c in result["context_items"] if contains(c, evidence)]
        checks.append({"source": evidence["source"], "start_char": evidence["start_char"], "end_char": evidence["end_char"],
            "retrieved": bool(candidates), "passed_filter": any(c["passed_filter"] for c in candidates),
            "selected": any(c["selected"] for c in candidates), "included": bool(included),
            "cited": any(c["citation_id"] in result["citations"] for c in included),
            "false_rejection": bool(candidates) and not any(c["passed_filter"] for c in candidates)})
    return checks

def validate_day23_review(review, result, question):
    # Reuse Day 22's human rubric; successful context refusals are also reviewable.
    base = validate_review(dict(review), {**result, "mode": "rag", "status": "completed" if result["status"] in TERMINAL else result["status"]}, question)
    meaning = review.get("rewrite_meaning")
    search_effect = review.get("rewrite_search")
    if result["mode"] in {"rewrite", "rewrite_filter"} and result["rewrite"].get("status") != "fallback":
        if meaning not in {"preserved", "changed"} or search_effect not in {"better", "same", "worse"}:
            raise ValueError("Оцените смысл rewrite и его влияние на поиск")
    if not question["answerable_from_corpus"] and type(review.get("irrelevant_context")) is not bool:
        raise ValueError("Отметьте, остался ли нерелевантный контекст")
    return {**base, "reviewed_by": "human", "rewrite_meaning": meaning, "rewrite_search": search_effect,
            "irrelevant_context": review.get("irrelevant_context")}

def unique_rewrites(run):
    rows = list(run["metadata"].get("rewrites", []))
    rows.extend(a.get("rewrite", {}) for a in run["answers"])
    return list({r["rewrite_id"]: r for r in rows if r.get("rewrite_id")}.values())

def token_sum(rows):
    return {"known_total_tokens": sum((r.get("usage") or {}).get("total_tokens", 0) for r in rows),
            "unknown_usage_calls": sum(r.get("usage") is None for r in rows),
            "prompt_tokens": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows),
            "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)}

def summarize_day23(run):
    rewrites = unique_rewrites(run)
    generated = [a for a in run["answers"] if a.get("generation_calls")]
    result = {"answers": len(run["answers"]), "generation_calls": sum(a.get("generation_calls", 0) for a in generated),
              "rewrite_calls": sum(r.get("logical_calls", 0) for r in rewrites),
              "generation_attempts": sum(a.get("provider_attempts", 0) for a in generated),
              "rewrite_attempts": sum(r.get("provider_attempts", 0) for r in rewrites),
              "rewrite_latency_ms": sum(r.get("latency_ms", 0) for r in rewrites),
              "generation_usage": token_sum(generated), "rewrite_usage": token_sum(rewrites), "modes": {}}
    questions = {q["id"]: q for q in run["metadata"].get("questions", [])}
    for mode in dict.fromkeys((*MODES, *(a["mode"] for a in run["answers"]))):
        rows = [a for a in run["answers"] if a["mode"] == mode]
        reviewed = [a for a in rows if a.get("review", {}).get("status") == "reviewed"]
        mean = lambda values: sum(values) / len(values) if values else None
        quality = {k: mean([a["review"][k] for a in reviewed if a["review"].get(k) is not None]) for k in
                   ("correctness", "completeness", "unsupported_claims", "citation_support", "recognized_insufficiency")}
        checks = [evidence_checks(a, questions[a["question_id"]]) for a in rows if a["question_id"] in questions and questions[a["question_id"]]["answerable_from_corpus"] and a["status"] in TERMINAL]
        successful = [a for a in rows if a["status"] in TERMINAL]
        result["modes"][mode] = {**quality, "reviewed_answers": len(reviewed), "answers": len(rows),
            "sufficient_evidence_in_context": mean([all(c["included"] for c in group) for group in checks if group]),
            "false_rejections": sum(c["false_rejection"] for group in checks for c in group),
            "empty_context_fraction": mean([not a["context_items"] for a in successful]),
            "errors": sum(a["status"] == "failed" for a in rows), "citation_errors": sum(bool(a["invalid_citations"]) for a in rows),
            "latency_ms": {stage: mean([a.get("stage_latency_ms", {}).get(stage, 0) for a in rows]) for stage in
                           ("rewrite", "retrieval", "post_retrieval", "reranker", "context", "generation")},
            "rewrite_latency_attributed_ms": mean([a.get("rewrite", {}).get("latency_ms", 0) for a in rows]),
            "generation_usage": token_sum([a for a in rows if a.get("generation_calls")]),
            "rewrite_usage_attributed": token_sum([a["rewrite"] for a in rows if a.get("rewrite", {}).get("rewrite_id")])}
    return result

def report_formats(run, snapshot=None):
    stats = summarize_day23(run)
    modes = run["metadata"].get("report_modes", MODES)
    titles = {"baseline": "Без фильтра", "filter": "С фильтром", "reranker": "Реранкер", "rewrite": "Rewrite", "rewrite_filter": "Rewrite + Filter"}
    stats["modes"] = {mode: stats["modes"][mode] for mode in modes}
    questions = {q["id"]: q for q in run["metadata"].get("questions", [])}
    checks = {a["request_id"]: evidence_checks(a, questions[a["question_id"]]) for a in run["answers"] if a["question_id"] in questions}
    full = {**run, "summary": stats, "evidence_checks": checks, "source_snapshot": snapshot}
    lines = ["# День 23 — сравнение режимов", "", "Run: " + run["run_id"],
        "Пороги и метод: " + json.dumps({k: run["metadata"].get("pipeline", {}).get(k) for k in ("filter_method", "similarity_threshold", "reranker_threshold", "reranker_model", "reranker_revision")}, ensure_ascii=False),
        "Калибровка: " + str(run["metadata"].get("pipeline", {}).get("calibration_status", "uncalibrated")),
        f"Фактические логические вызовы: rewrite {stats['rewrite_calls']}, generation {stats['generation_calls']}; HTTP-попытки: rewrite {stats['rewrite_attempts']}, generation {stats['generation_attempts']}.",
        "", "| Метрика | " + " | ".join(titles[m] for m in modes) + " |", "|---|" + "---|" * len(modes)]
    metrics = [("Корректность", "correctness"), ("Полнота", "completeness"), ("Неподтверждённые утверждения", "unsupported_claims"),
               ("Поддержка цитирований", "citation_support"), ("Достаточные доказательства в контексте (диапазоны)", "sufficient_evidence_in_context"),
               ("False rejection (доказательства)", "false_rejections"), ("Доля пустого контекста", "empty_context_fraction"),
               ("Проверено ответов", "reviewed_answers"), ("Ошибки", "errors"), ("Ошибки ссылок", "citation_errors")]
    for label, key in metrics:
        lines.append("| " + label + " | " + " | ".join(str(stats["modes"][m][key]) if stats["modes"][m][key] is not None else "N/A" for m in modes) + " |")
    for stage in ("rewrite", "retrieval", "post_retrieval", "reranker", "context", "generation"):
        lines.append("| Время " + stage + ", мс | " + " | ".join(str(stats["modes"][m]["latency_ms"][stage]) for m in modes) + " |")
    for label, key in (("Generation usage", "generation_usage"), ("Rewrite usage (атрибуция)", "rewrite_usage_attributed")):
        lines.append("| " + label + " | " + " | ".join(str(stats["modes"][m][key]["known_total_tokens"]) + " (неизвестно: " + str(stats["modes"][m][key]["unknown_usage_calls"]) + ")" for m in modes) + " |")
    lines += ["", "Rewrite повторно используется: атрибуцию по режимам нельзя суммировать. Нулевое время повторного использования не означает бесплатный исходный rewrite.",
              "Recall калибровки измеряется только на найденных парах. Проверка диапазонов не заменяет смысловую оценку.",
              "Непроверенные ответы исключены из средних; по 10 вопросам статистически сильные выводы не делаются."]
    stream = io.StringIO(newline="")
    fields = ["question_id", "mode", "original_query", "rewritten_query", "status", "correctness", "completeness", "review_status", "stage_latency_ms", "generation_usage", "rewrite_usage", "error", "answer", "candidates"]
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for a in run["answers"]:
        review = a.get("review", {})
        row = {k: a.get(k) for k in fields}
        row.update(correctness=review.get("correctness"), completeness=review.get("completeness"), review_status=review.get("status", "unreviewed"), generation_usage=a["usage"], rewrite_usage=a.get("rewrite", {}).get("usage"))
        def cell(v):
            value = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else "N/A" if v is None else str(v)
            return "'" + value if value.startswith(("=", "+", "-", "@")) else value
        writer.writerow({k: cell(v) for k, v in row.items()})
        lines += ["", f"## {a['question_id']} · {a['mode']}", a["original_query"],
                  "Rewrite: " + a["rewritten_query"], a["answer"] or "Ответ не получен",
                  "Оценка: " + json.dumps(review, ensure_ascii=False), "Кандидаты и решения:",
                  "```json", json.dumps(a.get("candidates", []), ensure_ascii=False, indent=2), "```"]
    if run.get("comparison_review"):
        lines += ["", "## Ручное сравнение", json.dumps(run["comparison_review"], ensure_ascii=False)]
    if run["metadata"].get("questions"):
        lines += case_studies(run)
    lines += ["", "## Учёт вызовов и настроек", "```json", json.dumps(stats, ensure_ascii=False, indent=2), "```"]
    return {"json": json.dumps(full, ensure_ascii=False, indent=2), "csv": "\ufeff" + stream.getvalue(), "md": "\n\n".join(lines)}

def case_studies(run):
    lines = ["", "## Разбор результатов"]
    questions = {q["id"]: q for q in run["metadata"].get("questions", [])}
    categories = {"Улучшение": None, "Ухудшение или отсутствие улучшения": None, "Недостаточно информации": None}
    for qid, q in questions.items():
        rows = [a for a in run["answers"] if a["question_id"] == qid and a.get("review", {}).get("status") == "reviewed"]
        base = next((a for a in rows if a["mode"] == "baseline"), None)
        if not base:
            continue
        for a in rows:
            if a["mode"] == "baseline":
                continue
            bscore = (base["review"]["correctness"], base["review"].get("completeness") or 0)
            score = (a["review"]["correctness"], a["review"].get("completeness") or 0)
            category = "Недостаточно информации" if not q["answerable_from_corpus"] else "Улучшение" if score > bscore else "Ухудшение или отсутствие улучшения"
            if categories[category] is None:
                categories[category] = (q, base, a)
    for label, entry in categories.items():
        lines += ["### " + label]
        if entry is None:
            lines += ["Подтверждённых ручными оценками случаев нет. Это не доказательство превосходства режима."]
            continue
        q, base, improved = entry
        lines += [q["question"], "Rewrite: " + improved["rewritten_query"]]
        for a in (base, improved):
            lines += [a["mode"] + ": " + a["answer"], "Оценка и обоснование человека: " + json.dumps(a["review"], ensure_ascii=False),
                      "Этапы доказательств: " + json.dumps(evidence_checks(a, q), ensure_ascii=False),
                      "Отсеянные чанки: " + ", ".join(c["chunk_id"] for c in a["candidates"] if not c["passed_filter"])]
    return lines

def export_day23_report(storage, run_id):
    run = storage.get_run(run_id)
    snapshot = storage.snapshot(run["metadata"]["build_id"]) if run["metadata"].get("build_id") else None
    for ext, content in report_formats(run, snapshot).items():
        (storage.directory / (run_id + "." + ext)).write_text(content, encoding="utf-8")
    return run
