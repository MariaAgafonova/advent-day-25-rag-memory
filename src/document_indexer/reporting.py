import csv
import io
import json
from pathlib import Path

def report_formats(report):
    json_text = json.dumps(report, ensure_ascii=False, indent=2)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["id", "question", "fixed_first_relevant_rank", "structural_first_relevant_rank"])
    strategies = report["strategies"]
    for a, b in zip(strategies["fixed"]["questions"], strategies["structural"]["questions"]):
        writer.writerow([a["id"], a["question"], a["first_relevant_rank"], b["first_relevant_rank"]])
    lines = ["# Сравнение chunking", "", f"Вопросов: {report['question_count']}. Модель загружена за {report['model_load_seconds']:.3f} с.", "",
             "| Стратегия | Hit@1 | Hit@5 | MRR@5 | Чанков | median, мс | p95, мс |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, r in strategies.items():
        m, t = r["metrics"], r["timing"]
        lines.append(f"| {key} | {m['Hit@1']:.3f} | {m['Hit@5']:.3f} | {m['MRR@5']:.3f} | {r['chunk_count']} | {t['median_seconds']*1000:.2f} | {t['p95_seconds']*1000:.2f} |")
    lines += ["", "Релевантность: чанк полностью содержит размеченный диапазон достаточного ответа.",
              "Время: после прогрева, кодирование запроса + ранжирование. Similarity не является вероятностью.",
              "", "## По вопросам", "", "| Вопрос | fixed | structural |", "|---|---:|---:|"]
    for a, b in zip(strategies["fixed"]["questions"], strategies["structural"]["questions"]):
        question = a["question"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {question} | {a['first_relevant_rank'] or '—'} | {b['first_relevant_rank'] or '—'} |")
    return {"json": json_text, "csv": "\ufeff" + stream.getvalue(), "md": "\n".join(lines) + "\n"}

def export_report(report, base):
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, content in report_formats(report).items():
        base.with_suffix("." + suffix).write_text(content, encoding="utf-8")
