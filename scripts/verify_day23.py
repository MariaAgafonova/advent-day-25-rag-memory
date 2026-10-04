"""Explicit, bounded live smoke run. Never imported/executed by the UI."""
from pathlib import Path
from dataclasses import asdict, replace
import argparse
import hashlib
import json
import sys
from document_indexer.llm_client import load_configuration, ChatCompletionsClient
from document_indexer.storage import load_index
from document_indexer.rag import RAGService
from document_indexer.embeddings import get_embedding_service
from document_indexer.experiment_storage import ExperimentStorage
from document_indexer.post_retrieval import PipelineSettings, MODES
from document_indexer.query_rewrite import QueryRewriter
from document_indexer.day23_pipeline import answer
from document_indexer.day23_evaluation import prepare_calibration, export_day23_report, summarize_day23

def main():
    parser = argparse.ArgumentParser(description="Live smoke: up to 4 generations + 1 shared rewrite; optional 8 calibration rewrites")
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--threshold", type=float, required=True, help="Explicit uncalibrated smoke threshold")
    parser.add_argument("--prepare-calibration", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config, settings = load_configuration(root)
    config.validate()
    index_path = root / "output/structural.sqlite"
    before = hashlib.sha256(index_path.read_bytes()).hexdigest()
    index = load_index(index_path)
    embedding = get_embedding_service(index.config["model_id"], index.config["revision"])
    client = ChatCompletionsClient(config)
    storage = ExperimentStorage(root / "output/day23_experiments")
    service = RAGService(client, storage, embedding)
    pipeline = PipelineSettings(similarity_threshold=args.threshold)
    question = "Подскажи, как выполнить фоновую работу, которая переживёт перезапуск приложения?"
    run_id = storage.create_run("day23_live_smoke", {"planned_calls": 4, "planned_answers": 4,
        "settings": asdict(settings), "pipeline": pipeline.snapshot(), "rewrites": [],
        "note": "Uncalibrated smoke only; not the 10-question quality experiment"}, index)
    rewrite = QueryRewriter(client).rewrite(question, settings, embedding)
    storage.append_rewrite(run_id, asdict(rewrite))
    for mode in MODES:
        result = answer(service, question, index, settings, replace(pipeline, mode=mode), run_id, "smoke01",
                        rewrite=rewrite if mode in {"rewrite", "rewrite_filter"} else None)
        print(json.dumps({"mode": mode, "status": result.status, "counts": result.counts}, ensure_ascii=False), flush=True)
    storage.finish_run(run_id)
    run = export_day23_report(storage, run_id)
    # Threshold 1 is a rejection edge check, never a candidate calibrated threshold.
    empty = service.run_query(question, "filter", settings, index,
        pipeline=replace(pipeline, mode="filter", similarity_threshold=1.))
    export_day23_report(storage, empty["run_id"])
    reread = storage.get_run(run_id)
    assert reread == run
    verification = {"smoke_run_id": run_id, "empty_run_id": empty["run_id"], "summary": summarize_day23(run),
        "empty_status": empty["answers"][0]["status"], "empty_generation_skipped": empty["answers"][0]["generation_skipped"],
        "index_sha256_before": before, "index_sha256_after": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "main_experiment": "blocked: awaiting human calibration labels", "human_reviews": 0,
        "embedding_model": index.config["model_id"], "embedding_revision": index.config["revision"],
        "generation_model": client.model}
    if args.prepare_calibration:
        prepared = prepare_calibration(service, index, settings, pipeline,
            json.loads((root / "data/day23_calibration_questions.json").read_text(encoding="utf-8")),
            json.loads((root / "data/rag_questions.json").read_text(encoding="utf-8")),
            progress=lambda stage, done, total: print(f"{stage}: {done}/{total}", flush=True))
        verification["calibration_pairs_id"] = prepared["run_id"]
        verification["calibration_pairs"] = len(prepared["pairs"])
        verification["calibration_rewrite_calls"] = sum(r["logical_calls"] for r in prepared["rewrites"])
        verification["calibration_provider_attempts"] = sum(r["provider_attempts"] for r in prepared["rewrites"])
        verification["calibration_fallbacks"] = sum(r["status"] == "fallback" for r in prepared["rewrites"])
    (storage.directory / "live_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    assert before == verification["index_sha256_after"]
    assert run["status"] == "completed", "Inspect saved smoke errors"
    assert verification["empty_status"] == "no_relevant_context" and verification["empty_generation_skipped"]
    print(json.dumps(verification, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
