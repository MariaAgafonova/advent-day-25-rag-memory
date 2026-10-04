"""Explicit real-model comparison; does not run on app startup."""
import argparse
from dataclasses import asdict
from pathlib import Path
from hashlib import sha256
import json
import sys
from document_indexer.reranker import get_reranker_service, DEFAULT_MODEL, DEFAULT_REVISION
from document_indexer.post_retrieval import PipelineSettings
from document_indexer.embeddings import get_embedding_service
from document_indexer.storage import load_index
from document_indexer.llm_client import ChatCompletionsClient, load_configuration
from document_indexer.rag import RAGService
from document_indexer.experiment_storage import ExperimentStorage
from document_indexer.day23_pipeline import compare
from document_indexer.day23_evaluation import export_day23_report, summarize_day23

def main():
    parser = argparse.ArgumentParser(description="CPU reranker + 3 comparisons, up to 6 answer generations")
    parser.add_argument("--live", action="store_true", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config, generation = load_configuration(root)
    config.validate()
    path = root / "output/structural.sqlite"
    before = sha256(path.read_bytes()).hexdigest()
    index = load_index(path)
    embeddings = get_embedding_service(index.config["model_id"], index.config["revision"])
    reranker = get_reranker_service()
    store = ExperimentStorage(root / "output/day23_experiments")
    service = RAGService(ChatCompletionsClient(config), store, embeddings, reranker_service=reranker)
    pipeline = PipelineSettings(mode="reranker", filter_method="cross_encoder", reranker_model=DEFAULT_MODEL,
        reranker_revision=DEFAULT_REVISION, reranker_threshold=None)
    questions = json.loads((root / "data/rag_questions.json").read_text(encoding="utf-8"))["questions"]
    verification = {"model": DEFAULT_MODEL, "revision": DEFAULT_REVISION, "device": "cpu",
        "score_type": "raw_logit", "threshold": None, "model_load_ms": reranker.load_latency_ms, "comparisons": []}
    for question in questions:
        if question["id"] not in {"q01", "q02", "q10"}:
            continue
        run = compare(service, question["question"], index, generation, pipeline)
        export_day23_report(store, run["run_id"])
        baseline, ranked = run["answers"]
        assert [c["chunk_id"] for c in baseline["retrieved_chunks"]] == [c["chunk_id"] for c in ranked["retrieved_chunks"]]
        row = {"question_id": question["id"], "run_id": run["run_id"], "status": run["status"],
            "changed_positions": sum(c["reranker_rank"] != c["retrieval_rank"] for c in ranked["candidates"]),
            "truncated_inputs": sum(c["reranker_truncated"] for c in ranked["candidates"]),
            "reranker_ms": ranked["stage_latency_ms"]["reranker"], "counts": {a["mode"]: a["counts"] for a in run["answers"]},
            "summary": summarize_day23(run)}
        verification["comparisons"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "summary"}, ensure_ascii=False), flush=True)
    verification["generation_calls"] = sum(r["summary"]["generation_calls"] for r in verification["comparisons"])
    verification["provider_attempts"] = sum(r["summary"]["generation_attempts"] for r in verification["comparisons"])
    verification["index_unchanged"] = before == sha256(path.read_bytes()).hexdigest()
    (store.directory / "reranker_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    assert verification["index_unchanged"] and all(r["status"] == "completed" for r in verification["comparisons"])
    print(f"Complete: {verification['generation_calls']} generations; {verification['provider_attempts']} HTTP attempts")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
