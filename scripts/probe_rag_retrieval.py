"""Real E5 retrieval/context diagnostics for three scenarios. Does NOT call any LLM."""
from pathlib import Path
from dataclasses import asdict
import json
from document_indexer.storage import load_index
from document_indexer.rag import check_index
from document_indexer.search import search
from document_indexer.rag_models import RetrievedChunk, GenerationSettings
from document_indexer.context_builder import TokenCounter, ContextBuilder
from document_indexer.rag_evaluation import validate_questions

def main():
    root = Path(__file__).resolve().parents[1]
    index = load_index(root / "output/structural.sqlite")
    data = json.loads((root / "data/rag_questions.json").read_text(encoding="utf-8"))
    questions = validate_questions(data, index)
    report = {"index": check_index(index, root / "data/raw"), "llm_calls": 0, "status": "retrieval_only", "scenarios": []}
    for q in (questions[0], questions[6], questions[9]):
        output = search(index, q["question"], 5)
        chunks = [RetrievedChunk(r["chunk_id"], r["source"], r["title"], r["sections"], r["line_start"], r["line_end"],
             r["page_start"], r["page_end"], r["text"], r["score"], r["rank"], r["start_char"], r["end_char"],
             index.documents[r["source"]].content_hash, r["source_url"]) for r in output["results"]]
        bundle = ContextBuilder(TokenCounter("unconfigured-provider")).build(chunks, q["question"], GenerationSettings(), index.documents)
        report["scenarios"].append({"id": q["id"], "question": q["question"], "retrieval_seconds": output["seconds"],
            "retrieved": [asdict(c) for c in chunks], "included": [asdict(i) for i in bundle.items], "context_tokens": bundle.token_count,
            "token_count_method": bundle.token_count_method})
        print(q["id"], "retrieved", len(chunks), "included", len(bundle.items), "seconds", round(output["seconds"], 3), flush=True)
    path = root / "output/rag_retrieval_probe.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved retrieval-only report; no generated answers or quality scores")

if __name__ == "__main__":
    main()
