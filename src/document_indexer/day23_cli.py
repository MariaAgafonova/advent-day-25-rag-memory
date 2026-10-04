from pathlib import Path
from dataclasses import replace
import json
from .post_retrieval import PipelineSettings
from .reranker import DEFAULT_MODEL, DEFAULT_REVISION
from .experiment_storage import ExperimentStorage
from .llm_client import load_configuration, ChatCompletionsClient
from .storage import load_index
from .rag import RAGService
from .day23_evaluation import prepare_calibration, calibrate_filter, evaluate_day23, export_day23_report

def add_options(parser):
    parser.add_argument("--retrieval-k", type=int, default=15)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--filter-method", choices=["similarity_threshold", "cross_encoder"])
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--reranker-model", default=DEFAULT_MODEL)
    parser.add_argument("--reranker-revision", default=DEFAULT_REVISION)
    parser.add_argument("--reranker-threshold", type=float, help="Отдельный порог raw logit; без порога — только пересортировка")
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--calibration-id")

def register_commands(commands):
    calibration = commands.add_parser("calibrate-filter", help="Подготовить пары или рассчитать порог после ручной разметки")
    calibration.add_argument("--action", choices=["prepare", "fit"], required=True)
    calibration.add_argument("--pairs-id")
    calibration.add_argument("--dataset", default="data/day23_calibration_questions.json")
    evaluate = commands.add_parser("evaluate-day23", help="10 вопросов × 4 режима после калибровки")
    for parser in (calibration, evaluate):
        parser.add_argument("--index", default="output/structural.sqlite")
        parser.add_argument("--questions", default="data/rag_questions.json")
        parser.add_argument("--output", default="output/day23_experiments")
        add_options(parser)
    export = commands.add_parser("export-day23-report", help="Экспорт без вызова LLM")
    export.add_argument("--run-id", required=True)
    export.add_argument("--output", default="output/day23_experiments")

def execute(args, progress):
    output = args.output
    if args.command == "ask" and output == "output/rag_experiments":
        output = "output/day23_experiments"
    store = ExperimentStorage(output)
    if args.command == "export-day23-report":
        return export_day23_report(store, args.run_id)
    if args.command == "calibrate-filter" and args.action == "fit":
        if not args.pairs_id:
            raise ValueError("Для fit нужен --pairs-id")
        return calibrate_filter(store, args.pairs_id)
    config, settings = load_configuration()
    config.validate()
    if not args.index:
        raise ValueError("Для RAG нужен --index")
    index = load_index(args.index)
    mode = getattr(args, "mode", "baseline")
    method = args.filter_method or ("cross_encoder" if mode == "reranker" else "similarity_threshold")
    if method != "cross_encoder" and args.reranker_threshold is not None:
        raise ValueError("--reranker-threshold требует cross_encoder")
    if method == "cross_encoder" and args.threshold is not None and args.reranker_threshold is not None:
        raise ValueError("Укажите только один порог реранкера")
    pipeline = PipelineSettings(mode=mode, retrieval_k=args.retrieval_k,
        final_k=args.final_k, filter_method=method, similarity_threshold=args.threshold if method != "cross_encoder" else None,
        calibration_id=args.calibration_id, reranker_model=args.reranker_model if method == "cross_encoder" else None,
        reranker_revision=args.reranker_revision if method == "cross_encoder" else None,
        reranker_threshold=(args.reranker_threshold if args.reranker_threshold is not None else args.threshold) if method == "cross_encoder" else None,
        reranker_max_length=args.reranker_max_length)
    if args.calibration_id and args.threshold is None:
        pipeline = replace(pipeline, similarity_threshold=store.get_artifact(args.calibration_id)["threshold"])
    pipeline.validate()
    service = RAGService(ChatCompletionsClient(config), store)
    if args.command == "ask":
        run = service.run_query(args.question, args.mode, settings, index, progress, pipeline)
    else:
        test = json.loads(Path(args.questions).read_text(encoding="utf-8-sig"))
        if args.command == "calibrate-filter":
            dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8-sig"))
            return prepare_calibration(service, index, settings, pipeline, dataset, test, progress)
        run = evaluate_day23(service, index, test, settings, pipeline, progress)
    return export_day23_report(store, run["run_id"])
