from pathlib import Path
from dataclasses import replace
import json
from .llm_client import load_configuration, ChatCompletionsClient
from .experiment_storage import ExperimentStorage
from .storage import load_index
from .rag import RAGService, check_index
from .rag_evaluation import evaluate_rag, export_run

def register_commands(commands):
    for name, help_text in [("ask", "Ответ с RAG или без RAG"), ("compare-answers", "Один вопрос, два режима"), ("evaluate-rag", "20 вызовов на 10 контрольных вопросах")]:
        parser = commands.add_parser(name, help=help_text)
        if name == "ask":
            parser.add_argument('--structured-evidence', action='store_true', help='День 24: точные цитаты и проверка')
            parser.add_argument("--mode", choices=["rag", "no-rag", "baseline", "rewrite", "filter", "rewrite_filter", "reranker"], required=True)
            from .day23_cli import add_options
            add_options(parser)
        if name != "evaluate-rag":
            parser.add_argument("--question", required=True)
        else:
            parser.add_argument("--questions", default="data/rag_questions.json")
        parser.add_argument("--index", required=name != "ask")
        parser.add_argument("--top-k", type=int, default=5)
        parser.add_argument("--output", default="output/rag_experiments")
        parser.add_argument("--json", action="store_true", help="JSON (также формат по умолчанию)")

def execute(args, progress):
    if args.command == 'ask' and args.structured_evidence:
        from .day24_cli import execute as execute_day24
        return execute_day24(args, progress)
    if args.command == "ask" and args.mode in {"baseline", "rewrite", "filter", "rewrite_filter", "reranker"}:
        from .day23_cli import execute as execute_day23
        return execute_day23(args, progress)
    config, settings = load_configuration()
    config.validate()  # Missing key: no provider calls and no phantom experiment.
    settings = replace(settings, top_k=args.top_k)
    settings.validate()
    needs_index = args.command != "ask" or args.mode == "rag"
    if needs_index and not args.index:
        raise ValueError("Для RAG нужен --index")
    index = load_index(args.index) if needs_index else None
    if index is not None:
        check_index(index)
    storage = ExperimentStorage(args.output)
    service = RAGService(ChatCompletionsClient(config), storage)
    if args.command == "evaluate-rag":
        data = json.loads(Path(args.questions).read_text(encoding="utf-8-sig"))
        progress("Запланировано 20 вызовов LLM", 0, 20)
        run = evaluate_rag(service, index, data, settings, progress)
    else:
        mode = args.mode if args.command == "ask" else "compare"
        run = service.run_query(args.question, mode, settings, index, progress)
    return export_run(storage, run["run_id"])
