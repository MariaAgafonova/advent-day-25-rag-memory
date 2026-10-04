from dataclasses import replace
from pathlib import Path
import json
from .day23_cli import add_options
from .post_retrieval import PipelineSettings
from .llm_client import load_configuration, ChatCompletionsClient
from .experiment_storage import ExperimentStorage
from .storage import load_index
from .rag import RAGService
from .day24_evaluation import evaluate_day24, export_day24_report

def register_commands(commands):
    evaluate = commands.add_parser('evaluate-day24', help='10 контрольных вопросов только в режиме Дня 24')
    evaluate.add_argument('--index', default='output/structural.sqlite')
    evaluate.add_argument('--questions', default='data/day24_questions.json')
    evaluate.add_argument('--output', default='output/day24_experiments')
    evaluate.add_argument('--mode', choices=['filter', 'rewrite_filter', 'reranker'], default='filter')
    add_options(evaluate)
    export = commands.add_parser('export-day24-report', help='JSON/CSV/Markdown без LLM')
    export.add_argument('--run-id', required=True)
    export.add_argument('--output', default='output/day24_experiments')

def execute(args, progress):
    output = args.output
    if output == 'output/rag_experiments':
        output = 'output/day24_experiments'
    store = ExperimentStorage(output)
    if args.command == 'export-day24-report':
        return export_day24_report(store, args.run_id)
    config, settings = load_configuration()
    config.validate()
    if args.mode == 'no-rag':
        raise ValueError('Structured evidence требует RAG')
    index = load_index(args.index or 'output/structural.sqlite')
    mode = 'filter' if args.mode == 'rag' else args.mode
    method = args.filter_method or ('cross_encoder' if mode == 'reranker' else 'similarity_threshold')
    if method != 'cross_encoder' and args.reranker_threshold is not None:
        raise ValueError('Порог реранкера требует cross_encoder')
    if method == 'cross_encoder' and args.threshold is not None:
        raise ValueError('Используйте --reranker-threshold для raw logit')
    threshold = args.threshold
    if args.calibration_id:
        threshold = store.get_artifact(args.calibration_id)['threshold']
    elif threshold is None and method == 'similarity_threshold' and mode in {'filter', 'rewrite_filter'}:
        threshold = .85
    pipeline = PipelineSettings(mode=mode, retrieval_k=args.retrieval_k, final_k=args.final_k,
        filter_method=method, similarity_threshold=threshold, calibration_id=args.calibration_id,
        reranker_model=args.reranker_model if method == 'cross_encoder' else None,
        reranker_revision=args.reranker_revision if method == 'cross_encoder' else None,
        reranker_threshold=args.reranker_threshold, reranker_max_length=args.reranker_max_length)
    service = RAGService(ChatCompletionsClient(config), store)
    if args.command == 'ask':
        run = service.answer_with_evidence(args.question, index, settings, pipeline, progress)
        result = export_day24_report(store, run['run_id'])
        result['answers'] = run['answers']
        return result
    dataset = json.loads(Path(args.questions).read_text(encoding='utf-8-sig'))
    run = evaluate_day24(service, index, dataset, settings, pipeline, progress)
    return export_day24_report(store, run['run_id'])
