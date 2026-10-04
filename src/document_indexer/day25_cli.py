from pathlib import Path
import sys
from .day25_evaluation import run


def register_commands(commands):
    parser = commands.add_parser('evaluate-day25', help='Два диалога по 12 сообщений; mock явно отделён от live')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--mock', action='store_true')
    mode.add_argument('--live', action='store_true')
    parser.add_argument('--index', default='output/structural.sqlite')
    parser.add_argument('--output')
    parser.add_argument('--scenario', choices=['A', 'B'])


def execute(args):
    return run(Path(__file__).resolve().parents[2], args.mock, args.index, args.output,
        lambda name, step, status, failures: print(f'{name}/{step}: {status}; failures={failures}', file=sys.stderr, flush=True), args.scenario)
