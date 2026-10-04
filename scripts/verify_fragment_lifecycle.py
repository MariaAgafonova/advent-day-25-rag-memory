"""Verify both a fresh dialogue and a private copy of the reported dialogue."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sqlite3
import argparse
from document_indexer.chat import build_chat
from document_indexer.chat_storage import ChatStore

parser = argparse.ArgumentParser(description='Live regression: Fragment lifecycle queries and optional copied conversation')
parser.add_argument('--live', action='store_true', required=True)
parser.add_argument('--conversation-id', help='Continue only a COPY of this existing conversation')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
output = root / 'output/fragment_lifecycle_checks' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
output.mkdir(parents=True, exist_ok=False)
records = []


def run(service, cid, query, label):
    result = service.send_message(cid, label, query)
    checks = dict(answered=result['status'] in ('answered', 'partial'),
        fragment_source=any(s['source'] == 'fragment-lifecycle.md' for s in result['sources']),
        no_repeated_choice='Fragment или Activity?' not in result['answer'] and 'фрагмента (Fragment) или' not in result['answer'],
        exact_quotes=result['diagnostics'].get('validation', {}).get('quote_validation', {}).get('status') == 'passed',
        correct_query='fragment' in result['diagnostics'].get('rewritten_query', '').lower())
    records.append(dict(label=label, query=query, result=result, checks=checks, passed=all(checks.values())))
    (output/'results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(label, result['status'], checks, flush=True)
    print(result['answer'], flush=True)


store = ChatStore(output / 'fresh.sqlite')
service = build_chat(root, store)
cid = store.create('Fragment lifecycle regression')
for number, query in enumerate(('жизненный цикл фрагмента', 'жизненный цикл фрагмента',
                                'жизненный цико fragment', 'жизненный цикл Fragment'), 1):
    run(service, cid, query, 'fresh-' + str(number))

if args.conversation_id:
    original = root / 'output/chat.sqlite'
    with sqlite3.connect('file:' + str(original) + '?mode=ro', uri=True) as source:
        with sqlite3.connect(output / 'continuation.sqlite') as destination:
            source.backup(destination)
    store = ChatStore(output / 'continuation.sqlite')
    service = build_chat(root, store)
    run(service, args.conversation_id, 'жизненный цикл Fragment', 'existing-dialogue-copy')
summary = dict(passed=sum(r['passed'] for r in records), total=len(records), output=str(output))
(output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False), flush=True)
raise SystemExit(0 if all(r['passed'] for r in records) else 1)
