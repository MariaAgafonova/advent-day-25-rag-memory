"""Create the Day 24 rubric BEFORE generation; never derive expectations from output."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root / 'data/rag_questions.json').read_text(encoding='utf-8-sig'))
data['version'] = 2
data['method'] = 'Day 24 rubric frozen before generation; inherited evidence and expected facts unchanged'
for q in data['questions']:
    q['answerable'] = q['answerable_from_corpus']
    q['expected_status'] = 'answered' if q['answerable'] else 'insufficient_context'
    q['acceptable_clarifications'] = (['Уточнить тему или название документа',
        'Уточнить компонент или версию, если отсутствует в вопросе'] if q['answerable'] else
        ['Уточнить команду и предоставить документ с её SLA и настройками'])
(root / 'data/day24_questions.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
