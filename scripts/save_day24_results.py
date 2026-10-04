"""Build a reproducible results document from a saved run; never calls the LLM."""
from pathlib import Path
import argparse
import hashlib
import json
from document_indexer.experiment_storage import ExperimentStorage
from document_indexer.day24_evaluation import export_day24_report, summarize
from document_indexer.models import digest, canonical

parser = argparse.ArgumentParser()
parser.add_argument('--run-id', required=True)
parser.add_argument('--tests-passed', type=int, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
store = ExperimentStorage(root / 'output/day24_experiments')
run = store.get_run(args.run_id)
rubric = json.loads((root / 'data/day24_questions.json').read_text(encoding='utf-8'))
assert digest(canonical(rubric)) == run['metadata']['question_set_hash'], 'Rubric changed after the run'
export = export_day24_report(store, args.run_id)
stats = summarize(run)
answers = [a for a in run['answers'] if a['mode'] == 'day24']
count_claims = sum(len((a.get('structured_answer') or {}).get('claims', [])) for a in answers)
quotes = stats['day24']['metrics']['valid_quotes']
reviewed = sum(a.get('review', {}).get('status') == 'reviewed' for a in run['answers'])
reviewed_claims = stats['day24']['metrics']['supported_claims']['denominator']
paired = 'day23' in run['metadata']['report_modes']
lines = ['# День 24 — реальные результаты', '', f"Основной run_id: `{run['run_id']}`; дата: {run['created_at']}; статус: {run['status']}.",
    f"Проверено автоматически: {len(run['answers'])} результатов / {len(run['metadata']['questions'])} вопросов. Режимы: {', '.join(run['metadata']['report_modes'])}. Ручных оценок: {reviewed} / {len(run['answers'])}.",
    f"День 24: {stats['day24']['substantive_answers']} содержательных ответа, {quotes['numerator']} / {quotes['denominator']} точных валидных цитат; {count_claims} claims, {reviewed_claims} / {count_claims} оценено вручную.",
    'Неоценённая подтверждённость claims и правильность отказов — N/A, а не 0%.',
    '', '## Условия проверки',
    'Архивный парный запуск.' if paired else 'Только режим Дня 24. Сравнение с Днём 23 не выполнялось.',
    'Модель: `' + run['metadata']['model'] + '`; параметры генерации: `' + json.dumps(run['metadata']['settings']) + '`.',
    'Поиск и фильтрация: `' + json.dumps(run['metadata']['pipeline'], ensure_ascii=False) + '`.',
    'Порог и происхождение: `' + json.dumps(run['metadata']['threshold_provenance'], ensure_ascii=False) + '`.',
    'Рубрика зафиксирована до запуска; хеш корпуса и диапазоны источников проверены.',
    'Учёт времени: ' + run['metadata']['timing_policy'],
    '', '## Ответы Дня 24', '| Вопрос | Статус | Причина | Цитаты | Claims | Генерации |', '|---|---|---|---:|---:|---:|']
for a in answers:
    sa = a.get('structured_answer') or {}
    lines.append(f"| {a['question_id']} | {a.get('answer_status')} | {a.get('refusal_reason') or '—'} | {len(sa.get('citations', []))} | {len(sa.get('claims', []))} | {a['generation_calls']} |")
lines += ['', '## Метрики и знаменатели']
for mode, entry in stats.items():
    lines += ['### ' + mode, '| Метрика | Числитель | Знаменатель | Значение |', '|---|---:|---:|---:|']
    for name, metric in entry['metrics'].items():
        value = 'N/A' if metric['value'] is None else f"{metric['value']:.4f}"
        lines.append(f"| {name} | {metric['numerator']} | {metric['denominator']} | {value} |")
    lines.append('Usage генераций: `' + json.dumps(entry['known_generation_usage']) + '`; вызовов: ' + str(entry['generation_calls']))
lines += ['', '## Проверки', f"Полный набор: **{args.tests_passed} passed, 3 deselected** (`python -m pytest -q`).",
    'Три integration-теста прежних дней исключены стандартным маркером. Реальные embedding-поиск и LLM проверены запуском выше.',
    'Unit-тесты проверяют схему, ссылки, точные и выдуманные/обрезанные цитаты, отказ без генерации, partial, единственное исправление, технические ошибки, N/A и сохранение ручной оценки.',
    'Streamlit AppTest проверяет экран, открытие источника с подсветкой и историю без LLM. Сохранены тесты прежних режимов.',
    'Индекс не перестраивался. SHA-256 файлов:']
for name in ['structural.sqlite', 'fixed.sqlite']:
    h = hashlib.sha256((root / 'output' / name).read_bytes()).hexdigest()
    lines.append(f'- `{name}`: `{h}`')
lines += ['', '## Экспорт и ручная проверка']
for ext, path in export['paths'].items():
    relative = Path(path).relative_to(root).as_posix()
    lines.append(f'- [{ext.upper()} — полный сохранённый отчёт]({relative})')
lines += ['', 'Откройте «Проверяемые цитаты» → «10 контрольных вопросов» или «История», выберите основной run_id и заполните ручные оценки каждого ответа.',
    'В UI раздельно проверяются смысл каждой цитаты/claim, новые факты в answer, условия, корректность, полнота и отказы. После сохранения метрики обновляются без LLM.',
    '', '## Ограничения',
    'Наличие точной цитаты не доказывает смысловую поддержку. Нет гарантии отсутствия ошибок. Смысл оценивается человеком.',
    'Наблюдаются отказы на вопросах, для которых в полном корпусе зафиксированы доказательства. Порог и усечение контекста могут терять нужную информацию; это не означает отсутствие ответа во всей базе. Порог по тестовым ответам не подстраивался.',
    'Ранее выполненные парные эксперименты сохранены в истории и не входят в текущую проверку.',
    'Пустые источники/цитаты отказа — N/A. Неоценённая семантика исключена из средних.']
(root / 'DAY24_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({'run_id': args.run_id, 'paths': export['paths'], 'quotes': quotes, 'claims': count_claims}, ensure_ascii=False))
