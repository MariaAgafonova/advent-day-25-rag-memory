from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from dataclasses import asdict
import json
import sqlite3

def now():
    return datetime.now(timezone.utc).isoformat()

class ExperimentStorage:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "experiments.sqlite"
        with self.connection() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS snapshots(build_id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    kind TEXT NOT NULL, status TEXT NOT NULL, metadata TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS answers(request_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
                    question_id TEXT NOT NULL, mode TEXT NOT NULL, data TEXT NOT NULL, UNIQUE(run_id,question_id,mode));
                CREATE TABLE IF NOT EXISTS reviews(request_id TEXT PRIMARY KEY REFERENCES answers(request_id),
                    updated_at TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS artifacts(artifact_id TEXT PRIMARY KEY, kind TEXT NOT NULL, data TEXT NOT NULL);
            ''')

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create_run(self, kind, metadata, index=None):
        run_id = str(uuid4())
        metadata = dict(metadata)
        with self.connection() as conn:
            if index is not None:
                build_id = index.config["build_id"]
                snapshot = {"config": index.config, "documents": {k: asdict(d) for k, d in index.documents.items()}}
                conn.execute("INSERT OR IGNORE INTO snapshots VALUES (?,?)", (build_id, json.dumps(snapshot, ensure_ascii=False)))
                metadata["build_id"] = build_id
            conn.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (run_id, now(), kind, "running", json.dumps(metadata, ensure_ascii=False)))
        return run_id

    def save_answer(self, run_id, question_id, result):
        with self.connection() as conn:
            conn.execute("INSERT INTO answers VALUES (?,?,?,?,?)", (result.request_id, run_id, question_id, result.mode, json.dumps(asdict(result), ensure_ascii=False)))

    def finish_run(self, run_id):
        run = self.get_run(run_id)
        planned = run["metadata"].get("planned_answers", run["metadata"]["planned_calls"])
        success = sum(a["status"] in {"completed", "citation_error", "no_relevant_context", "no_results", "empty_context"} for a in run["answers"])
        status = "completed" if success == planned else "partial" if success else "failed"
        with self.connection() as conn:
            conn.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))

    def list_runs(self):
        with self.connection() as conn:
            return [{"run_id": rid, "created_at": time, "kind": kind, "status": status} for rid, time, kind, status in
                    conn.execute("SELECT run_id,created_at,kind,status FROM runs ORDER BY created_at DESC")]

    def get_run(self, run_id):
        with self.connection() as conn:
            row = conn.execute("SELECT created_at,kind,status,metadata FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise ValueError("Эксперимент не найден")
            answers = []
            for qid, data, review in conn.execute("SELECT a.question_id,a.data,r.data FROM answers a LEFT JOIN reviews r ON a.request_id=r.request_id WHERE a.run_id=? ORDER BY a.rowid", (run_id,)):
                answers.append({**json.loads(data), "question_id": qid, "review": json.loads(review) if review else {"status": "unreviewed"}})
                current = answers[-1]
                if current['mode'] == 'day24' and current['review'].get('status') == 'reviewed':
                    current.setdefault('validation', {})['semantic_support'] = {
                        'status': 'reviewed', 'reviewed_by': 'human', 'claims': current['review']['claims'],
                        'answer_has_extra_facts': current['review']['answer_has_extra_facts'],
                        'comment': current['review']['comment']}
            review_row = conn.execute("SELECT data FROM artifacts WHERE artifact_id=?", ("comparison_review:" + run_id,)).fetchone()
        result = {"run_id": run_id, "created_at": row[0], "kind": row[1], "status": row[2], "metadata": json.loads(row[3]), "answers": answers}
        if row[1] == "day23_comparison":
            result["comparison_review"] = json.loads(review_row[0]) if review_row else None
        return result

    def save_comparison_review(self, run_id, winner, comment=""):
        run = self.get_run(run_id)
        modes = {a["mode"] for a in run["answers"]}
        if run["kind"] != "day23_comparison" or modes not in ({"baseline", "filter"}, {"baseline", "reranker"}):
            raise ValueError("Оценка доступна для сравнения baseline с фильтром или реранкером")
        if winner not in modes | {"equal", "neither"}:
            raise ValueError("Выберите оценку после чтения ответов")
        if any(a["status"] == "failed" for a in run["answers"]):
            raise ValueError("Техническую ошибку нельзя оценивать как качество ответа")
        self.save_artifact("comparison_review", {"winner": winner, "comment": str(comment)[:5000],
            "reviewed_by": "human", "updated_at": now()}, "comparison_review:" + run_id)

    def snapshot(self, build_id):
        with self.connection() as conn:
            row = conn.execute("SELECT data FROM snapshots WHERE build_id=?", (build_id,)).fetchone()
        if row is None:
            raise ValueError("Снимок источников не найден")
        return json.loads(row[0])

    def save_review(self, request_id, review):
        from .rag_evaluation import validate_review
        with self.connection() as conn:
            row = conn.execute("SELECT a.data,a.question_id,r.metadata FROM answers a JOIN runs r ON a.run_id=r.run_id WHERE a.request_id=?", (request_id,)).fetchone()
            if row is None:
                raise ValueError("Ответ не найден")
            result, qid, metadata = json.loads(row[0]), row[1], json.loads(row[2])
            question = next((q for q in metadata.get("questions", []) if q["id"] == qid), None)
            if question is None:
                raise ValueError("Рубрика доступна для контрольных вопросов")
            if result["mode"] in {'day23', 'day24'}:
                from .day24_evaluation import validate_day24_review
                review = validate_day24_review(review, result, question)
            elif result["mode"] in {"baseline", "rewrite", "filter", "rewrite_filter", "reranker"}:
                from .day23_evaluation import validate_day23_review
                review = validate_day23_review(review, result, question)
            else:
                review = validate_review(review, result, question)
            conn.execute("INSERT INTO reviews VALUES (?,?,?) ON CONFLICT(request_id) DO UPDATE SET updated_at=excluded.updated_at,data=excluded.data", (request_id, now(), json.dumps(review, ensure_ascii=False)))

    def save_artifact(self, kind, data, artifact_id=None):
        artifact_id = artifact_id or str(uuid4())
        with self.connection() as conn:
            conn.execute("INSERT INTO artifacts VALUES (?,?,?) ON CONFLICT(artifact_id) DO UPDATE SET data=excluded.data",
                         (artifact_id, kind, json.dumps(data, ensure_ascii=False)))
        return artifact_id

    def get_artifact(self, artifact_id):
        with self.connection() as conn:
            row = conn.execute("SELECT data FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        if row is None:
            raise ValueError("Артефакт не найден")
        return json.loads(row[0])

    def list_artifacts(self, kind):
        with self.connection() as conn:
            return [{"id": rid, **json.loads(data)} for rid, data in conn.execute("SELECT artifact_id,data FROM artifacts WHERE kind=? ORDER BY rowid DESC", (kind,))]

    def append_rewrite(self, run_id, rewrite):
        with self.connection() as conn:
            row = conn.execute("SELECT metadata FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise ValueError("Эксперимент не найден")
            metadata = json.loads(row[0])
            metadata.setdefault("rewrites", []).append(rewrite)
            conn.execute("UPDATE runs SET metadata=? WHERE run_id=?", (json.dumps(metadata, ensure_ascii=False), run_id))
