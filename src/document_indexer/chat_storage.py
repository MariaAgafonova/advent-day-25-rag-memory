"""Chat data lives separately from the immutable knowledge index."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4, UUID
import json
import os
import sqlite3
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def empty_state(cid, revision=0):
    return dict(conversation_id=cid, goal=[], clarifications=[], constraints=[], terms=[],
                decisions=[], open_questions=[], summary="", revision=revision, updated_at=now())


class ChatStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS conversations(
                  id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL, epoch INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS messages(
                  seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                  conversation_id TEXT NOT NULL REFERENCES conversations(id),
                  role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL,
                  created_at TEXT NOT NULL, status TEXT NOT NULL, request_id TEXT NOT NULL,
                  epoch INTEGER NOT NULL, result TEXT, UNIQUE(conversation_id,request_id,role));
                CREATE TABLE IF NOT EXISTS task_states(
                  conversation_id TEXT PRIMARY KEY REFERENCES conversations(id), state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS revisions(
                  conversation_id TEXT NOT NULL REFERENCES conversations(id), revision INTEGER NOT NULL,
                  message_id TEXT, reason TEXT NOT NULL, state TEXT NOT NULL,
                  PRIMARY KEY(conversation_id,revision));
                CREATE TABLE IF NOT EXISTS requests(
                  conversation_id TEXT NOT NULL REFERENCES conversations(id), request_id TEXT NOT NULL,
                  status TEXT NOT NULL, memory_done INTEGER NOT NULL DEFAULT 0,
                  memory_error INTEGER NOT NULL DEFAULT 0, preparation TEXT, result TEXT,
                  PRIMARY KEY(conversation_id,request_id));
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def serial(self, cid):
        # OS releases this lock on process death; works across Streamlit sessions/processes.
        UUID(cid)
        directory = self.path.parent / (self.path.name + '.locks')
        directory.mkdir(exist_ok=True)
        with (directory / cid).open('a+b') as handle:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b'0')
                handle.flush()
            start = time.monotonic()
            while True:
                try:
                    handle.seek(0)
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() - start > 600:
                        raise TimeoutError('Диалог занят; повторите запрос с прежним request_id')
                    time.sleep(.05)
            try:
                self.conversation(cid)
                self.recover(cid)
                yield
            finally:
                handle.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def create(self, title='Новый диалог'):
        cid, stamp = str(uuid4()), now()
        state = empty_state(cid)
        with self.connect() as db:
            db.execute('INSERT INTO conversations VALUES(?,?,?,?,0)', (cid, title[:100], stamp, stamp))
            db.execute('INSERT INTO task_states VALUES(?,?)', (cid, json.dumps(state, ensure_ascii=False)))
        return cid

    def conversations(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM conversations ORDER BY updated_at DESC')]

    def conversation(self, cid):
        with self.connect() as db:
            row = db.execute('SELECT * FROM conversations WHERE id=?', (cid,)).fetchone()
        if row is None:
            raise ValueError('Диалог не найден')
        return dict(row)

    def state(self, cid):
        self.conversation(cid)
        with self.connect() as db:
            return json.loads(db.execute('SELECT state FROM task_states WHERE conversation_id=?', (cid,)).fetchone()[0])

    def history(self, cid, current_epoch=False):
        conv = self.conversation(cid)
        with self.connect() as db:
            rows = db.execute('SELECT * FROM messages WHERE conversation_id=?' +
                              (' AND epoch=?' if current_epoch else '') + ' ORDER BY seq',
                              (cid, conv['epoch']) if current_epoch else (cid,)).fetchall()
        return [{**dict(r), 'result': json.loads(r['result']) if r['result'] else None} for r in rows]

    def request(self, cid, rid):
        with self.connect() as db:
            row = db.execute('SELECT * FROM requests WHERE conversation_id=? AND request_id=?', (cid, rid)).fetchone()
        if row is None:
            return None
        return {**dict(row), 'result': json.loads(row['result']) if row['result'] else None,
                'preparation': json.loads(row['preparation']) if row['preparation'] else None}

    def begin(self, cid, rid, content, retry=False):
        old = self.request(cid, rid)
        if old:
            original = next(m for m in self.history(cid) if m['request_id'] == rid and m['role'] == 'user')
            if original['content'] != content:
                raise ValueError('request_id уже использован для другого сообщения')
            if not retry or old['status'] != 'failed':
                return original['id'], old['result']
            if original['epoch'] != self.conversation(cid)['epoch']:
                raise ValueError('Нельзя повторять запрос до сброса памяти')
            users = [m for m in self.history(cid, True) if m['role'] == 'user']
            if users[-1]['id'] != original['id']:
                raise ValueError('Повтор доступен только для последнего сообщения')
            with self.connect() as db:
                db.execute("UPDATE requests SET status='pending' WHERE conversation_id=? AND request_id=?", (cid, rid))
                db.execute("UPDATE messages SET status='pending' WHERE conversation_id=? AND request_id=?", (cid, rid))
            return original['id'], None
        stamp, mid = now(), str(uuid4())
        conv = self.conversation(cid)
        with self.connect() as db:
            db.execute('INSERT INTO requests(conversation_id,request_id,status) VALUES(?,?,?)', (cid, rid, 'pending'))
            for role, ident, text in [('user', mid, content), ('assistant', str(uuid4()), '')]:
                db.execute('INSERT INTO messages(id,conversation_id,role,content,created_at,status,request_id,epoch) VALUES(?,?,?,?,?,?,?,?)',
                           (ident, cid, role, text, stamp, 'pending', rid, conv['epoch']))
            db.execute('UPDATE conversations SET updated_at=?, title=CASE WHEN title=? THEN ? ELSE title END WHERE id=?',
                       (stamp, 'Новый диалог', content[:70], cid))
        return mid, None

    def memory_blocked(self, cid, rid):
        epoch = self.conversation(cid)['epoch']
        with self.connect() as db:
            return bool(db.execute('''SELECT 1 FROM requests r JOIN messages m
              ON r.conversation_id=m.conversation_id AND r.request_id=m.request_id
              WHERE r.conversation_id=? AND r.request_id<>? AND r.memory_error=1 AND m.epoch=? LIMIT 1''',
                                   (cid, rid, epoch)).fetchone())

    def save_memory(self, cid, rid, mid, state, preparation):
        with self.connect() as db:
            existing = json.loads(db.execute('SELECT state FROM task_states WHERE conversation_id=?', (cid,)).fetchone()[0])
            if state['revision'] != existing['revision'] + 1:
                raise ValueError('Конфликт ревизии памяти')
            serialized = json.dumps(state, ensure_ascii=False)
            db.execute('UPDATE task_states SET state=? WHERE conversation_id=?', (serialized, cid))
            db.execute('INSERT INTO revisions VALUES(?,?,?,?,?)', (cid, state['revision'], mid, 'message', serialized))
            db.execute('UPDATE requests SET memory_done=1,memory_error=0,preparation=? WHERE conversation_id=? AND request_id=?',
                       (json.dumps(preparation, ensure_ascii=False), cid, rid))
            if preparation.get('reset_memory'):
                db.execute('UPDATE conversations SET epoch=epoch+1 WHERE id=?', (cid,))
                db.execute('UPDATE messages SET epoch=(SELECT epoch FROM conversations WHERE id=?) WHERE conversation_id=? AND request_id=?', (cid, cid, rid))

    def finish(self, cid, rid, result, memory_error=False):
        serialized = json.dumps(result, ensure_ascii=False)
        status = 'failed' if result['status'] == 'error' else 'completed'
        with self.connect() as db:
            db.execute('UPDATE requests SET status=?,result=?,memory_error=? WHERE conversation_id=? AND request_id=?',
                       (status, serialized, int(memory_error), cid, rid))
            db.execute('UPDATE messages SET status=? WHERE conversation_id=? AND request_id=?', (status, cid, rid))
            db.execute("UPDATE messages SET content=?,result=? WHERE conversation_id=? AND request_id=? AND role='assistant'",
                       (result['answer'], serialized, cid, rid))
            db.execute('UPDATE conversations SET updated_at=? WHERE id=?', (now(), cid))

    def recover(self, cid):
        with self.connect() as db:
            pending = db.execute("SELECT * FROM requests WHERE conversation_id=? AND status='pending'", (cid,)).fetchall()
        for row in pending:
            self.finish(cid, row['request_id'], dict(status='error', answer='Обработка прервана. Повторите запрос.',
                        sources=[], error=dict(stage='interrupted', code='interrupted'), diagnostics={}), not row['memory_done'])

    def reset(self, cid):
        with self.serial(cid):
            state = empty_state(cid, self.state(cid)['revision'] + 1)
            text = json.dumps(state, ensure_ascii=False)
            with self.connect() as db:
                db.execute('UPDATE task_states SET state=? WHERE conversation_id=?', (text, cid))
                db.execute('INSERT INTO revisions VALUES(?,?,?,?,?)', (cid, state['revision'], None, 'reset', text))
                db.execute('UPDATE conversations SET epoch=epoch+1,updated_at=? WHERE id=?', (now(), cid))
            return state

    def revisions(self, cid):
        with self.connect() as db:
            return [{**dict(r), 'state': json.loads(r['state'])} for r in db.execute(
                'SELECT * FROM revisions WHERE conversation_id=? ORDER BY revision', (cid,))]
