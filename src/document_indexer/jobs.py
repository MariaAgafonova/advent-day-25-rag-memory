from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from copy import deepcopy
from uuid import uuid4

class JobManager:
    """Process-lifetime state; workers never call Streamlit. No duplicate running key."""
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indexer")
        self.lock = RLock()
        self.jobs = {}

    def submit(self, key, action):
        with self.lock:
            for job_id, job in self.jobs.items():
                if job["key"] == key and job["status"] in {"queued", "running"}:
                    return job_id
            job_id = str(uuid4())
            self.jobs[job_id] = {"key": key, "status": "queued", "progress": {}, "result": None, "error": None}
            self.executor.submit(self._run, job_id, action)
            return job_id

    def _run(self, job_id, action):
        def progress(stage, done=0, total=None):
            with self.lock:
                self.jobs[job_id]["progress"] = {"stage": stage, "done": done, "total": total}
        with self.lock:
            self.jobs[job_id]["status"] = "running"
        try:
            result = action(progress)
            with self.lock:
                self.jobs[job_id].update(status="completed", result=result)
        except Exception as exc:
            with self.lock:
                self.jobs[job_id].update(status="failed", error=str(exc))

    def snapshot(self, job_id):
        with self.lock:
            return deepcopy(self.jobs.get(job_id))

    def active(self, key):
        with self.lock:
            return any(j["key"] == key and j["status"] in {"queued", "running"} for j in self.jobs.values())

    def close(self):
        self.executor.shutdown(wait=True)

MANAGER = JobManager()
