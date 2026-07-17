"""HTTP job API for Gemma 4 E2B video analysis.

    python api.py                       # or: uvicorn api:app --host 0.0.0.0 --port 8000

Submit a job, then poll it:

    curl -F video=@tolam.mp4 http://localhost:8000/api/v1/video/analyze
    # -> {"job_id": "3f2b...", "status": "running"}

    curl http://localhost:8000/api/v1/video/analyze/3f2b...
    # -> {"job_id": "3f2b...", "status": "running", "queue_position": 1}   still waiting
    # -> {"job_id": "3f2b...", "status": "running", "progress": {...}}     model working
    # -> {"job_id": "3f2b...", "status": "succeeded",
    #     "result": {"description": "...", "safety": {"nudity": false, "nsfw": false}}}
    # -> {"job_id": "3f2b...", "status": "failed", "error": "..."}

Pass job_id=<your-id> on submit to choose the id yourself (404 if it's taken).

Or block until it's done and skip the job bookkeeping entirely:

    curl -F video=@2girls.mp4 http://localhost:8000/api/v1/video/analyze/wait
    # -> {"description": "...", "safety": {"nudity": false, "nsfw": false}}

Analysis is GPU-bound and strictly serial, so jobs run one at a time on a single
worker thread. Job state lives in memory: it does not survive a restart, and it
assumes this one process is the only worker.
"""
import os
import hmac
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from dotenv import load_dotenv

from fastapi import FastAPI, File, Header, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from service import DEFAULT_MAX_NEW_TOKENS, VideoAnalyzer

# Clients routinely upload video as application/octet-stream (curl does), so the
# suffix — which is what the decoder dispatches on anyway — is the real gate.
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg"}
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
JOB_TTL_S = 3600  # finished jobs are reaped an hour after they finish

# Three terminal-facing states. A job waiting its turn is still "running" as far as
# a client is concerned — queue_position vs progress tells the two apart.
RUNNING, SUCCEEDED, FAILED = "running", "succeeded", "failed"

load_dotenv()

@dataclass
class Job:
    id: str
    video_path: str
    max_new_tokens: int
    status: str = RUNNING
    started: bool = False   # False = still queued behind other jobs
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    chunks_done: int = 0
    chunks_total: int = 0
    result: dict | None = None
    error: str | None = None
    # Set once the job reaches a terminal state, so /wait can block on it instead
    # of polling its own store.
    done: threading.Event = field(default_factory=threading.Event)

    def public(self, queue_position: int | None = None) -> dict:
        body = {"job_id": self.id, "status": self.status}
        if self.status == RUNNING:
            if self.started:
                body["progress"] = {"chunks_done": self.chunks_done,
                                    "chunks_total": self.chunks_total}
            elif queue_position is not None:
                body["queue_position"] = queue_position
        if self.status == SUCCEEDED:
            body["result"] = self.result
        if self.status == FAILED:
            body["error"] = self.error
        return body


class JobStore:
    """In-memory jobs + the single worker thread that drains them."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._analyzer: VideoAnalyzer | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    def start(self) -> None:
        self._analyzer = VideoAnalyzer()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # Abandon whatever is still queued rather than working through the whole
        # backlog first — an in-memory store loses those jobs on restart anyway, so
        # draining them only makes shutdown take as long as the queue. The job in
        # flight still finishes. Nothing is blocked on the abandoned ones: uvicorn
        # only runs lifespan shutdown once in-flight requests (i.e. any /wait) are
        # done, so by the time we get here they have no waiters.
        self._stopping.set()
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=30)
        # Uploads for jobs that never ran would otherwise linger in /tmp, and any
        # /wait caller blocked on a job we'll never run has to be released.
        with self._lock:
            for job in self._jobs.values():
                if job.status == RUNNING:
                    job.status, job.error = FAILED, "server shut down before this job ran"
                _unlink(job.video_path)
                job.done.set()
            self._jobs.clear()

    @property
    def model_id(self) -> str | None:
        return self._analyzer.model_id if self._analyzer else None

    def add(self, job: Job) -> None:
        self._reap()
        with self._lock:
            if job.id in self._jobs:
                raise KeyError(job.id)
            self._jobs[job.id] = job
        self._queue.put(job.id)

    def get(self, job_id: str) -> Job | None:
        self._reap()
        with self._lock:
            return self._jobs.get(job_id)

    def queue_position(self, job_id: str) -> int:
        """How many queued jobs sit ahead of this one."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return 0
            return sum(1 for j in self._jobs.values()
                       if j.status == RUNNING and not j.started
                       and j.created_at < job.created_at)

    def _reap(self) -> None:
        cutoff = time.time() - JOB_TTL_S
        with self._lock:
            stale = [i for i, j in self._jobs.items()
                     if j.finished_at is not None and j.finished_at < cutoff]
            for i in stale:
                del self._jobs[i]

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None or self._stopping.is_set():  # shutting down
                return
            with self._lock:
                job = self._jobs.get(job_id)
            if job is None:  # reaped before it ran
                continue

            job.started = True
            try:
                def progress(done: int, total: int, _j: Job = job) -> None:
                    _j.chunks_done, _j.chunks_total = done, total

                result = self._analyzer.analyze(
                    job.video_path, job.max_new_tokens, progress=progress,
                )
                job.result = {
                    "description": result.description,
                    "safety": {"nudity": result.nudity, "nsfw": result.nsfw},
                }
                job.status = SUCCEEDED
            except Exception as e:
                job.error = f"{type(e).__name__}: {e}"
                job.status = FAILED
            finally:
                job.finished_at = time.time()
                _unlink(job.video_path)
                job.done.set()


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


store = JobStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load at startup so the first job doesn't pay the model-load cost.
    store.start()
    yield
    store.stop()


app = FastAPI(title="Gemma 4 E2B video analysis", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"ok": store.model_id is not None, "model": store.model_id}


def _enqueue(video: UploadFile, max_new_tokens: int, job_id: str | None) -> Job:
    """Validate the upload, spool it to disk, and queue a job for it."""
    suffix = os.path.splitext(video.filename or "")[1].lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"expected a video file {sorted(VIDEO_SUFFIXES)}, "
                   f"got filename {video.filename!r}",
        )
    if job_id is not None and not JOB_ID_RE.match(job_id):
        raise HTTPException(
            status_code=400,
            detail="job_id must be 1-64 chars of [A-Za-z0-9_.-]",
        )

    fd, path = tempfile.mkstemp(suffix=suffix, prefix="gemma4_upload_")
    with os.fdopen(fd, "wb") as f:
        shutil.copyfileobj(video.file, f)

    job = Job(id=job_id or uuid.uuid4().hex, video_path=path,
              max_new_tokens=max_new_tokens)
    try:
        store.add(job)
    except KeyError:
        # A client-supplied id that's already taken; don't silently clobber it.
        _unlink(path)
        raise HTTPException(status_code=404, detail=f"job_id {job.id!r} already exists")
    return job

def verify_api_key(x_api_key):
    expected_api_key = os.environ.get("X_API_KEY")
    if not expected_api_key:
        raise HTTPException(status_code=503, detail="X_API_KEY is not configured.")

    if not x_api_key or not hmac.compare_digest(x_api_key, expected_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key.")

# @app.post("/api/v1/video/analyze", status_code=200)
# async def submit(
#     video: UploadFile = File(...),
#     max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
#     job_id: str | None = Form(None),
# ) -> dict:
#     """Queue a job and return its id; poll GET /api/v1/video/analyze/{job_id}."""
#     job = _enqueue(video, max_new_tokens, job_id)
#     return job.public()


@app.post("/api/v1/video/analyze", status_code=200)
async def analyze_and_wait(
    file: UploadFile = File(...),
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
    x_api_key: str | None = Header(default=None)
) -> dict:
    """Analyze and block until the result is ready — no job id, no polling.

    Runs through the same single worker as the job API, so this call also waits out
    any jobs already queued ahead of it. Long videos hold the connection open for
    the whole run; use the job API if that's a problem.
    """
    verify_api_key(x_api_key)
    
    job = _enqueue(file, max_new_tokens, None)
    await run_in_threadpool(job.done.wait)
    if job.status == FAILED:
        raise HTTPException(status_code=500, detail=job.error)
    return job.result


# @app.get("/api/v1/video/analyze/{job_id}")
# async def status(job_id: str) -> dict:
#     job = store.get(job_id)
#     if job is None:
#         raise HTTPException(
#             status_code=404,
#             detail=f"no job {job_id!r} (unknown, or finished over {JOB_TTL_S}s ago)",
#         )
#     return job.public(store.queue_position(job_id))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8000)),
    )
