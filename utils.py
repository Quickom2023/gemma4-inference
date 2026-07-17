"""Plumbing for the video analysis API: job queue, uploads, downloads, STT.

Everything here is transport/bookkeeping. Model inference lives in service.py and
the HTTP endpoints live in api.py.
"""
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from urllib.parse import unquote, urlparse

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException, UploadFile

from service import VideoAnalyzer

# Must run before the constants below read the environment, not after.
load_dotenv()

# Clients routinely upload video as application/octet-stream (curl does), so the
# suffix — which is what the decoder dispatches on anyway — is the real gate.
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg"}
MAX_VIDEO_URL_BYTES = 1000 * 1024 * 1024  # ~1 GB cap on downloaded videos; tune as needed
VIDEO_URL_TIMEOUT_S = 60  # connect+read timeout for the download
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
JOB_TTL_S = 3600  # finished jobs are reaped an hour after they finish

STT_API_URL = os.environ.get(
    "STT_API_URL", "http://34.142.218.10:8001/v1/audio/transcriptions",
)
STT_TIMEOUT_S = float(os.environ.get("STT_TIMEOUT_S", 300))

# Three terminal-facing states. A job waiting its turn is still "running" as far as
# a client is concerned — queue_position vs progress tells the two apart.
RUNNING, SUCCEEDED, FAILED = "running", "succeeded", "failed"


def unlink(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Getting a video onto disk
# --------------------------------------------------------------------------- #

def materialize_video(file: UploadFile | None, url: str | None) -> str:
    """Get the request's video onto disk, whichever way it arrived."""
    if (file is None) == (url is None):
        raise HTTPException(
            status_code=400,
            detail="provide exactly one of file (upload) or url",
        )
    return save_upload(file) if file is not None else download_video(url)


def save_upload(upload: UploadFile) -> str:
    """Spool a multipart upload to a temp file and return its path."""
    suffix = os.path.splitext(upload.filename or "")[1].lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"expected a video file {sorted(VIDEO_SUFFIXES)}, "
                   f"got filename {upload.filename!r}",
        )
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="gemma4_upload_")
    with os.fdopen(fd, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


def download_video(url: str) -> str:
    """Fetch a video URL to a temp file and return its path.

    Streams to disk with a hard byte cap, so a hostile or mistaken URL can't fill
    the disk. Content-Length is advisory — servers lie or omit it — so the running
    total is what actually enforces the cap.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400,
                            detail=f"url must be http(s), got {parsed.scheme!r}")

    suffix = os.path.splitext(unquote(parsed.path))[1].lower()
    if suffix not in VIDEO_SUFFIXES:
        suffix = ".mp4"  # URLs often carry no usable extension; let the decoder decide

    fd, path = tempfile.mkstemp(suffix=suffix, prefix="gemma4_download_")
    written = 0
    try:
        with os.fdopen(fd, "wb") as f:
            with httpx.stream("GET", url, timeout=VIDEO_URL_TIMEOUT_S,
                              follow_redirects=True) as r:
                if r.status_code >= 400:
                    raise HTTPException(
                        status_code=400,
                        detail=f"url returned HTTP {r.status_code}",
                    )
                for block in r.iter_bytes(1024 * 1024):
                    written += len(block)
                    if written > MAX_VIDEO_URL_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"url exceeds {MAX_VIDEO_URL_BYTES} bytes",
                        )
                    f.write(block)
    except HTTPException:
        unlink(path)
        raise
    except Exception as e:
        unlink(path)
        raise HTTPException(status_code=400,
                            detail=f"could not fetch url: {type(e).__name__}: {e}")

    if written == 0:
        unlink(path)
        raise HTTPException(status_code=400, detail="url returned an empty body")
    return path


# --------------------------------------------------------------------------- #
# Speech-to-text
# --------------------------------------------------------------------------- #

def extract_audio(video_path: str) -> str | None:
    """Rip the audio track to mp3. None if the video has no audio at all."""
    fd, path = tempfile.mkstemp(suffix=".mp3", prefix="gemma4_audio_")
    os.close(fd)
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", video_path, "-vn",
         "-acodec", "libmp3lame", "-q:a", "4", path],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    # ffmpeg fails outright when there's no audio stream to map.
    if proc.returncode != 0 or os.path.getsize(path) == 0:
        unlink(path)
        return None
    return path


def transcribe_audio(audio_path: str | None) -> str | None:
    """POST an extracted audio file to the STT service and return its text.

    Takes ownership of `audio_path` and deletes it. Returns None when there was no
    audio to send or STT is unreachable: the video description is the primary
    product here, and a dead STT service shouldn't take the whole request down.
    """
    if audio_path is None:
        return None
    try:
        with open(audio_path, "rb") as f:
            r = httpx.post(
                STT_API_URL,
                files={"file": (os.path.basename(audio_path), f, "audio/mpeg")},
                timeout=STT_TIMEOUT_S,
            )
        r.raise_for_status()
        # The service replies {"text": "...", "language": "..."}.
        text = r.json().get("text")
        return text.strip() if isinstance(text, str) else None
    except Exception:
        return None
    finally:
        unlink(audio_path)


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #

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
    # Set once the job reaches a terminal state, so callers can block on it instead
    # of polling the store.
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
        # only runs lifespan shutdown once in-flight requests are done, so by the
        # time we get here they have no waiters.
        self._stopping.set()
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=30)
        # Uploads for jobs that never ran would otherwise linger in /tmp, and any
        # caller blocked on a job we'll never run has to be released.
        with self._lock:
            for job in self._jobs.values():
                if job.status == RUNNING:
                    job.status, job.error = FAILED, "server shut down before this job ran"
                unlink(job.video_path)
                job.done.set()
            self._jobs.clear()

    @property
    def model_id(self) -> str | None:
        return self._analyzer.model_id if self._analyzer else None

    def text_is_safe(self, text: str | None) -> bool:
        """False only if the model judges `text` nsfw. Nothing to check = safe.

        Deliberately not fail-open: if the check itself blows up the exception
        propagates and the request 500s, because a safety check that silently
        returns "safe" when it didn't run is worse than a failed request.
        """
        if not text or not text.strip():
            return True
        return not self._analyzer.check_text(text)

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
                unlink(job.video_path)
                job.done.set()


store = JobStore()


def enqueue(video_path: str, max_new_tokens: int, job_id: str | None = None) -> Job:
    """Queue an already-materialized video file for analysis."""
    if job_id is not None and not JOB_ID_RE.match(job_id):
        unlink(video_path)
        raise HTTPException(status_code=400,
                            detail="job_id must be 1-64 chars of [A-Za-z0-9_.-]")

    job = Job(id=job_id or uuid.uuid4().hex, video_path=video_path,
              max_new_tokens=max_new_tokens)
    try:
        store.add(job)
    except KeyError:
        # A client-supplied id that's already taken; don't silently clobber it.
        unlink(video_path)
        raise HTTPException(status_code=404, detail=f"job_id {job.id!r} already exists")
    return job
