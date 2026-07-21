"""HTTP API for Gemma 4 E2B video analysis.

    python api.py                       # or: uvicorn api:app --host 0.0.0.0 --port 8000

Every call needs the x-api-key header, matched against X_API_KEY in .env.
Send a video either as a file upload or as a URL — exactly one of the two:

    curl -H "x-api-key: $X_API_KEY" -F file=@2girls.mp4 \
         http://localhost:8000/api/v1/video/analyze

    curl -H "x-api-key: $X_API_KEY" -F url=https://example.com/clip.mp4 \
         http://localhost:8000/api/v1/video/analyze

    # -> {"description": "...", "duration": 12.34, "transcription": "...", "safety": true}
    #    duration is the video length in seconds (0.0 if ffprobe can't tell)
    #    transcription is null if the video is silent or STT is down

Send an optional text field (a caption, title, comment) and it is judged too:

    curl -H "x-api-key: $X_API_KEY" -F file=@clip.mp4 -F text="check this out" \
         http://localhost:8000/api/v1/video/analyze

safety is the verdict, not the detail. Three sources feed it — what the video
shows, what the caller wrote in text, and what is said aloud (the transcript).
Any one of them being NSFW makes it false. The description and transcription are
returned either way.

The call blocks until the result is ready. Analysis is GPU-bound and strictly
serial, so requests queue on a single worker thread; speech-to-text runs against
the STT service concurrently with the model, not after it.

To judge a piece of text on its own — no audio, no video — POST JSON to
/api/v1/content/validate. Use it when you already hold the content as text:

    curl -H "x-api-key: $X_API_KEY" -H "content-type: application/json" \
         -d '{"text": "some comment to check"}' \
         http://localhost:8000/api/v1/content/validate

    # -> {"safety": true}   # false when the text is nsfw or political

Pass an optional "prompt" to add your own moderation rules on top of the defaults:

    curl -H "x-api-key: $X_API_KEY" -H "content-type: application/json" \
         -d '{"text": "...", "prompt": "Also flag spam and scam links."}' \
         http://localhost:8000/api/v1/content/validate
"""
import asyncio
import hmac
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from contextlib import asynccontextmanager

from service import DEFAULT_MAX_NEW_TOKENS
from utils import (
    FAILED,
    enqueue,
    extract_audio,
    materialize_video,
    store,
    transcribe_and_check,
    unlink,
)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load at startup so the first request doesn't pay the model-load cost.
    store.start()
    yield
    store.stop()


app = FastAPI(title="Gemma 4 E2B video analysis", lifespan=lifespan)


def verify_api_key(x_api_key: str | None) -> None:
    expected_api_key = os.environ.get("X_API_KEY")
    if not expected_api_key:
        raise HTTPException(status_code=503, detail="X_API_KEY is not configured.")
    # compare_digest, not ==, so a wrong key can't be recovered by timing the reply.
    if not x_api_key or not hmac.compare_digest(x_api_key, expected_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key.")


@app.get("/health")
async def health() -> dict:
    return {"ok": store.model_id is not None, "model": store.model_id}


@app.post("/api/v1/video/analyze", status_code=200)
async def analyze_and_wait(
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    text: str | None = Form(None),
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
    x_api_key: str | None = Header(default=None),
) -> dict:
    """Describe a video, judge it and any accompanying text, transcribe its speech."""
    verify_api_key(x_api_key)

    video_path = await run_in_threadpool(materialize_video, file, url)
    # Rip the audio before queueing: the worker deletes the video file the moment
    # it finishes, which would race the extraction.
    audio_path = await run_in_threadpool(extract_audio, video_path)
    try:
        job = enqueue(video_path, max_new_tokens)
    except Exception:
        unlink(audio_path)
        raise

    # Issued together. STT genuinely overlaps the model (it's a call to another
    # machine); the text checks share this GPU, so they interleave with the video
    # rather than running alongside it.
    _, (transcription, transcription_safe), text_safe = await asyncio.gather(
        run_in_threadpool(job.done.wait),
        transcribe_and_check(audio_path),
        run_in_threadpool(store.text_is_safe, text),
    )
    if job.status == FAILED:
        raise HTTPException(status_code=500, detail=job.error)

    # Three sources feed one verdict: what the video shows, what the caller wrote,
    # and what was said aloud. Any of them flagged makes the whole thing unsafe.
    flags = job.result["safety"]
    video_safe = not (flags["nudity"] or flags["nsfw"] or flags["politics"])
    return {
        "description": job.result["description"],
        "duration": job.result["duration"],
        "transcription": transcription,
        "safety": video_safe and text_safe and transcription_safe,
    }


class ValidateRequest(BaseModel):
    text: str
    prompt: str | None = None


@app.post("/api/v1/content/validate", status_code=200)
async def validate_content(
    body: ValidateRequest,
    x_api_key: str | None = Header(default=None),
) -> dict:
    """Judge whether a piece of text is safe — no audio, no video.

    The lightweight counterpart to /analyze: the caller already has the content as
    text (a caption, a comment, an existing transcript), so there's nothing to
    transcribe. It's run through the same Gemma text moderation /analyze applies to
    captions. `safety` is false when the text is nsfw or political.

    Optional `prompt` adds caller-supplied moderation rules on top of the defaults
    (e.g. extra categories or wording to catch); omit it for the standard check.
    """
    verify_api_key(x_api_key)
    safe = await run_in_threadpool(store.text_is_safe, body.text, body.prompt)
    return {"safety": safe}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8000)),
    )
