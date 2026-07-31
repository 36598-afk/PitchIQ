"""
runpod_handler.py — Wraps infer_pitch71.infer() as a RunPod Serverless
GPU endpoint.

Design: the web server (Railway/Render) never sends the video itself in the
request — videos go straight to object storage (Backblaze/R2) first, and
this handler just receives a URL to download from. That keeps the request
payload tiny regardless of video size, and matches how the rest of the
backend is designed (Phase 2/3 of the migration plan).

Expected input (what RunPod passes to handler()):
    {
        "input": {
            "video_url": "https://...",   # required — a downloadable URL
                                            # (e.g. a presigned R2/B2 link)
            "pitch_id": "abc123"           # optional — passed straight
                                            # through in the response, so
                                            # the backend can match results
                                            # to the right database row
        }
    }

Returns: the same dict infer_pitch71.infer() returns, plus "pitch_id" and
an "error" field (None on success) so the backend can distinguish a real
failure from a normal "no valid flight path found" result.
"""

import os
import tempfile
import traceback

import requests
import runpod

from infer_pitch71 import infer


def _download_video(url, suffix=".mov"):
    """Downloads the video to a local temp file and returns its path.
    Streams to disk rather than loading into memory, since clips can be
    tens of MB."""
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


def handler(event):
    inp = event.get("input", {})
    video_url = inp.get("video_url")
    pitch_id = inp.get("pitch_id")

    if not video_url:
        return {"error": "missing required field: video_url", "pitch_id": pitch_id}

    local_path = None
    try:
        local_path = _download_video(video_url)
        result = infer(local_path, debug=False)
        if result is None:
            return {"error": "inference returned no result (likely no audio track)",
                    "pitch_id": pitch_id}
        result["pitch_id"] = pitch_id
        result["error"] = None
        return result
    except Exception as e:
        # Always return a structured error rather than letting RunPod
        # surface a bare traceback — the backend needs a predictable shape
        # to handle failures gracefully (e.g. mark the pitch as failed
        # instead of leaving it stuck "processing" forever).
        return {
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "pitch_id": pitch_id,
        }
    finally:
        if local_path and os.path.exists(local_path):
            os.unlink(local_path)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
