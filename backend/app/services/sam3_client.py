"""SAM3 HTTP client — calls the standalone SAM3 server."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..core.config import hf_repo_cache_dir

logger = logging.getLogger(__name__)

SAM3_PORT = 8002
SAM3_URL = f"http://127.0.0.1:{SAM3_PORT}"
_sam3_process: subprocess.Popen | None = None
_last_activity: float = 0.0
_watchdog_thread: threading.Thread | None = None
_watchdog_stop: threading.Event | None = None


def _bump_activity() -> None:
    global _last_activity
    _last_activity = time.monotonic()


def _ensure_watchdog() -> None:
    """Start idle watchdog if not already running. Works regardless of how SAM3 was started."""
    global _watchdog_thread, _watchdog_stop, _last_activity
    if _watchdog_thread is not None:
        return

    from app.core.config import settings

    _last_activity = time.monotonic()
    _watchdog_stop = threading.Event()

    def _loop():
        while not _watchdog_stop.is_set():
            _watchdog_stop.wait(timeout=30)
            if _watchdog_stop.is_set():
                return
            if not is_sam3_running():
                continue
            idle = time.monotonic() - _last_activity
            if idle >= settings.model_idle_timeout_seconds:
                logger.info("SAM3 idle for %.0fs, auto-unloading...", idle)
                stop_sam3_server()

    _watchdog_thread = threading.Thread(target=_loop, daemon=True, name="sam3-idle-watchdog")
    _watchdog_thread.start()
    logger.info("SAM3 idle watchdog started (timeout=%ds)", settings.model_idle_timeout_seconds)


def is_sam3_running() -> bool:
    import urllib.request

    try:
        resp = urllib.request.urlopen(f"{SAM3_URL}/health", timeout=2)
        return resp.status == 200
    except Exception:
        return False


def _is_sam3_loaded() -> bool:
    """Check both that the server is alive AND the model has finished loading."""
    import json
    import urllib.request

    try:
        resp = urllib.request.urlopen(f"{SAM3_URL}/health", timeout=2)
        if resp.status != 200:
            return False
        data = json.loads(resp.read())
        return data.get("status") == "loaded"
    except Exception:
        return False


def start_sam3_server() -> None:
    global _sam3_process
    if _is_sam3_loaded():
        return

    hf_token = os.environ.get("HF_TOKEN", "")
    # Only require HF_TOKEN if the model hasn't been cached yet
    model_cache = hf_repo_cache_dir("facebook/sam3")
    if not hf_token and not model_cache.exists():
        raise RuntimeError(
            "HF_TOKEN environment variable is not set and SAM3 model is not cached. "
            "Visit https://huggingface.co/facebook/sam3 to accept the license, "
            "then set HF_TOKEN with your access token."
        )

    server_script = Path(__file__).resolve().parent.parent.parent / "sam3_server.py"
    sam3_venv = Path(__file__).resolve().parent.parent.parent / "sam3-venv"
    venv_python = sam3_venv / "bin" / "python3"
    if venv_python.exists():
        python = str(venv_python)
    else:
        python = sys.executable
        logger.warning("sam3-venv not found at %s, falling back to %s", venv_python, python)

    env = os.environ.copy()
    env["HF_TOKEN"] = hf_token

    sam3_log = open(Path(__file__).resolve().parent.parent.parent / "sam3_server.log", "a")  # noqa: SIM115
    logger.info("Starting SAM3 server on port %d (logging to sam3_server.log)...", SAM3_PORT)
    _sam3_process = subprocess.Popen(
        [python, str(server_script), "--port", str(SAM3_PORT)],
        stdout=sam3_log,
        stderr=sam3_log,
        env=env,
    )
    # Wait for model to be fully loaded (not just server alive)
    for _ in range(60):
        time.sleep(1)
        if _is_sam3_loaded():
            logger.info("SAM3 server ready")
            return
    raise RuntimeError("SAM3 model did not load within 60s")


def stop_sam3_server() -> None:
    global _sam3_process, _watchdog_thread, _watchdog_stop
    if _watchdog_stop:
        _watchdog_stop.set()
        _watchdog_thread = None
        _watchdog_stop = None
    if _sam3_process:
        _sam3_process.terminate()
        _sam3_process.wait(timeout=5)
        _sam3_process = None
    elif is_sam3_running():
        # Process reference lost (e.g. after uvicorn restart) — kill via port
        import signal

        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{SAM3_PORT}"], capture_output=True, text=True
            )
            for pid_str in result.stdout.strip().split("\n"):
                if pid_str:
                    os.kill(int(pid_str), signal.SIGTERM)
        except Exception:
            logger.exception("Failed to kill SAM3 process on port %d", SAM3_PORT)


def segment_sam3(
    image_path: str | Path,
    text: str = "",
    segmentation: bool = True,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> list[dict]:
    """Call SAM3 server to detect + segment. Returns list of box dicts."""
    import urllib.request

    _bump_activity()
    _ensure_watchdog()

    if not _is_sam3_loaded():
        start_sam3_server()

    boundary = "----FormBoundary" + os.urandom(8).hex()
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    parts: list[bytes] = []

    # File part
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="image.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        + image_bytes
        + b"\r\n"
    )

    # Text part
    if text:
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="text"\r\n\r\n{text}\r\n'
            ).encode()
        )

    # Segmentation part
    if not segmentation:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="segmentation"\r\n\r\n'
            f"false\r\n".encode()
        )

    # Threshold part
    if threshold != 0.5:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="threshold"\r\n\r\n'
            f"{threshold}\r\n".encode()
        )

    # Mask threshold part
    if mask_threshold != 0.5:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="mask_threshold"\r\n\r\n'
            f"{mask_threshold}\r\n".encode()
        )

    body = b"".join(parts) + f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{SAM3_URL}/segment",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    import json

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    if "error" in data:
        raise RuntimeError(data["error"])
    return data.get("boxes", [])
