from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings

# ── CUDA memory allocator tuning ───────────────────
# Use expandable segments (PyTorch 2.1+) for dynamic memory management;
# much better than the legacy max_split_size_mb for varying tensor sizes
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # Database
    database_url: str = ""

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        # Fallback to local SQLite if no DATABASE_URL is provided in env
        db_path = self.project_root / "autolabeling.db"
        return f"sqlite:///{db_path}"

    # Model (auto-detect: cuda → mps; CPU is not supported)
    model_dir: str = ""
    device: str = ""
    model_id: str = "nvidia/LocateAnything-3B"
    sam2_model_id: str = "facebook/sam2.1-hiera-base-plus"
    sam2_checkpoint_path: str = ""

    @property
    def resolved_device(self) -> str:
        from .gpu_memory import _detect_device

        return _detect_device(self.device)

    # Model idle timeout (seconds) — auto-unload to free GPU memory
    model_idle_timeout_seconds: int = 600

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Upload
    max_upload_size_mb: int = 20
    max_import_size_mb: int = 10240  # 10 GB
    max_video_upload_size_mb: int = 500

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @property
    def upload_dir(self) -> Path:
        d = self.project_root / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def resolved_model_dir(self) -> str:
        return self.model_dir or str(self.project_root / "model")


settings = Settings()


def hf_repo_cache_dir(model_id: str) -> Path:
    """Local cache directory for a Hugging Face repo id.

    Honors HF_HUB_CACHE first, then HF_HOME (the hub lives at <HF_HOME>/hub), then
    the huggingface_hub default. Reads os.environ directly so it always agrees with
    what transformers/huggingface_hub themselves see — backend/.env is only read by
    pydantic-settings and is never exported to the process environment.
    """
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        base = Path(hub_cache)
    else:
        base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return base / f"models--{model_id.replace('/', '--')}"
