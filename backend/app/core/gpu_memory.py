"""GPU memory management — Strategy Pattern.

Centralizes all device-specific memory operations (CUDA / MPS / CPU)
so service code never needs ``if cuda ... elif mps ...`` branches.
"""

from __future__ import annotations

import abc
import gc
import logging
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


# ── Abstract strategy ─────────────────────────────────


class GPUMemoryManager(abc.ABC):
    """Device-agnostic interface for GPU memory management."""

    @property
    @abc.abstractmethod
    def device(self) -> str:
        """Return the torch device string (e.g. 'cuda', 'mps', 'cpu')."""

    @abc.abstractmethod
    def empty_cache(self) -> None:
        """Lightweight cache release — call *before* inference to reclaim
        fragmented memory without a full GC cycle."""

    @abc.abstractmethod
    def full_cleanup(self) -> None:
        """Heavy cleanup — call *after* inference.

        Includes ``gc.collect()``, device synchronization, cache eviction,
        and any IPC collection specific to the backend.
        """

    @abc.abstractmethod
    def resolve_max_long_side(self) -> int:
        """Return the maximum image long-side (px) appropriate for this GPU."""

    @abc.abstractmethod
    def resolve_attn_impl(self) -> str:
        """Return the best attention implementation name for this device."""


# ── Concrete strategies ───────────────────────────────


class CUDAMemoryManager(GPUMemoryManager):
    """NVIDIA CUDA backend."""

    @property
    def device(self) -> str:
        return "cuda"

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def full_cleanup(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def resolve_max_long_side(self) -> int:
        total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        if total_mb >= 16 * 1024:
            return 1333
        if total_mb >= 12 * 1024:
            return 1024
        if total_mb >= 10 * 1024:
            return 800
        return 640  # preserve small-object detail on 8 GB cards

    def resolve_attn_impl(self) -> str:
        try:
            import flash_attn  # type: ignore[import-not-found]  # noqa: F401

            logger.info("flash_attention_2 detected — using it for attention")
            return "flash_attention_2"
        except ImportError:
            logger.info("flash-attn not installed, falling back to sdpa")
        return "sdpa"


class MPSMemoryManager(GPUMemoryManager):
    """Apple Silicon MPS backend."""

    @property
    def device(self) -> str:
        return "mps"

    def empty_cache(self) -> None:
        torch.mps.empty_cache()

    def full_cleanup(self) -> None:
        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()

    def resolve_max_long_side(self) -> int:
        return 1024  # Apple Silicon unified memory is ample

    def resolve_attn_impl(self) -> str:
        return "sdpa"


class CPUMemoryManager(GPUMemoryManager):
    """CPU-only fallback (no GPU cache to manage)."""

    @property
    def device(self) -> str:
        return "cpu"

    def empty_cache(self) -> None:
        pass

    def full_cleanup(self) -> None:
        gc.collect()

    def resolve_max_long_side(self) -> int:
        return 800

    def resolve_attn_impl(self) -> str:
        return "sdpa"


# ── Factory / singleton ───────────────────────────────


def _detect_device(override: str = "") -> str:
    """Resolve the best available device string."""
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError(
        "No GPU detected. LocateAnything-3B requires CUDA (NVIDIA GPU) "
        "or MPS (Apple Silicon). CPU inference is not supported."
    )


def validate_vlm_device(device: str) -> None:
    """Fail fast only for unsupported devices, without imposing a VRAM cutoff."""
    if device == "cpu":
        raise RuntimeError(
            "LocateAnything-3B cannot run on CPU in this application. "
            "Expose a CUDA GPU to the container or use Apple Silicon MPS in manual setup."
        )


def create_memory_manager(device: str) -> GPUMemoryManager:
    """Create the right strategy for *device*."""
    if device == "cuda":
        return CUDAMemoryManager()
    if device == "mps":
        return MPSMemoryManager()
    return CPUMemoryManager()


@lru_cache(maxsize=1)
def get_memory_manager() -> GPUMemoryManager:
    """Module-level singleton — lazily created on first call.

    Uses ``settings.device`` (user override) or auto-detects.
    """
    from .config import settings

    device = _detect_device(settings.device)
    mgr = create_memory_manager(device)
    logger.info("GPUMemoryManager initialised: %s (%s)", type(mgr).__name__, mgr.device)
    return mgr
