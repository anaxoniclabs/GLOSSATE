# SPDX-License-Identifier: MIT
"""Device detection for inference. Priority: MPS > CUDA > CPU."""

import platform
import subprocess
import sys
from typing import Optional


def _mps_available() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def get_optimal_device(override: Optional[str] = None) -> str:
    """Return the best available device for inference.

    Args:
        override: ``None`` or ``"auto"`` selects automatically.
            ``"mps"``, ``"cuda"``, or ``"cpu"`` forces that device.

    Raises:
        ValueError: if the requested device is unavailable.
    """
    if override and override != "auto":
        choice = override.lower()
        if choice == "mps":
            if not _mps_available():
                raise ValueError("Requested --device mps but MPS is not available.")
            return "mps"
        if choice == "cuda":
            if not _cuda_available():
                raise ValueError("Requested --device cuda but no CUDA device is available.")
            return "cuda"
        if choice == "cpu":
            return "cpu"
        raise ValueError(f"Unknown device override: {override!r}")

    if _mps_available():
        return "mps"
    if _cuda_available():
        return "cuda"
    return "cpu"


def _cuda_memory_info() -> dict | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return {
            "device": "cuda",
            "free_gb": free / (1024**3),
            "total_gb": total / (1024**3),
            "allocated_gb": allocated / (1024**3),
            "reserved_gb": reserved / (1024**3),
        }
    except Exception:
        return None


def _linux_memory_info() -> dict | None:
    try:
        values: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0])
        available = values.get("MemAvailable", values.get("MemFree", 0))
        total = values.get("MemTotal", 0)
        return {
            "device": "system",
            "free_gb": available / (1024**2),
            "total_gb": total / (1024**2),
        }
    except Exception:
        return None


def _macos_memory_info() -> dict:
    """Return free/active/wired memory in GB via vm_stat."""
    import os

    result = subprocess.run(["vm_stat"], capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")
    stats: dict[str, int] = {}
    for line in lines[1:]:
        if ":" in line:
            parts = line.split(":")
            key = parts[0].strip()
            value = parts[1].strip().rstrip(".")
            try:
                stats[key] = int(value)
            except ValueError:
                continue

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        page_size = 16384

    free_pages = stats.get("Pages free", 0)
    speculative_pages = stats.get("Pages speculative", 0)
    active_pages = stats.get("Pages active", 0)
    inactive_pages = stats.get("Pages inactive", 0)
    wired_pages = stats.get("Pages wired down", 0)

    return {
        "device": "system",
        "free_gb": ((free_pages + speculative_pages) * page_size) / (1024**3),
        "active_gb": (active_pages * page_size) / (1024**3),
        "inactive_gb": (inactive_pages * page_size) / (1024**3),
        "wired_gb": (wired_pages * page_size) / (1024**3),
    }


def get_memory_info(device: Optional[str] = None) -> dict:
    """Return runtime memory information.

    CUDA returns GPU memory. macOS returns ``vm_stat`` system memory.
    Linux without CUDA returns ``/proc/meminfo`` system memory.
    """
    try:
        if device == "cuda" or (device in (None, "auto") and _cuda_available()):
            cuda = _cuda_memory_info()
            if cuda is not None:
                return cuda
        if sys.platform == "linux":
            linux = _linux_memory_info()
            if linux is not None:
                return linux
        return _macos_memory_info()
    except Exception as e:
        return {"error": str(e)}
