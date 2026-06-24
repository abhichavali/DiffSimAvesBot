"""Explicit MuJoCo model teardown — Python GC does not reliably free C heap."""

from __future__ import annotations

import gc
from typing import Any


def release_mujoco(model: Any | None, data: Any | None = None) -> None:
    """Drop MuJoCo handles and encourage the allocator to return memory.

    MjData holds a strong reference to its MjModel via ``data.model``; delete
    data before the model. Python's GC often lags on C-extension destructors,
    so we collect twice after each release.
    """
    if data is not None:
        del data
    if model is not None:
        del model
    gc.collect()
    gc.collect()
