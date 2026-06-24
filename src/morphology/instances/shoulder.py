"""Bi-shoulder machine — loads the authoritative ``shoulder/shoulder.tg`` design."""

from __future__ import annotations

import os

from morphology.model import Morphology
from morphology.tg_parser import load_morphology

# Repo root: src/morphology/instances/ → ../../../
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SHOULDER_TG = os.path.join(ROOT, "design", "shoulder.tg")


def shoulder_morphology(tg_path: str | None = None) -> Morphology:
    """Load bi-shoulder morphology from a `.tg` file (default: ``shoulder/shoulder.tg``)."""
    return load_morphology(tg_path or SHOULDER_TG)
