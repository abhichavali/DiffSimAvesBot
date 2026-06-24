"""Morphology compiler: typed object model → MuJoCo MJCF."""

from morphology.model import Morphology

__all__ = ["Morphology", "compile_morphology"]


def __getattr__(name: str):
    if name == "compile_morphology":
        from morphology.compile import compile_morphology
        return compile_morphology
    raise AttributeError(name)
