from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from morphology.tg_parser import load_morphology


@dataclass(frozen=True)
class FourierControl:
    harmonics: int
    fundamental_frequency_hz: float
    a0: float
    a: tuple[float, ...]
    b: tuple[float, ...]

    @property
    def omega(self) -> float:
        return math.tau * self.fundamental_frequency_hz

    @property
    def period(self) -> float:
        return 1.0 / self.fundamental_frequency_hz


def load_control_for_morphology(tg_path: Path) -> FourierControl:
    morph = load_morphology(tg_path)
    control_path = Path(morph.motors.control_file)
    if not control_path.is_absolute():
        control_path = tg_path.parent / control_path
    return load_control(control_path)


def load_control(path: Path) -> FourierControl:
    values: dict[str, float] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        values[key] = _parse_scalar(value)

    harmonics = int(values.get("harmonics", 0))
    if harmonics <= 0:
        raise ValueError(f"{path}: harmonics must be positive")
    frequency = float(values.get("fundamental_frequency", 0.0))
    if frequency <= 0:
        raise ValueError(f"{path}: fundamental_frequency must be positive")
    a = tuple(float(values.get(f"a_{i}", 0.0)) for i in range(1, harmonics + 1))
    b = tuple(float(values.get(f"b_{i}", 0.0)) for i in range(1, harmonics + 1))
    return FourierControl(
        harmonics=harmonics,
        fundamental_frequency_hz=frequency,
        a0=float(values.get("a0", 0.0)),
        a=a,
        b=b,
    )


def ramp_factor(t: float, t0: float, ramp_seconds: float) -> float:
    """Linear 0->1 drive ramp over ``ramp_seconds`` from ``t0`` (1.0 after, or if ramp<=0).

    The motors are velocity servos: commanding full a0 onto the near-zero-inertia
    crank disks from rest is a step input that spikes qacc ~6e5 and can diverge the
    0.13 ms step. Ramping the command over the warmup kills that startup transient.
    """
    if ramp_seconds <= 0:
        return 1.0
    return min(1.0, max(0.0, (t - t0) / ramp_seconds))


def eval_control(control: FourierControl, t: float) -> float:
    phase = control.omega * t
    value = control.a0
    for n, (a_n, b_n) in enumerate(zip(control.a, control.b), start=1):
        value += a_n * math.cos(n * phase) + b_n * math.sin(n * phase)
    return value


def _parse_scalar(text: str) -> float:
    match = re.match(
        r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(hz|rad/s)?$",
        text.strip(),
        re.I,
    )
    if not match:
        raise ValueError(f"Invalid control scalar: {text}")
    value = float(match.group(1))
    unit = (match.group(2) or "").lower()
    return value if unit in ("", "hz") else value
