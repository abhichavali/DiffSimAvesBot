"""The drive signal: a control .tg's Fourier series, and the ramp that eases it in.

``load_control_for_morphology`` reads the control file a design names (and checks
the two agree on the bird); ``eval_control`` is the series at time t and
``drive_command`` the ramped, direction-signed crank command built from it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from morphology.model import BIRDS
from morphology.tg_parser import load_morphology


# Per-crank harmonic keys are `a_<n>.<crank>` / `b_<n>.<crank>` (absolute rad/s, like a_n) with the crank
# names below, in runtime.crank.CRANK_ACTUATORS order. The flight keys belong to the free-flight policy
# (hover.policy) and are not part of the drive.
CRANKS = ("L_t1", "L_t4", "R_t1", "R_t4")
FLIGHT_KEYS = ("hover_attitude", "phase_lock", "closed_loop", "command_loop", "command_observer")


@dataclass(frozen=True)
class FourierControl:
    harmonics: int
    fundamental_frequency_hz: float
    a0: float
    a: tuple[float, ...]
    b: tuple[float, ...]
    bird: str = ""  # one of morphology.model.BIRDS; required in every control .tg
    # (a, b) per crank in CRANKS order when the file has per-crank keys; a crank without its own
    # key for a harmonic takes the shared a_n / b_n. None = every crank runs the shared series.
    per_crank: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...] | None = None

    @property
    def omega(self) -> float:
        return math.tau * self.fundamental_frequency_hz

    @property
    def period(self) -> float:
        return 1.0 / self.fundamental_frequency_hz


def control_file_path(tg_path: Path, morph) -> Path:
    """The control file ``morph`` (parsed from ``tg_path``) names, resolved next to the .tg."""
    path = Path(morph.motors.control_file)
    return path if path.is_absolute() else tg_path.parent / path


def load_control_for_morphology(tg_path: Path) -> FourierControl:
    morph = load_morphology(tg_path)
    path = control_file_path(tg_path, morph)
    control = load_control(path)
    if control.bird != morph.bird:
        raise ValueError(
            f"{tg_path} is tagged bird = {morph.bird} but its control {path} is tagged bird = {control.bird}"
        )
    return control


def load_control(path: Path) -> FourierControl:
    values: dict[str, float] = {}
    bird = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key == "bird":
            bird = value
            continue
        if key in FLIGHT_KEYS:
            continue
        values[key] = _parse_scalar(value)
    if bird not in BIRDS:
        raise ValueError(
            f"{path}: missing or unknown bird tag {bird!r}; put `bird = hummer` "
            "or `bird = eagle` at the top of the control .tg"
        )

    harmonics = int(values.get("harmonics", 0))
    if harmonics <= 0:
        raise ValueError(f"{path}: harmonics must be positive")
    frequency = float(values.get("fundamental_frequency", 0.0))
    if frequency <= 0:
        raise ValueError(f"{path}: fundamental_frequency must be positive")
    a = tuple(float(values.get(f"a_{i}", 0.0)) for i in range(1, harmonics + 1))
    b = tuple(float(values.get(f"b_{i}", 0.0)) for i in range(1, harmonics + 1))
    per_crank = None
    crank_keys = [k for k in values if "." in k]
    if crank_keys:
        valid = {f"{ab}_{i}.{c}" for ab in "ab" for i in range(1, harmonics + 1) for c in CRANKS}
        bad = sorted(set(crank_keys) - valid)
        if bad:
            raise ValueError(f"{path}: unknown per-crank keys {bad}; expected a_<n>.<crank> / b_<n>.<crank>, "
                             f"n <= harmonics, crank in {CRANKS}")
        per_crank = tuple(
            (tuple(float(values.get(f"a_{i}.{c}", a[i - 1])) for i in range(1, harmonics + 1)),
             tuple(float(values.get(f"b_{i}.{c}", b[i - 1])) for i in range(1, harmonics + 1)))
            for c in CRANKS)
    return FourierControl(
        harmonics=harmonics,
        fundamental_frequency_hz=frequency,
        a0=float(values.get("a0", 0.0)),
        a=a,
        b=b,
        bird=bird,
        per_crank=per_crank,
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


def eval_control_cranks(control: FourierControl, t: float) -> list[float]:
    """The series at ``t`` for each crank (CRANKS order): per-crank harmonics when the file has them."""
    if control.per_crank is None:
        return [eval_control(control, t)] * len(CRANKS)
    phase = control.omega * t
    return [control.a0 + sum(a_n * math.cos(n * phase) + b_n * math.sin(n * phase)
                             for n, (a_n, b_n) in enumerate(zip(a, b), start=1))
            for a, b in control.per_crank]


def drive_command(control: FourierControl, motor_direction: float, t: float,
                  t0: float = 0.0, ramp_seconds: float = 0.0) -> float | list[float]:
    """The crank velocity command at ``t``: the ramped, direction-signed Fourier drive.

    One command for every crank, or one per crank (CRANKS order) when the control has
    per-crank harmonics; PlateWingSim.step takes either.
    """
    k = ramp_factor(t, t0, ramp_seconds) * motor_direction
    if control.per_crank is None:
        return k * eval_control(control, t)
    return [k * w for w in eval_control_cranks(control, t)]


def _parse_scalar(text: str) -> float:
    match = re.match(
        r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(hz|rad/s)?$",
        text.strip(),
        re.I,
    )
    if not match:
        raise ValueError(f"Invalid control scalar: {text}")
    return float(match.group(1))
