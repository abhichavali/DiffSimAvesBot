"""The free-flight policy a control .tg carries: the open loop (per-crank Fourier drive + how to launch)
and, optionally, the closed loop (a per-wingbeat MLP next to it) or a command-following loop.

A flight control .tg is an ordinary control file (runtime.control_signal) plus:
    a_<n>.<crank> / b_<n>.<crank> = ...   per-crank harmonics, absolute rad/s (crank in L_t1 L_t4 R_t1 R_t4)
    hover_attitude = w x y z               release attitude of the bird (the hover tilt)
    phase_lock = 20                        crank encoder phase-lock gain [1/s] (hover.launch.Drive)
    closed_loop = mlp_closed_loop.npz      per-wingbeat MLP, path relative to this file (omit = open loop)
    command_loop = command_policy.npz      command-following servo (hover.command: HOVER / FWD / REV / UP / DOWN,
                                           flown live by hover.command.LiveCommand; opt/hover_cmd.py export)
    command_observer = command_observer.npz  its sensor-fed observer (hover.reduced lqg on pos,att,w; optional)
The crank speed a0 must equal 2 pi fundamental_frequency (the crank-locked clock).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from runtime.control_signal import CRANKS, FLIGHT_KEYS, FourierControl, load_control


@dataclass(frozen=True)
class FlightPolicy:
    control: FourierControl
    theta: np.ndarray          # hover.launch theta: [0 (speed = control a0), a depths (4,K), b depths (4,K)]
    attitude: np.ndarray       # wxyz release attitude
    phase_lock: float
    mlp: dict | None           # {"layers": [(W, b), ...], "obs_scale": (12,), "u_max": float} or None
    path: Path
    command: dict | None = None    # hover.command.load_policy of command_loop, or None
    observer: dict | None = None   # hover.reduced.load_controller of command_observer, or None

    @property
    def K(self) -> int:
        return self.control.harmonics

    def correction(self, s) -> np.ndarray:
        """Per-wingbeat correction to theta from the strobed deviation state (zeros when open loop)."""
        if self.mlp is None:
            return np.zeros_like(self.theta)
        x = np.asarray(s) * self.mlp["obs_scale"]
        *hidden, (W, b) = self.mlp["layers"]
        for Wh, bh in hidden:
            x = np.tanh(Wh @ x + bh)
        return np.concatenate([[0.0], self.mlp["u_max"] * np.tanh(W @ x + b)])


def _flight_values(path: Path) -> dict[str, str]:
    out = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            if key in FLIGHT_KEYS:
                out[key] = value
    return out


def has_flight_policy(path) -> bool:
    return "hover_attitude" in _flight_values(Path(path))


def has_command_loop(path) -> bool:
    """A flight control flown with live commands (command_loop): the viewer / GUI steer it."""
    fv = _flight_values(Path(path))
    return "hover_attitude" in fv and "command_loop" in fv


def load_mlp(path) -> dict:
    z = np.load(path)
    n = sum(1 for k in z.files if k.startswith("W"))
    return {"layers": [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)], "obs_scale": z["obs_scale"], "u_max": float(z["u_max"])}


def save_mlp(path, layers, obs_scale, u_max) -> None:
    arrays = {f"W{i}": np.asarray(W) for i, (W, _) in enumerate(layers)}
    arrays |= {f"b{i}": np.asarray(b) for i, (_, b) in enumerate(layers)}
    np.savez(path, obs_scale=np.asarray(obs_scale), u_max=np.asarray(u_max), **arrays)


def load_flight_policy(path) -> FlightPolicy:
    path = Path(path)
    control = load_control(path)
    fv = _flight_values(path)
    if "hover_attitude" not in fv:
        raise ValueError(f"{path} has no hover_attitude: not a flight policy")
    att = np.array([float(v) for v in re.split(r"[,\s]+", fv["hover_attitude"].strip("[] ")) if v])
    if att.shape != (4,):
        raise ValueError(f"{path}: hover_attitude needs 4 numbers (w x y z)")
    if abs(control.a0 - control.omega) > 1e-6 * abs(control.a0):
        raise ValueError(f"{path}: a0 ({control.a0}) must equal 2 pi fundamental_frequency ({control.omega:.6f})")
    per = control.per_crank or ((control.a, control.b),) * len(CRANKS)
    a = np.array([pa for pa, _ in per]) / control.a0
    b = np.array([pb for _, pb in per]) / control.a0
    mlp = load_mlp(path.parent / fv["closed_loop"]) if "closed_loop" in fv else None
    command = observer = None
    if "command_loop" in fv:
        from hover.command import load_policy          # hover.command -> hover.lqr -> this module
        command = load_policy(path.parent / fv["command_loop"])
    if "command_observer" in fv:
        from hover.reduced import load_controller
        observer = load_controller(path.parent / fv["command_observer"])
    return FlightPolicy(control, np.concatenate([[0.0], a.ravel(), b.ravel()]), att / np.linalg.norm(att),
                        float(re.match(r"[-+.\deE]+", fv.get("phase_lock", "20")).group()), mlp, path, command, observer)


def write_flight_control(path, *, a0, depth_a, depth_b, attitude, phase_lock=20.0, closed_loop=None,
                         bird="hummer", header="", command_loop=None, command_observer=None) -> None:
    """Write a flight control .tg: crank speed a0 [rad/s], per-crank harmonic DEPTHS (4, K) as fractions of a0."""
    depth_a, depth_b = np.asarray(depth_a), np.asarray(depth_b)
    K = depth_a.shape[1]
    lines = [f"# {line}" for line in header.strip().splitlines()] if header else []
    lines += [f"bird = {bird}", f"harmonics = {K}", f"fundamental_frequency = {a0 / (2 * math.pi):.10f} Hz",
              f"a0 = {a0:.10f}"]
    lines += [f"a_{k} = 0" for k in range(1, K + 1)] + [f"b_{k} = 0" for k in range(1, K + 1)]
    lines.append("# per-crank harmonics [rad/s] (the shared a_n/b_n above are unused by this file)")
    for i, c in enumerate(CRANKS):
        lines += [f"a_{k}.{c} = {a0 * depth_a[i, k - 1]:.10g}" for k in range(1, K + 1)]
        lines += [f"b_{k}.{c} = {a0 * depth_b[i, k - 1]:.10g}" for k in range(1, K + 1)]
    lines += ["# free-flight policy (hover.policy)",
              "hover_attitude = " + " ".join(f"{q:.12g}" for q in np.asarray(attitude) / np.linalg.norm(attitude)),
              f"phase_lock = {phase_lock:g}"]
    if closed_loop:
        lines.append(f"closed_loop = {closed_loop}")
    if command_loop:
        lines.append(f"command_loop = {command_loop}")
    if command_observer:
        lines.append(f"command_observer = {command_observer}")
    Path(path).write_text("\n".join(lines) + "\n")
