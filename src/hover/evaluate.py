"""Shared by opt/verify_hover.py, opt/push_sweep.py and opt/hover_mc.py: the policy / sensor CLI flags, turning
them into a (picklable) flight spec, and the per-flight summary written to JSON.

Policy selection (first match): --open-loop; --ctrl <hover.reduced controller npz>; --lqr <lqr.npz> with
--obs (a reduced controller designed on the fly, --mode sof|lqg); --lqr alone (the full-state LQR, as
before); the flight control's closed_loop (a state MLP, or a hover.reduced controller npz).
Sensors (hover.sensors) are used when --noise-scale or --att-mode is given or the policy reads non-state
channels; otherwise the policy reads the true strobed state exactly as before (bit-identical defaults).
"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hover import launch as L
from hover import reduced as Rd
from hover import sensors as S
from hover.policy import load_flight_policy

# verify_hover's push schedule: wingbeat -> rigid kick (dx[6:9] COM velocity, dx[9:12] angular velocity)
VERIFY_PUSHES = {30: np.r_[0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0],         # 0.2 m/s forward
                 60: np.r_[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],         # 1 rad/s roll rate
                 90: np.r_[0, 0, 0, 0, 0, 0, 0, 0.2, 0.1, 0, 0.5, 0]}      # sideways + up + pitch rate
# push_sweep's single pushes (at wingbeat 20 of 80)
SWEEP_PUSHES = [(f"{axis} {mag:g} {unit}", np.eye(12)[idx] * mag)
                for axis, idx, unit, mags in (("side vel", 7, "m/s", (0.2, 0.4, 0.6, 0.8)), ("fwd vel", 6, "m/s", (0.4, 0.8)),
                                              ("up vel", 8, "m/s", (0.3, 0.6)), ("roll rate", 9, "rad/s", (1, 2, 3, 4)),
                                              ("pitch rate", 10, "rad/s", (1, 2, 3, 4)), ("yaw rate", 11, "rad/s", (2, 4, 8)))
                for mag in mags]


def add_args(ap, *, dt_default=(0.13,)) -> None:
    g = ap.add_argument_group("policy / sensors (hover.evaluate)")
    g.add_argument("--open-loop", action="store_true", help="ignore the control's closed loop")
    g.add_argument("--lqr", type=Path, default=None, help="lqr.npz: fly its full-state K, or with --obs design a reduced controller")
    g.add_argument("--ctrl", type=Path, default=None, help="a hover.reduced controller npz (sof / lqg / state, optionally distilled)")
    g.add_argument("--obs", default=None, help="channel subset for --lqr (hover.sensors.parse_obs: pos,att,vel,w,gyro,acc, pos_z, ...)")
    g.add_argument("--mode", default="lqg", choices=("sof", "lqg"), help="reduced controller for --lqr --obs")
    g.add_argument("--process-scale", type=float, default=1.0, help="LQG process noise x hover.reduced.PROCESS_STD")
    g.add_argument("--noise-scale", type=float, default=None, help="sensor noise multiplier (enables the sensor model; 0 = noise-free)")
    g.add_argument("--pos-noise-scale", type=float, default=1.0, help="extra factor on the mocap position noise")
    g.add_argument("--pos-latency-ms", type=float, default=0.0, help="mocap latency")
    g.add_argument("--att-mode", default=None, choices=("filter", "truth"), help="attitude channel: on-board filter (default) or truth")
    g.add_argument("--seed", type=int, default=0, help="sensor noise seed")
    g.add_argument("--dt-ms", type=float, nargs="+", default=list(dt_default), help="timestep(s) (snapped per wingbeat)")
    g.add_argument("--json", type=Path, default=None, help="write a machine-readable summary here")


def spec_from_args(a, control) -> dict:
    """Picklable flight spec: {"policy": open|mlp|linear, "ctl": dict|None, "sensor": dict|None, "label": str}."""
    fp = load_flight_policy(control)
    noise = S.SensorNoise(scale=a.noise_scale if a.noise_scale is not None else 0.0, pos_scale=a.pos_noise_scale,
                          pos_latency_ms=a.pos_latency_ms)
    ctl, kind = None, "open"
    if a.open_loop:
        pass
    elif a.ctrl:
        ctl, kind = Rd.load_controller(a.ctrl), "linear"
    elif a.lqr and a.obs:
        dnoise = noise if a.noise_scale is not None else S.SensorNoise(scale=0.0, pos_latency_ms=a.pos_latency_ms)
        ctl = Rd.design(Rd.load_model(a.lqr), a.obs, a.mode, noise=dnoise, beat_s=2 * math.pi / fp.control.omega,
                        process_scale=a.process_scale)
        kind = "linear"
    elif a.lqr:
        ctl, kind = {"kind": "state", "obs": np.arange(12), "K": np.load(a.lqr)["K"], "u_max": Rd.U_MAX}, "linear"
    elif fp.mlp is not None:
        cl = fp.path.parent / _closed_loop_name(fp.path)
        ctl, kind = (Rd.load_controller(cl), "linear") if Rd.is_controller(cl) else (None, "mlp")
    obs = np.asarray(ctl["obs"]) if ctl is not None else np.arange(12)
    use_sensor = a.noise_scale is not None or a.att_mode is not None or obs.max(initial=0) >= 12
    sensor = {"noise": noise, "att_mode": a.att_mode or "filter"} if use_sensor else None
    label = {"open": "open loop", "mlp": "closed loop (state MLP)"}.get(kind) or \
        f"closed loop ({ctl['kind']} on {','.join(S.obs_names(obs))}{', MLP' if ctl.get('mlp') else ''})"
    return {"policy": kind, "ctl": ctl, "sensor": sensor, "label": label}


def _closed_loop_name(path) -> str:
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0]
        if line.split("=", 1)[0].strip() == "closed_loop":
            return line.split("=", 1)[1].strip()
    raise ValueError(f"{path} has no closed_loop")


class _StateMLP:
    """The flight control's state MLP reading y[:12] (the true-state channels) of the sensor vector."""

    def __init__(self, fp):
        self.fp = fp

    def __call__(self, y):
        return self.fp.correction(np.asarray(y)[:12])


def make_policy(spec, fp):
    """A fresh policy for one flight (observers carry state)."""
    if spec["policy"] == "open":
        return None
    if spec["policy"] == "mlp":
        return fp.correction if spec["sensor"] is None else _StateMLP(fp)
    return Rd.LinearController(spec["ctl"])


def make_sensor(spec, seed):
    if spec["sensor"] is None:
        return None
    return S.Sensors(spec["sensor"]["noise"], att_mode=spec["sensor"]["att_mode"], seed=seed)


_CACHE = {}


def load(tg, control, dt_ms=0.13):
    """(bird, released state, flight policy) for a design + flight control, cached per process and dt."""
    key = (str(tg), str(control), float(dt_ms))
    if key not in _CACHE:
        fp = load_flight_policy(control)
        bird = L.load_bird(tg, control=control, dt_ms=dt_ms)
        _CACHE[key] = (bird, L.release(bird, fp.theta, fp.attitude, fp.K, k_phi=fp.phase_lock), fp)
    return _CACHE[key]


def fly(spec, tg, control, dt_ms, n_beats, kicks=None, seed=0):
    """Fly one flight under a spec. Returns (true strobed states, sensor, policy)."""
    bird, rel, fp = load(tg, control, dt_ms)
    pol, sen = make_policy(spec, fp), make_sensor(spec, seed)
    tr = L.fly(bird, rel, fp.theta, n_beats, policy=pol, kicks=kicks, sensor=sen)
    return tr, sen, pol


def summarize(tr, n_beats, beat_s, sensor=None) -> dict:
    tilt = np.degrees(np.linalg.norm(tr[:, 3:5], axis=1))
    pos = np.linalg.norm(tr[:, :3], axis=1) * 1e3
    spd = np.linalg.norm(tr[:, 6:9], axis=1) * 1e3
    out = {"beats": len(tr) - 1, "flew_all": bool(len(tr) == n_beats + 1), "seconds": (len(tr) - 1) * beat_s,
           "tilt_max_deg": float(tilt.max()), "tilt_mean_deg": float(tilt.mean()),
           "tilt_last20_deg": float(tilt[-20:].mean()), "tilt_last10_deg": float(tilt[-10:].mean()),
           "pos_max_mm": float(pos.max()), "drift_final_mm": float(pos[-1]), "alt_final_mm": float(tr[-1, 2] * 1e3),
           "speed_last20_mm_s": float(spd[-20:].mean()), "yaw_final_deg": float(np.degrees(tr[-1, 5])),
           "tilt_every10_deg": np.round(tilt[::10], 2).tolist()}
    if sensor is not None and sensor.log:
        Y = np.array(sensor.log)[:len(tr)]
        e = np.degrees(Y[:, 3:6] - tr[:len(Y), 3:6])
        out.update(att_est_err_rms_deg=float(np.sqrt((e[:, :2] ** 2).sum(1).mean())),
                   yaw_est_err_final_deg=float(e[-1, 2]))
    return out


def config(a, spec) -> dict:
    s = spec["sensor"]
    return {"label": spec["label"], "policy": spec["policy"],
            "controller": None if spec["ctl"] is None else {"kind": str(spec["ctl"]["kind"]),
                                                            "obs": S.obs_names(spec["ctl"]["obs"]),
                                                            "n_non_imu": S.n_non_imu(spec["ctl"]["obs"]),
                                                            "info": spec["ctl"].get("info", {})},
            "sensor": None if s is None else {"att_mode": s["att_mode"], **asdict(s["noise"])},
            "seed": a.seed, "dt_ms": a.dt_ms, "tg": str(a.tg), "control": str(a.control),
            "lqr": str(a.lqr) if a.lqr else None, "ctrl": str(a.ctrl) if a.ctrl else None}
