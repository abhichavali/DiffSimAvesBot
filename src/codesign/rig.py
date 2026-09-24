"""The hummer co-design rig: a compiled .tg as a settled CPU model plus the MJX configs its rollouts share.

setup() compiles the design, settles it (wing pitch, crank sync) and puts it on device as a Rig.
variant() re-puts the same model under another integrator or timestep for the robustness rollouts.
The tendon helpers pick out the tendons the genome tunes.
"""

from __future__ import annotations

import fcntl
import math
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from compile_tg import compile_tg
from morphology.model import require_bird
from morphology.tg_parser import load_morphology
from runtime.control_signal import load_control_for_morphology
from runtime.mjx import aero_runtime, rollout, wing_pitch
from runtime.sim import rest_state

ROOT = Path(__file__).resolve().parents[2]
INTEGRATORS = {"euler": 0, "rk4": 1, "implicit": 2, "implicitfast": 3}
WINGTIP_SITES = ("L_wing_aero_blade_04", "R_wing_aero_blade_04")  # L/R tip, full 3D stroke
DISK_BODIES = ("L_t1_disk", "L_t4_disk", "R_t1_disk", "R_t4_disk")


class Rig(NamedTuple):
    model: mujoco.MjModel  # CPU model, settled; variant() borrows it
    mx: object  # mjx.Model
    dx0: object  # mjx.Data at rest
    control: object  # the design's FourierControl
    drive: rollout.DriveConfig
    pitch: wing_pitch.WingPitchConfig
    aero: aero_runtime.AeroConfig
    tip_ids: jnp.ndarray  # [L_tip, R_tip] site ids
    tendons: tuple  # structural_tendon_pairs(...)
    motor_ids: jnp.ndarray
    morph: object  # the design's Morphology


def setup(tg_path: Path, scene_path: Path, *, aero_substep: int = 4, integrator: str = "euler",
          phase_lag: float | None = None, timestep_ms: float | None = None,
          lock_wings: bool = False) -> Rig:
    """Compile, build a calibrated CPU model/data and the device-side configs.

    ``aero_substep`` matches the CPU runtime (4): aero is recomputed every N physics
    steps and held in between, with the finite-difference window N*dt.

    ``integrator`` defaults to **euler**, the one integrator where MJX and CPU MuJoCo
    agree per step (implicitfast diverges between the two solvers, so designs
    optimized under it overfit MJX and fail CPU validation).
    """
    # Concurrent optimizer / EA processes share the default --scene (and the EA its
    # scene_ea_genome.xml): hold a lock across the write and the load, or one loads
    # another's half-written XML.
    with open(ROOT / "opt" / ".compile.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        compile_tg(tg_path, scene_path)
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.integrator = INTEGRATORS[integrator]
    if timestep_ms is not None:  # override the tg's dt (must precede put_model)
        model.opt.timestep = timestep_ms * 1e-3
    if lock_wings:
        symmetrize_wings(model)  # BEFORE MjData/put_model: both wings must be one design
    morph = load_morphology(tg_path)
    require_bird(morph, "hummer", tool="the codesign optimizer")
    data = rest_state(model, morph, phase_lag)

    control = load_control_for_morphology(tg_path)
    drive = rollout.build_drive_config(model, morph, control, ramp_duration=0.0)
    pitch = wing_pitch.build_wing_pitch_config(model, morph, data)
    aero = aero_runtime.build_aero_config(model, morph, substep=aero_substep)
    tip_ids = jnp.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
                           for s in WINGTIP_SITES], dtype=jnp.int32)
    return Rig(model, mjx.put_model(model), mjx.put_data(model, data), control, drive, pitch,
               aero, tip_ids, structural_tendon_pairs(model, individual=not lock_wings),
               motor_tendon_ids(model), morph)


def variant(rig: Rig, integrator: str, dt: float, aero_substep: int):
    """(mx, aero config) for ``rig.model`` under another integrator and timestep [s].

    The model itself is left as it was."""
    model = rig.model
    saved = model.opt.integrator, model.opt.timestep
    model.opt.integrator, model.opt.timestep = INTEGRATORS[integrator], dt
    try:
        return mjx.put_model(model), aero_runtime.build_aero_config(model, rig.morph, substep=aero_substep)
    finally:
        model.opt.integrator, model.opt.timestep = saved


def step_counts(period: float, dt: float, warmup_periods: float, sample_periods: float):
    """(steps per cycle, warmup steps, sample steps) for a rollout at timestep ``dt`` [s]."""
    return (max(1, round(period / dt)), math.ceil(warmup_periods * period / dt),
            math.ceil(sample_periods * period / dt))


def lr_tendon_pairs(model) -> dict[str, dict[str, int]]:
    """{tendon name without its L_/R_ prefix: {"L": id, "R": id}}, in tendon-id order."""
    pairs: dict[str, dict[str, int]] = {}
    for i in range(model.ntendon):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
        if n and n[:2] in ("L_", "R_"):
            pairs.setdefault(n[2:], {})[n[0]] = i
    return pairs


def symmetrize_wings(model) -> int:
    """Force L/R tendons onto one shared design (in place); returns the pair count.

    The genome mirrors L=R only as a *factor*; the .tg base it multiplies is not
    mirrored (hover-soft-it009p ships L_t1_to_t1_2 = 184 N/m against R's 341).
    Stiffness is a multiplicative knob (the genome is a log-factor), so pairs meet
    at the GEOMETRIC mean; rest length is a length, so arithmetic.
    """
    n_sym = 0
    for d in lr_tendon_pairs(model).values():
        if "L" not in d or "R" not in d:
            continue
        li, ri = d["L"], d["R"]
        k = math.sqrt(max(model.tendon_stiffness[li], 0.0) * max(model.tendon_stiffness[ri], 0.0))
        model.tendon_stiffness[li] = model.tendon_stiffness[ri] = k
        ls = 0.5 * (model.tendon_lengthspring[li] + model.tendon_lengthspring[ri])
        model.tendon_lengthspring[li] = model.tendon_lengthspring[ri] = ls
        n_sym += 1
    return n_sym


def structural_tendon_pairs(model, individual: bool = True):
    """Structural tendon indices + base stiffness/rest-length arrays.

    Returns (l_ids, r_ids, base_stiffness(ntendon,), base_lengthspring(ntendon,2)).
    ``individual=True`` (default) tunes EVERY structural tendon on its own: l_ids =
    all structural ids and r_ids = the same, so the mirror write is idempotent.
    ``individual=False`` keeps the 12 L=R mirror pairs.
    """
    l_ids, r_ids = [], []
    for suffix, d in lr_tendon_pairs(model).items():
        if "motor" not in suffix and "L" in d and "R" in d:
            l_ids.append(d["L"])
            r_ids.append(d["R"])
    if individual:
        l_ids = r_ids = l_ids + r_ids
    return (
        jnp.asarray(l_ids, dtype=jnp.int32),
        jnp.asarray(r_ids, dtype=jnp.int32),
        jnp.asarray(model.tendon_stiffness),
        jnp.asarray(model.tendon_lengthspring),
    )


def motor_tendon_ids(model):
    """The 4 motor-tendon indices (L/R x t1/t4); the genome tunes them with one shared
    stiffness and one shared pretension factor."""
    ids = [i for i in range(model.ntendon)
           if (n := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)) and "motor" in n]
    return jnp.asarray(ids, dtype=jnp.int32)


def flap_t1_mask(model, drive):
    """1 for the t1 ("blue", upstroke) flap actuators, 0 for t4: the two 180-deg-offset
    motors the anti-overlap metric compares."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i))
             for i in np.asarray(drive.flap_actuator_ids)]
    return jnp.asarray([1.0 if (n and "t1" in n) else 0.0 for n in names], dtype=jnp.float32)


def strokeplane_info(model, morph, max_rad: float) -> dict:
    """What the stroke-plane genes drive: the four crank-disk mount bodies, phased on the
    L_t1 crank hinge angle, around the compiled mount angles, bounded to +/- max_rad."""
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in DISK_BODIES]
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "L_t1_disk_hinge")
    assert min(ids) >= 0 and jid >= 0, "disk bodies / crank hinge missing from model"
    spec = morph.motors.disk
    return dict(qadr=int(model.jnt_qposadr[jid]),
                body_ids=jnp.asarray(ids, dtype=jnp.int32),
                tilt_signs=jnp.asarray([1.0, 1.0, -1.0, -1.0], dtype=jnp.float32),
                tilt0=float(spec.tilt_rad), pitch0=float(spec.pitch_rad), max_rad=max_rad)
