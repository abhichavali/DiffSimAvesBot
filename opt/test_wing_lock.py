"""Self-check for --lock-wings (run: uv run python opt/test_wing_lock.py [design.tg]).

One command drives all four crank motors, so the CONTROL signal was always shared.
The morphology was not: `structural_tendon_pairs(individual=True)` gives each of the
24 structural tendons its own (stiffness, pretension) gene, and the seed .tg bakes
wildly different L/R values on top (hover-soft-it009p: L_t1_to_t1_2 = 184 N/m vs
R = 341 N/m). So past runs optimized two DIFFERENT wings sharing one command.

--lock-wings geometric-means each L/R pair at load and tunes the 12 mirrored pairs.
This check asserts, on CPU physics with aero in the loop:
  1. the seed really is asymmetric without the flag (else the flag is pointless),
  2. with it, every L/R tendon pair is identical and the genome halves (48 -> 24),
  3. the two wingtips trace mirror-identical paths and the two shoulders carry
     equal motor torque.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np

from opt.optimize_control import ROOT, WINGTIP_SITES, setup

TG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "designs" / "hummer" / "hover-soft-it009p" / "shoulder.tg"
SCENE = ROOT / "scene.xml"
CYCLES = 3.0


def tendon_pairs(model):
    """{suffix: (l_id, r_id)} for the mirrored structural/motor tendons."""
    seen: dict[str, dict[str, int]] = {}
    for i in range(model.ntendon):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
        if n and n[:2] in ("L_", "R_"):
            seen.setdefault(n[2:], {})[n[0]] = i
    return {k: (v["L"], v["R"]) for k, v in seen.items() if "L" in v and "R" in v}


def roll(model, control, cycles=CYCLES):
    """Drive the shared crank command with aero+pitch live; return per-wing traces."""
    from runtime.aero_runtime import AeroBladeRuntime
    from runtime.crank_sync import sync_crank_disks_at_rest
    from runtime.viewer_runtime import get_actuator_id
    from runtime.wing_pitch_limit import WingPitchLimiter
    from runtime.morphology_runtime import resolve_morphology_path
    from morphology.tg_parser import load_morphology

    data = mujoco.MjData(model)
    from opt.optimize_control import calibrate_wing_pitch_at_rest
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data, load_morphology(resolve_morphology_path()).motors.crank_phase_lag_rad)
    aero = AeroBladeRuntime(model, substep=4)
    pitch = WingPitchLimiter(model, data)
    aids = [get_actuator_id(model, n) for n in ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")]
    tips = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in WINGTIP_SITES]
    a0, a, b = control.a0, np.asarray(control.a), np.asarray(control.b)
    harm = np.arange(1, len(a) + 1)
    omega = control.omega

    tip_tr, tau_tr = [], []
    for _ in range(int(cycles * (math.tau / omega) / model.opt.timestep)):
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        aero.apply(model, data)
        pitch.apply(model, data)
        phase = omega * data.time
        cmd = a0 * (1.0 + float(np.sum(a * np.cos(harm * phase) + b * np.sin(harm * phase))) / a0)
        for aid in aids:
            data.ctrl[aid] = cmd
        mujoco.mj_step(model, data)
        tip_tr.append(data.site_xpos[tips].copy())            # (2, xyz) = [L, R]
        tau_tr.append(np.abs(data.actuator_force[aids]).copy())  # [Lt1, Lt4, Rt1, Rt4]
    return np.asarray(tip_tr), np.asarray(tau_tr)


def asymmetry(tip_tr):
    """Peak |L - mirror(R)| over the run, as a fraction of the stroke extent [both m]."""
    mirror = tip_tr[:, 1, :] * np.array([1.0, -1.0, 1.0])     # R reflected through y=0
    err = np.max(np.linalg.norm(tip_tr[:, 0, :] - mirror, axis=1))
    extent = float(np.max(np.ptp(tip_tr[:, 0, :], axis=0)))
    return err, err / max(extent, 1e-9)


def main():
    print(f"design: {TG}")
    free_model, *_, free_tendons, _ = setup(TG, SCENE, lock_wings=False)
    lock_model, _, _, control, *_, lock_tendons, _ = setup(TG, SCENE, lock_wings=True)

    n_free, n_lock = int(np.asarray(free_tendons[0]).shape[0]), int(np.asarray(lock_tendons[0]).shape[0])
    assert n_lock * 2 == n_free, f"lock must halve the tendon genome: {n_free} -> {n_lock}"

    pairs = tendon_pairs(free_model)
    spread = max(abs(free_model.tendon_stiffness[l] - free_model.tendon_stiffness[r])
                 / max(free_model.tendon_stiffness[l], 1e-9) for l, r in pairs.values())
    assert spread > 0.05, (
        f"seed already symmetric ({spread*100:.1f}% max L/R stiffness spread) -- "
        "nothing for --lock-wings to fix, so this check proves nothing")
    locked = max(abs(lock_model.tendon_stiffness[l] - lock_model.tendon_stiffness[r])
                 for l, r in tendon_pairs(lock_model).values())
    assert locked < 1e-9, f"locked model still has L/R stiffness spread: {locked}"

    free_tip, free_tau = roll(free_model, control)
    lock_tip, lock_tau = roll(lock_model, control)
    free_err, free_rel = asymmetry(free_tip)
    lock_err, lock_rel = asymmetry(lock_tip)
    free_dtau = abs(free_tau[:, :2].max() - free_tau[:, 2:].max()) / max(free_tau.max(), 1e-9)
    lock_dtau = abs(lock_tau[:, :2].max() - lock_tau[:, 2:].max()) / max(lock_tau.max(), 1e-9)

    # The wings are the same machine now, so the tips trace the SAME arc -- bit for
    # bit, not approximately (the mirror is exact in the model, so any drift means a
    # real asymmetry crept back in). 1 um on a 300 mm stroke is the noise gate.
    assert lock_err < 1e-6, f"locked wings not mirror-identical: {lock_err*1e3:.4f} mm"
    assert lock_dtau < 1e-9, f"locked shoulders carry different torque: {lock_dtau*100:.4f}%"
    assert lock_rel < free_rel, "locking must reduce asymmetry, not raise it"

    print(f"ok: genome {n_free*2} -> {n_lock*2} params; seed L/R stiffness spread {spread*100:.0f}%\n"
          f"    unlocked: tip mismatch {free_err*1e3:.2f} mm ({free_rel*100:.2f}% of stroke), "
          f"peak-torque mismatch {free_dtau*100:.2f}%\n"
          f"    locked:   tip mismatch {lock_err*1e3:.4f} mm ({lock_rel*100:.4f}% of stroke), "
          f"peak-torque mismatch {lock_dtau*100:.4f}%")


if __name__ == "__main__":
    main()
