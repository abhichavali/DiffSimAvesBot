"""Spot-check a codesign genome's peak-torque transfer across integrators/timesteps.

    uv run python opt/eval_transfer.py [genome.json]  (default: current EA elite)
"""
import json, math, sys
from pathlib import Path

import jax, jax.numpy as jnp
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optimize_control as oc

ROOT = Path(__file__).resolve().parents[1]
K = 5


def eval_one(g, tg, integrator, dt):
    scene = ROOT / f"opt/xfer_{integrator}_{int(dt*1000)}.xml"
    model, mx, dx0, control, drive, pitch, aero, tip_ids, tendons, motor_ids = oc.setup(
        tg, scene, aero_substep=2, integrator=integrator, timestep_ms=dt)
    period = control.period
    steps = model.opt.timestep
    spc = max(1, round(period / steps))
    flap_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i))
                  for i in np.asarray(drive.flap_actuator_ids)]
    mask = jnp.asarray([1.0 if (n and "t1" in n) else 0.0 for n in flap_names], jnp.float32)
    metrics = oc.build_rollout(
        mx, dx0, drive, pitch, aero, tip_ids, float(control.omega), K,
        math.ceil(8.0 * period / steps), math.ceil(3.0 * period / steps),
        steps_per_cycle=spc, aero_substep=2, base_a0=float(drive.a0),
        tune_control=True, control_fixed=jnp.zeros(2 * K + 1, jnp.float32),
        tune_frequency=False, tune_tendons=True, tendons=tendons,
        tune_motor=True, motor_ids=motor_ids, flap_t1_mask=mask)
    m = np.asarray(jax.jit(metrics)(jnp.asarray(g, jnp.float32)))
    return float(m[1]) * 1e3, float(m[0]) * 1e3  # tau mNm, lift mN


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "opt/codesign_validity/seeds/best_genome.json"
    d = json.loads(src.read_text())
    g = d["genome"]
    print(f"[genome] {src} gen={d.get('gen')} fit={d.get('fitness')}", flush=True)
    taus = {}
    for integ, dt in [("euler", 0.06), ("implicitfast", 0.06), ("rk4", 0.06),
                      ("euler", 0.03), ("euler", 0.13)]:
        tau, lift = eval_one(g, ROOT / "flight-orientations/hovering/shoulder.tg", integ, dt)
        taus[(integ, dt)] = tau
        print(f"  {integ:>12s} dt={dt:g}ms  tau={tau:6.1f} mNm  lift={lift:6.1f} mN", flush=True)
    vals = list(taus.values())
    print(f"[spread] max/min = {max(vals)/min(vals):.3f}  ({max(vals)-min(vals):.1f} mNm)", flush=True)


if __name__ == "__main__":
    main()
