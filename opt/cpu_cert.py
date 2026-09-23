"""CPU transfer check for the reward arms: seed vs preharmonics(it010) vs final.

No MJX at all -- pure numpy rollouts of the same compiled scene, at the deploy dt
and a holdout dt. This is the certification that decides whether a harmonic gain
found in MJX is real or a discretization artifact.
"""
import json, math, sys
from pathlib import Path
ROOT = Path("/workspace/DiffSimAvesBot")
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
from opt.optimize_control import setup, cpu_validate_design, load_control_seed

K = 3
TG = ROOT / "flight-orientations" / "hovering-soft-it009p" / "shoulder.tg"
SCENE = ROOT / "scene.xml"
ARMS = sys.argv[1:] or ["swing", "swing_rms", "power_crest", "tau_rms", "overlap", "lift_ripple"]
model, mx, dx0, control, drive, *_ = setup(TG, SCENE, aero_substep=1, integrator="euler",
                                           timestep_ms=0.06, lock_wings=True)
omega, base_a0 = float(control.omega), float(drive.a0)
ta = (np.asarray(model.tendon_stiffness), np.asarray(model.tendon_lengthspring))

def roll(theta, dt_ms):
    t = np.asarray(theta, dtype=float)
    cp = (base_a0 * math.exp(t[0]), t[1:1+K], t[1+K:], omega * math.exp(t[0]))
    L, T, P, pc, sw = cpu_validate_design(SCENE, cp, ta, "euler", warmup=8.0, periods=6.0,
                                          phase_lag=math.radians(6), timestep_ms=dt_ms,
                                          aero_substep=1, cycle_stats=True, with_swing=True)
    return {"lift": L, "tau_peak": T, "power": P, "swing": sw,
            "drift": float(max(pc) - min(pc))}

def theta_of(p):
    f = Path(p) / "theta_full.json"
    return json.loads(f.read_text())["theta"][:2*K+1] if f.exists() else None

DTS = (0.06, 0.03, 0.02)
seed = [float(x) for x in load_control_seed(None, K, drive, base_a0)]
out = {}
for dt in DTS:
    out[f"seed@{dt}"] = roll(seed, dt)
    print(f"seed      dt={dt:g}  {out[f'seed@{dt}']}", flush=True)
for arm in ARMS:
    run = ROOT / "opt" / f"runs_{arm}"
    cps = sorted((run / "checkpoints").glob("it*"))
    for label, p in (("it010", run / "preharmonics"), ("final", cps[-1] if cps else run)):
        th = theta_of(p)
        if th is None:
            print(f"{arm:12s} {label}: MISSING", flush=True); continue
        for dt in DTS:
            r = roll(th, dt)
            out[f"{arm}/{label}@{dt}"] = r
            b = out[f"seed@{dt}"]
            print(f"{arm:12s} {label:6s} dt={dt:g}  lift={r['lift']*1e3:6.1f}mN "
                  f"({(r['lift']/b['lift']-1)*100:+5.2f}%)  peak={r['tau_peak']*1e3:6.1f}mNm "
                  f"({(r['tau_peak']/b['tau_peak']-1)*100:+5.2f}%)  P={r['power']:.3f}W "
                  f"({(r['power']/b['power']-1)*100:+5.2f}%)  "
                  f"SWING={r['swing']*1e3:6.2f}mNm ({(r['swing']/b['swing']-1)*100:+6.2f}%)  "
                  f"a0x={math.exp(th[0]):.4f}",
                  flush=True)
        # harmonic energy actually used, as a fraction of a0
        print(f"{arm:12s} {label:6s} harmonics |a|,|b| = "
              f"{np.round(np.asarray(th[1:]), 5).tolist()}", flush=True)
(ROOT / "opt" / "cpu_cert.json").write_text(json.dumps(out, indent=2))
print("wrote opt/cpu_cert.json")
