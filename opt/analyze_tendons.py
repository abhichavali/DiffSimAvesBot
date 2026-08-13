"""Which tendon SETS work best at ~10 Hz: structure stats from the wide sweep.

Regenerates the sweep's deterministic genome pool (same rng seed/order), joins to
opt/wide_sweep_out/all_rows.json, slices designs whose crank speed is ~baseline
(|a0_log| <= 0.1 => actual ~10 Hz flap), lift-feasible, and contrasts per-tendon
stiffness between the lowest-torque quartile and the rest.
"""
import json, math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import optimize_control as oc

ROOT = Path(__file__).resolve().parents[1]
K = 5
LIFT_FLOOR = 0.347


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="opt/wide_sweep_out")
    ap.add_argument("--freqs", type=float, nargs="+", default=[8.0, 10.0, 12.0, 14.0, 16.0])
    ap.add_argument("--n", type=int, default=1536)
    ap.add_argument("--tendon-only", action="store_true")
    a = ap.parse_args()
    rows = json.loads((ROOT / a.out / "all_rows.json").read_text())
    freqs, n, seed = a.freqs, a.n, 0
    # tendon names in genome order
    model = mujoco.MjModel.from_xml_path(str(ROOT / "opt/ws_100.xml"))
    l_ids, r_ids, *_ = oc.structural_tendon_pairs(model)
    l_ids = np.asarray(l_ids)
    tnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, int(i)) for i in np.asarray(l_ids)]
    base_stiff = model.tendon_stiffness[l_ids].copy()
    n_tend = len(tnames)

    rng = np.random.default_rng(seed)
    G = np.concatenate([__import__("wide_sweep").sample_genomes(rng, n, n_tend, tendon_only=a.tendon_only)
                        for _ in freqs], 0)
    if len(rows) != len(G):
        print(f"WARNING: rows {len(rows)} != genomes {len(G)}; non-finite rollouts were dropped "
              f"-- falling back to top50-only analysis is NOT implemented; aborting.")
        return
    L = np.array([r["L"] for r in rows]); tau = np.array([r["tau"] for r in rows])
    a0 = G[:, 0]
    sel = (np.abs(a0) <= 0.1) & (L >= LIFT_FLOOR)
    idx = np.where(sel)[0]
    print(f"{len(idx)} lift-feasible ~10Hz designs of {int((np.abs(a0)<=0.1).sum())} sampled at ~10Hz")
    if len(idx) == 0:
        print("NONE feasible at ~10 Hz -- report nearest misses by lift:")
        near = np.where(np.abs(a0) <= 0.1)[0]
        for j in sorted(near, key=lambda j: -L[j])[:10]:
            print(f"  L={L[j]*1e3:5.0f}mN tau={tau[j]*1e3:6.1f}mNm")
        idx = np.array(sorted(near, key=lambda j: -L[j])[:100])  # analyze high-lift instead
        print("-- analyzing the 100 HIGHEST-LIFT ~10Hz designs instead --")
    order = idx[np.argsort(tau[idx])]
    top = order[: max(10, len(order) // 4)]
    rest = order[len(order) // 4:]
    s = 2 * K + 1
    st_top = G[top, s:s + n_tend]; st_rest = G[rest, s:s + n_tend]
    print(f"\ntop group n={len(top)} (tau {tau[top].mean()*1e3:.0f}mNm avg, lift {L[top].mean()*1e3:.0f}mN)"
          f" vs rest n={len(rest)}")
    print(f"{'tendon':22s} {'base':>5s} {'top N/m':>8s} {'rest N/m':>9s} {'delta':>7s}")
    deltas = []
    for t in range(n_tend):
        mt = base_stiff[t] * math.exp(np.median(st_top[:, t]))
        mr = base_stiff[t] * math.exp(np.median(st_rest[:, t]))
        deltas.append((mt / mr, tnames[t], mt, mr))
    for ratio, name, mt, mr in sorted(deltas, key=lambda d: -abs(math.log(d[0]))):
        tag = "STIFFER" if ratio > 1.1 else ("SOFTER" if ratio < 0.9 else "")
        print(f"{name:22s} {200:5.0f} {mt:8.0f} {mr:9.0f} {ratio:6.2f}x {tag}")
    # print the 5 best individual designs' full tendon sets
    print("\n=== 5 best ~10Hz designs ===")
    for j in order[:5]:
        st = [f"{base_stiff[t]*math.exp(G[j, s+t]):.0f}" for t in range(n_tend)]
        print(f"tau={tau[j]*1e3:6.1f} L={L[j]*1e3:5.0f} A={rows[j]['A']*1e3:4.0f}mm "
              f"stiff=[{','.join(st)}]")


if __name__ == "__main__":
    main()
