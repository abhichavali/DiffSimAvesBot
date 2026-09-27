#!/usr/bin/env python3
"""The free-flight hover tools on the GPU: the CPU loop (hover.trim -> hover.lqr -> opt/distill_hover.py ->
opt/verify_hover.py / opt/push_sweep.py / opt/hover_mc.py) as batched MJX flights (hover.mjx_flight,
hover.mjx_sensors, hover.mjx_cradle), plus the parity check against the CPU reference.

    P=".venv/bin/python opt/hover_gpu.py"          # on a GPU box (gpu-box skill); every command takes --help
    $P derive --out D --dt-ms 0.10 0.13 0.16 [--trim] [--cpu-check]   # [trim ->] linearize -> distill -> verify + sweep
    $P trim --out D/control.tg                      # hover.trim: 55 cradle runs per Gauss-Newton iteration, one batch
    $P linearize --out D/lqr.npz [--ref L.npz] [--jacfwd]              # hover.lqr (+ sensor model if hover.sensors)
    $P verify [--dt-ms 0.10 0.13 0.16] [--lqr L.npz | --open-loop]    # opt/verify_hover.py
    $P sweep                                        # opt/push_sweep.py
    $P mc --lqr L.npz --obs pos,att,w --noise-scales 1 3 --seeds 500   # opt/hover_mc.py (needs hover.sensors)
    $P parity --dt-ms 0.10 0.13 0.16 --lqr L.npz --cpu-ref ref.npz     # CPU vs device, open loop + LQR, pushes
    $P bench --batch 512 2048 8192                  # wingbeats/s vs batch width

Device knobs (all commands): float32 unless --x64; the model's solver iterations (never 1: see the
mjx-free-flight-parity finding); --ls-iterations 8 (default); contacts off unless --contact. CPU references
(parity) run on a process pool on the host, or come from --cpu-ref (compute them where the cores are free).
The release (cradle settle) of flights runs on CPU, ~10 s per (design, dt).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]

import numpy as np

from design_paths import FREE_HOVER_TG

VERIFY_PUSHES = {30: np.r_[0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0],         # opt/verify_hover.py's
                 60: np.r_[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
                 90: np.r_[0, 0, 0, 0, 0, 0, 0, 0.2, 0.1, 0, 0.5, 0]}
# parity error scale per state component: 10% of the smallest verify push (0.2 m/s, 1 rad/s), 1 cm, ~0.6 deg
ERR_SCALE = np.array([0.01] * 3 + [0.01] * 3 + [0.02] * 3 + [0.1] * 3)
_CACHE: dict = {}


def load(tg, control, dt_ms):
    """(bird, released state, flight policy) on CPU. The release is cached per (design, open loop, dt); the
    policy is re-read every call (derive adds the closed loop to a control whose release is cached)."""
    from hover import launch as L
    from hover.policy import load_flight_policy
    fp = load_flight_policy(control)
    key = (str(tg), fp.theta.tobytes(), fp.attitude.tobytes(), fp.phase_lock, float(dt_ms))
    if key not in _CACHE:
        bird = L.load_bird(tg, control=control, dt_ms=dt_ms)
        _CACHE[key] = (bird, L.release(bird, fp.theta, fp.attitude, fp.K, k_phi=fp.phase_lock))
    return (*_CACHE[key], fp)


def kick_array(kicks: dict | None, n_beats: int) -> np.ndarray:
    k = np.zeros((n_beats, 12))
    for n, v in (kicks or {}).items():
        if n < n_beats:                      # hover.launch.fly never reaches a later push
            k[n] = v
    return k


# ---------------------------------------------------------------------------------------------- CPU reference

def _cpu_job(job):
    """One CPU flight: (tg, control, dt_ms, n_beats, K gain or None (open loop) or "mlp", kicks dict)."""
    from hover import launch as L
    tg, control, dt_ms, n_beats, gain, kicks = job
    bird, rel, fp = load(tg, control, dt_ms)
    if gain is None:
        pol = None
    elif isinstance(gain, str):
        pol = fp.correction
    else:
        pol = lambda s: np.concatenate([[0.0], np.clip(-gain @ s, -0.15, 0.15)])
    t = time.time()
    tr = L.fly(bird, rel, fp.theta, n_beats, policy=pol, kicks=kicks)
    return tr, time.time() - t


def cpu_flights(jobs, procs=None):
    with Pool(procs or os.cpu_count()) as p:
        return p.map(_cpu_job, jobs, chunksize=1)


# ---------------------------------------------------------------------------------------------- metrics

def verdict(tr, n_beats):
    """opt/verify_hover.py's summary of one strobed trace (CPU list or cpu_trace of a device flight)."""
    tilt = np.degrees(np.linalg.norm(tr[:, 3:5], axis=1)); pos = np.linalg.norm(tr[:, :3], axis=1) * 1e3
    return {"flew_all": len(tr) == n_beats + 1, "beats": len(tr) - 1, "tilt_max": float(tilt.max()),
            "tilt_last20": float(tilt[-20:].mean()), "pos_max_mm": float(pos.max()), "pos_final_mm": float(pos[-1])}


def state_error(cpu, gpu, windows=(10, 40, 120)):
    """Max over beats of max_i |gpu_i - cpu_i| / ERR_SCALE_i, over the beats both flew, per window."""
    n = min(len(cpu), len(gpu))
    e = np.max(np.abs(gpu[:n] - cpu[:n]) / ERR_SCALE, axis=1)
    return {f"err_{w}": float(e[:w + 1].max()) for w in windows if w < n} | {"err_all": float(e.max())}, e


def growth_rate(tr, lo=5, hi=None):
    """Per-wingbeat growth factor of |s| (att + rates) fitted over beats lo..hi (open-loop divergence)."""
    hi = hi or len(tr)
    x = np.linalg.norm(tr[lo:hi, 3:6] / 0.01, axis=1) + np.linalg.norm(tr[lo:hi, 9:12] / 0.1, axis=1)
    return float(np.exp(np.polyfit(np.arange(lo, lo + len(x)), np.log(x), 1)[0]))




SWEEP_PUSHES = [(f"{axis} {mag:g} {unit}", np.eye(12)[idx] * mag)                # opt/push_sweep.py's
                for axis, idx, unit, mags in (("side vel", 7, "m/s", (0.2, 0.4, 0.6, 0.8)), ("fwd vel", 6, "m/s", (0.4, 0.8)),
                                              ("up vel", 8, "m/s", (0.3, 0.6)), ("roll rate", 9, "rad/s", (1, 2, 3, 4)),
                                              ("pitch rate", 10, "rad/s", (1, 2, 3, 4)), ("yaw rate", 11, "rad/s", (2, 4, 8)))
                for mag in mags]


# ---------------------------------------------------------------------------------------------- device helpers

def flight(a, dt_ms, control=None):
    """(Flight on device, bird, released state, flight policy) at ``dt_ms`` with the device knobs of ``a``."""
    from hover import mjx_flight as F
    bird, rel, fp = load(a.tg, control or a.control, dt_ms)
    fl = F.Flight(bird, rel, fp.theta, fp.K, k_phi=fp.phase_lock, solver_iterations=a.iterations,
                  ls_iterations=a.ls_iterations, disable_contact=a.no_contact)
    return fl, bird, rel, fp


def knobs(fl, a) -> str:
    import jax
    return (f"{'float64' if jax.config.jax_enable_x64 else 'float32'}, solver iterations {fl.model.opt.iterations}, "
            f"line search {fl.model.opt.ls_iterations}{', contacts off' if a.no_contact else ''}")


def policy_batch(fl, fp, n_beats, a, B):
    """(batched flyer, gains (B, 8K, 12), label) for the policy chosen by --open-loop / --lqr / the control's MLP."""
    from hover import mjx_flight as F
    nu = 8 * fp.K
    if a.open_loop or (a.lqr is None and fp.mlp is None):
        return F.batch_flyer(fl, n_beats, "linear"), np.zeros((B, nu, 12)), "open loop"
    if a.lqr is not None:
        return F.batch_flyer(fl, n_beats, "linear"), np.repeat(np.load(a.lqr)["K"][None], B, 0), f"LQR {a.lqr}"
    return F.batch_flyer(fl, n_beats, "mlp", fp.mlp), np.zeros((B, nu, 12)), "closed loop (MLP)"


# ---------------------------------------------------------------------------------------------- commands

def cmd_parity(a):
    from hover import mjx_flight as F
    lqr_K = np.load(a.lqr)["K"]
    scen = [("open", None, None), ("open+push", None, VERIFY_PUSHES), ("lqr", lqr_K, None), ("lqr+push", lqr_K, VERIFY_PUSHES)]
    ref = dict(np.load(a.cpu_ref, allow_pickle=True)) if a.cpu_ref and a.cpu_ref.exists() else {}
    missing = [(dt, n, g, k) for dt in a.dt_ms for n, g, k in scen if f"{dt}/{n}" not in ref]
    if missing:
        print(f"CPU reference: {len(missing)} flights of {a.beats} wingbeats on {os.cpu_count()} processes", flush=True)
        t = time.time()
        res = cpu_flights([(a.tg, a.control, dt, a.beats, g, k) for dt, n, g, k in missing])
        for (dt, n, _, _), (tr, sec) in zip(missing, res):
            ref[f"{dt}/{n}"] = tr; ref[f"{dt}/{n}/sec"] = sec
        print(f"  {time.time() - t:.0f} s", flush=True)
        if a.cpu_ref:
            np.savez(a.cpu_ref, **ref)
    if a.cpu_only:
        return
    rows, traces = [], {}
    for dt in a.dt_ms:
        fl, bird, rel, fp = flight(a, dt)
        f = F.batch_flyer(fl, a.beats, "linear")
        kicks = np.stack([kick_array(k, a.beats) for _, _, k in scen])
        gains = np.stack([np.zeros((8 * fp.K, 12)) if g is None else g for _, g, _ in scen])
        t = time.time()
        Sg, Ug, Ag = F.run_batched(f, fl.arrays, kicks, gains, np.zeros((len(scen), 12)))
        sec = time.time() - t
        print(f"== dt {dt} ms (snapped {fl.static.dt * 1e3:.5f}, {fl.static.n_step} steps/wingbeat) | {knobs(fl, a)} | "
              f"device: {len(scen)} flights {sec:.1f} s (with compile)", flush=True)
        for i, (name, _, _) in enumerate(scen):
            cpu = ref[f"{dt}/{name}"]
            gpu = F.cpu_trace(Sg[i], Ag[i])
            traces[f"{dt}/{name}"] = gpu
            vc, vg = verdict(cpu, a.beats), verdict(gpu, a.beats)
            err, _ = state_error(cpu, gpu)
            row = {"dt_ms": dt, "scenario": name, "cpu": vc, "gpu": vg, **err, "gpu_sec": sec,
                   "cpu_sec": float(ref.get(f"{dt}/{name}/sec", np.nan))}
            if name.startswith("open"):
                row["growth_cpu"], row["growth_gpu"] = growth_rate(cpu, 5, min(len(cpu), 35)), growth_rate(gpu, 5, min(len(gpu), 35))
            rows.append(row)
            same = vc["flew_all"] == vg["flew_all"]
            print(f"  {name:9s} CPU {'flew all' if vc['flew_all'] else 'LOST %3d' % vc['beats']} tilt max {vc['tilt_max']:5.1f} "
                  f"pos max {vc['pos_max_mm']:6.0f} | GPU {'flew all' if vg['flew_all'] else 'LOST %3d' % vg['beats']} "
                  f"tilt max {vg['tilt_max']:5.1f} pos max {vg['pos_max_mm']:6.0f} | verdict {'same' if same else 'DIFFERS'} | "
                  f"err/scale max over wb 0-10 {err.get('err_10', np.nan):.3g}, 0-40 {err.get('err_40', np.nan):.3g}, "
                  f"all {err['err_all']:.3g}"
                  + (f" | growth/wb CPU {row['growth_cpu']:.3f} GPU {row['growth_gpu']:.3f}" if "growth_cpu" in row else ""),
                  flush=True)
    if a.save_traces:
        np.savez(a.save_traces, **traces)
    dump(a, {"tool": "hover_gpu parity", "beats": a.beats, "rows": rows})
    return rows


def compare_ab(A, B, ref):
    """Agreement of a linear model with a reference (A, B): relative errors, B column cosines, the LQR gains and
    closed-loop spectral radii (and the device gain closed on the reference model)."""
    from hover.lqr import lqr
    Ar, Br = ref["A"], ref["B"]
    cos = np.sum(B * Br, 0) / (np.linalg.norm(B, axis=0) * np.linalg.norm(Br, axis=0))
    K, Kr = lqr(A, B), lqr(Ar, Br)
    eig = lambda M: float(np.abs(np.linalg.eigvals(M)).max())
    return {"A_rel_err": float(np.linalg.norm(A - Ar) / np.linalg.norm(Ar)),
            "B_rel_err": float(np.linalg.norm(B - Br) / np.linalg.norm(Br)),
            "B_col_cos_median": float(np.median(cos)), "B_col_cos_min": float(cos.min()),
            "K_rel_err": float(np.linalg.norm(K - Kr) / np.linalg.norm(Kr)),
            "open_eig_max": eig(A), "open_eig_max_ref": eig(Ar),
            "closed_eig_max": eig(A - B @ K), "closed_eig_max_ref": eig(Ar - Br @ Kr),
            "cross_closed_eig_max": eig(Ar - Br @ K)}


def cmd_linearize(a):
    from hover import mjx_flight as F
    from hover.lqr import lqr
    fl, bird, rel, fp = flight(a, a.dt_ms[0])
    t = time.time()
    obs = {}
    try:                                            # with the sensor model (hover.sensors) when it exists
        from hover import mjx_sensors as MS
        from hover import sensors as Sn
        A, B, drift, obs = MS.linearize_obs(fl, chunk=a.chunk)
        obs = {**obs, "channels": np.array(Sn.CHANNELS)}
    except ImportError:
        A, B, drift = F.linearize(fl, chunk=a.chunk)
    sec = time.time() - t
    K = lqr(A, B)
    ol = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]; cl = np.sort(np.abs(np.linalg.eigvals(A - B @ K)))[::-1]
    print(f"device finite differences ({len(F.lqr_tasks(fp.K)[0])} one-wingbeat rollouts, {knobs(fl, a)}): {sec:.1f} s with compile")
    print("open-loop |eig| per wingbeat:", np.round(ol, 3))
    print("closed-loop |eig| (LQR):     ", np.round(cl, 3))
    print("one-beat drift of the unperturbed orbit:", np.round(drift, 4))
    out = {"tool": "hover_gpu linearize", "sec": sec, "open_eig": ol, "closed_eig": cl}
    if a.ref:
        ref = np.load(a.ref)
        out["fd_vs_ref"] = compare_ab(A, B, ref)
        for k in ("C_now", "C_prev", "J"):
            if k in obs and k in ref.files:
                out["fd_vs_ref"][f"{k}_max_abs_err"] = float(np.abs(obs[k] - ref[k]).max())
                out["fd_vs_ref"][f"{k}_max_abs"] = float(np.abs(ref[k]).max())
        print("vs", a.ref, json.dumps(out["fd_vs_ref"], default=float))
    if a.jacfwd:
        t = time.time()
        Aj, Bj, _ = F.linearize_jacfwd(fl)
        out["jacfwd_sec"] = time.time() - t
        out["jacfwd_vs_fd"] = compare_ab(Aj, Bj, {"A": A, "B": B})
        print(f"jacfwd ({out['jacfwd_sec']:.1f} s with compile) vs device FD:", json.dumps(out["jacfwd_vs_fd"], default=float))
        if a.ref:
            out["jacfwd_vs_ref"] = compare_ab(Aj, Bj, np.load(a.ref))
            print("jacfwd vs", a.ref, json.dumps(out["jacfwd_vs_ref"], default=float))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out, A=A, B=B, K=K, drift=drift, **obs)
        print(f"-> {a.out}")
    dump(a, out)
    return A, B, K


def cmd_verify(a):
    """opt/verify_hover.py on the device: no pushes and the verify pushes, at every --dt-ms, one batch per dt."""
    from hover import mjx_flight as F
    rows = []
    for dt in a.dt_ms:
        fl, bird, rel, fp = flight(a, dt)
        f, gains, label = policy_batch(fl, fp, a.beats, a, 2)
        kicks = np.stack([kick_array(None, a.beats), kick_array(VERIFY_PUSHES, a.beats)])
        t = time.time()
        S_, U_, A_ = F.run_batched(f, fl.arrays, kicks, gains, np.zeros((2, 12)))
        sec = time.time() - t
        print(f"== dt {dt} ms (snapped {fl.static.dt * 1e3:.5f}) | {label} | {knobs(fl, a)} | {sec:.1f} s with compile", flush=True)
        for i, name in enumerate(("no pushes", "pushes at wb 30/60/90")):
            tr = F.cpu_trace(S_[i], A_[i])
            v = verdict(tr, a.beats)
            tilt = np.degrees(np.linalg.norm(tr[:, 3:5], axis=1)); spd = np.linalg.norm(tr[:, 6:9], axis=1) * 1e3
            print(f"{label}, {name}: {'flew all %d wingbeats' % a.beats if v['flew_all'] else 'LOST at wb %d' % v['beats']}"
                  f" | tilt max {v['tilt_max']:5.1f} deg, last-20 mean {v['tilt_last20']:4.1f} | pos max {v['pos_max_mm']:6.0f} mm, "
                  f"final {v['pos_final_mm']:6.0f} | speed last-20 {spd[-20:].mean():5.0f} mm/s", flush=True)
            print("   tilt every 10 wb:", np.round(tilt[::10], 1).tolist())
            rows.append({"dt_ms": dt, "scenario": name, **v, "sec": sec})
    dump(a, {"tool": "hover_gpu verify", "pass": all(r["flew_all"] for r in rows), "rows": rows})
    return rows


def sweep_rows(S_, A_, names):
    from hover import mjx_flight as F
    rows = []
    for i, name in enumerate(names):
        tr = F.cpu_trace(S_[i], A_[i])
        tilt = np.degrees(np.linalg.norm(tr[:, 3:5], axis=1)); pos = np.linalg.norm(tr[:, :3], axis=1) * 1e3
        ok = len(tr) == 81 and tilt[-10:].mean() < 2.0
        rows.append({"push": name, "recovered": bool(ok), "beats": len(tr) - 1, "tilt_max": float(tilt.max()),
                     "pos_max_mm": float(pos.max()), "tilt_final": float(tilt[-10:].mean())})
    return rows


def cmd_sweep(a):
    """opt/push_sweep.py on the device: each push at wingbeat 20 of 80, all in one batch."""
    from hover import mjx_flight as F
    fl, bird, rel, fp = flight(a, a.dt_ms[0])
    names = [n for n, _ in SWEEP_PUSHES]
    kicks = np.stack([kick_array({20: d}, 80) for _, d in SWEEP_PUSHES])
    f, gains, label = policy_batch(fl, fp, 80, a, len(names))
    t = time.time()
    S_, U_, A_ = F.run_batched(f, fl.arrays, kicks, gains, np.zeros((len(names), 12)))
    sec = time.time() - t
    print(f"{label} | dt {a.dt_ms[0]} | {knobs(fl, a)} | {len(names)} flights {sec:.1f} s with compile")
    rows = sweep_rows(S_, A_, names)
    for r in rows:
        n = r["beats"]
        print(f"{r['push']:22s}: {'RECOVERED' if r['recovered'] else 'LOST at wb %d' % n if n < 80 else 'not settled'} | "
              f"peak tilt {r['tilt_max']:5.1f} deg, peak excursion {r['pos_max_mm']:5.0f} mm, final tilt {r['tilt_final']:4.1f} deg", flush=True)
    dump(a, {"tool": "hover_gpu sweep", "rows": rows, "sec": sec})
    return rows


def cmd_trim(a):
    """hover.trim on the device: every Gauss-Newton iteration's 55 cradle runs in one batch."""
    from hover import launch as L
    from hover import mjx_cradle as MC
    from hover.policy import load_flight_policy, write_flight_control
    fp = load_flight_policy(a.control)
    bird = L.load_bird(a.tg, control=a.control, dt_ms=a.dt_ms[0])
    cr = MC.Cradle(bird, fp.K, k_phi=fp.phase_lock, solver_iterations=a.iterations, ls_iterations=a.ls_iterations,
                   disable_contact=a.no_contact)
    t = time.time()
    x, w, att, J = MC.trim(cr, fp, iters=a.iters, chunk=a.chunk, log=lambda s: print(s, f"({time.time() - t:.0f} s)", flush=True))
    K, nt = fp.K, 1 + 8 * fp.K
    write_flight_control(a.out, a0=fp.control.a0 * math.exp(x[0]), depth_a=x[1:1 + 4 * K].reshape(4, K),
                         depth_b=x[1 + 4 * K:nt].reshape(4, K), attitude=att, phase_lock=fp.phase_lock, bird=fp.control.bird,
                         header=f"Trimmed by opt/hover_gpu.py trim (device) from {a.control.name}: the launch cradle carries "
                                f"{np.abs(w[:3]).max() * 1e3:.2f} mN / {np.abs(w[3:]).max() * 1e3:.3f} mNm.")
    if a.jacobian is not None and J is not None:
        np.savez(a.jacobian, J=J)
    print(f"-> {a.out} ({time.time() - t:.0f} s)")
    dump(a, {"tool": "hover_gpu trim", "wrench": w, "x": x, "sec": time.time() - t})
    return a.out


def cmd_bench(a):
    """Device throughput: wingbeats/s of the closed-loop flight vs batch width (compile time separately)."""
    from hover import mjx_flight as F
    fl, bird, rel, fp = flight(a, a.dt_ms[0])
    rows = []
    for B in a.batch:
        f, gains, label = policy_batch(fl, fp, a.beats, a, B)
        args = (np.zeros((B, a.beats, 12)), gains, np.zeros((B, 12)))
        t = time.time(); F.run_batched(f, fl.arrays, *args); t1 = time.time() - t
        t = time.time(); F.run_batched(f, fl.arrays, *args); t2 = time.time() - t
        r = {"batch": B, "first_call_s": t1, "run_s": t2, "wingbeats_per_s": B * a.beats / t2,
             "flights120_per_s": B * a.beats / t2 / 120, "ms_per_step": t2 / a.beats / fl.static.n_step * 1e3}
        rows.append(r)
        print(f"B={B:6d} | first call {t1:6.1f} s | {t2:6.2f} s: {r['wingbeats_per_s']:8.0f} wingbeats/s = "
              f"{r['flights120_per_s']:6.2f} flights of 120 wb/s | {r['ms_per_step']:.3f} ms per batched step | {knobs(fl, a)}", flush=True)
    dump(a, {"tool": "hover_gpu bench", "rows": rows})


def cmd_derive(a):
    """The whole free-flight-policy loop on the device: [trim ->] linearize + LQR -> distill (JAX) -> verify (all
    --dt-ms) + push sweep; optionally the CPU reference verify at the first dt."""
    import subprocess
    from argparse import Namespace
    from hover.policy import load_flight_policy, write_flight_control
    a.out.mkdir(parents=True, exist_ok=True)
    ctl = a.out / "control.tg"
    t0 = time.time()
    base = vars(a).copy()
    if a.trim:
        cmd_trim(Namespace(**{**base, "out": ctl, "jacobian": None, "json": None, "dt_ms": [0.13]}))
    else:                                               # the given open loop, without its closed loop
        fp = load_flight_policy(a.control)
        K = fp.K
        write_flight_control(ctl, a0=fp.control.a0, depth_a=fp.theta[1:1 + 4 * K].reshape(4, K),
                             depth_b=fp.theta[1 + 4 * K:].reshape(4, K), attitude=fp.attitude, phase_lock=fp.phase_lock,
                             bird=fp.control.bird, header=f"Open loop of {a.control} (opt/hover_gpu.py derive).")
    t_trim = time.time() - t0
    lin = Namespace(**{**base, "control": ctl, "ref": a.ref, "jacfwd": False, "out": a.out / "lqr.npz", "json": None,
                       "dt_ms": [0.13]})
    cmd_linearize(lin)
    t_lin = time.time() - t0 - t_trim
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "distill_hover.py"), "--lqr", str(a.out / "lqr.npz"),
                        "--out", str(a.out / "mlp_closed_loop.npz")], capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-2] if r.returncode == 0 else r.stderr[-2000:], flush=True)
    if r.returncode:
        raise SystemExit("distill failed")
    with open(ctl, "a") as fh:
        fh.write("closed_loop = mlp_closed_loop.npz\n")
    t_dist = time.time() - t0 - t_trim - t_lin
    ver = Namespace(**{**base, "control": ctl, "open_loop": False, "lqr": None, "json": None})
    rows = cmd_verify(ver)
    srows = cmd_sweep(Namespace(**{**vars(ver), "dt_ms": [0.13]}))
    t_ver = time.time() - t0 - t_trim - t_lin - t_dist
    ok = all(r["flew_all"] for r in rows) and all(r["recovered"] for r in srows)
    print(f"derive: {'PASS' if ok else 'FAIL'} | verify {sum(r['flew_all'] for r in rows)}/{len(rows)} flew all, sweep "
          f"{sum(r['recovered'] for r in srows)}/{len(srows)} recovered | trim {t_trim:.0f} s, linearize {t_lin:.0f} s, "
          f"distill {t_dist:.0f} s, verify+sweep {t_ver:.0f} s", flush=True)
    if a.cpu_check:
        print("CPU reference check (opt/verify_hover.py, float64, full solver):", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / "verify_hover.py"), "--tg", str(a.tg), "--control", str(ctl)])
    dump(a, {"tool": "hover_gpu derive", "pass": ok, "verify": rows, "sweep": srows,
             "sec": {"trim": t_trim, "linearize": t_lin, "distill": t_dist, "verify_sweep": t_ver}})


def mc_controllers(a, fp):
    """{noise scale: hover.reduced controller dict or None} as opt/hover_mc.py builds them (an LQG designed per
    noise scale unless --design-noise-scale fixes it)."""
    from hover import reduced as Rd
    from hover import sensors as Sn
    out = {}
    for ns in a.noise_scales:
        noise = Sn.SensorNoise(scale=ns, pos_scale=a.pos_noise_scale, pos_latency_ms=a.pos_latency_ms)
        if a.ctrl:
            out[ns] = Rd.load_controller(a.ctrl)
        elif a.lqr and a.obs:
            dn = a.design_noise_scale if a.design_noise_scale is not None else ns
            dnoise = Sn.SensorNoise(scale=dn, pos_scale=a.pos_noise_scale, pos_latency_ms=a.pos_latency_ms)
            out[ns] = Rd.design(Rd.load_model(a.lqr), a.obs, a.mode, noise=dnoise, beat_s=2 * math.pi / fp.control.omega,
                                process_scale=a.process_scale)
        elif a.lqr:
            out[ns] = {"kind": "state", "obs": np.arange(12), "K": np.load(a.lqr)["K"], "u_max": Rd.U_MAX}
        else:
            out[ns] = None                          # the control's state MLP on y[:12] (or open loop)
    return out


def mc_scenarios(a):
    """[(name, kicks dict or None, n_beats)]: hover_mc's none / verify / sweep:<push>."""
    scen = []
    for s in a.scenarios:
        if s == "none":
            scen.append(("none", None, a.beats))
        elif s == "verify":
            scen.append(("verify", VERIFY_PUSHES, a.beats))
        else:
            scen += [(f"sweep:{n}", {20: d}, 80) for n, d in SWEEP_PUSHES]
    return scen


def summarize(tr, n_beats, beat_s, Y=None):
    """hover.evaluate.summarize for one flight (true strobed trace; Y = its sensor readings)."""
    tilt = np.degrees(np.linalg.norm(tr[:, 3:5], axis=1)); pos = np.linalg.norm(tr[:, :3], axis=1) * 1e3
    spd = np.linalg.norm(tr[:, 6:9], axis=1) * 1e3
    out = {"beats": len(tr) - 1, "flew_all": bool(len(tr) == n_beats + 1), "seconds": (len(tr) - 1) * beat_s,
           "tilt_max_deg": float(tilt.max()), "tilt_mean_deg": float(tilt.mean()),
           "tilt_last20_deg": float(tilt[-20:].mean()), "tilt_last10_deg": float(tilt[-10:].mean()),
           "pos_max_mm": float(pos.max()), "drift_final_mm": float(pos[-1]), "alt_final_mm": float(tr[-1, 2] * 1e3),
           "speed_last20_mm_s": float(spd[-20:].mean()), "yaw_final_deg": float(np.degrees(tr[-1, 5]))}
    if Y is not None:
        e = np.degrees(Y[:len(tr), 3:6] - tr[:, 3:6])
        out.update(att_est_err_rms_deg=float(np.sqrt((e[:, :2] ** 2).sum(1).mean())), yaw_est_err_final_deg=float(e[-1, 2]))
    return out


def aggregate(rows, a):
    agg = []
    for dt in a.dt_ms:
        for ns in a.noise_scales:
            for sname in a.scenarios:
                rs = [r for r in rows if r["dt_ms"] == dt and r["noise_scale"] == ns and r["scenario"].split(":")[0] == sname]
                if not rs:
                    continue
                g = {"dt_ms": dt, "noise_scale": ns, "scenario": sname, "n": len(rs),
                     "pass_rate": float(np.mean([r["pass"] for r in rs])), "beats_mean": float(np.mean([r["beats"] for r in rs])),
                     "beats_min": int(min(r["beats"] for r in rs)),
                     "tilt_max_deg_median": float(np.nanmedian([r["tilt_max_deg"] for r in rs])),
                     "tilt_last20_deg_median": float(np.nanmedian([r["tilt_last20_deg"] for r in rs])),
                     "drift_final_mm_median": float(np.nanmedian([r["drift_final_mm"] for r in rs]))}
                if "att_est_err_rms_deg" in rs[0]:
                    g["att_est_err_rms_deg_median"] = float(np.median([r["att_est_err_rms_deg"] for r in rs]))
                agg.append(g)
                print(f"dt {dt:5.3f} noise x{ns:<5g} {sname:7s}: pass {g['pass_rate'] * 100:5.1f}% of {g['n']:4d} | beats mean "
                      f"{g['beats_mean']:6.1f} min {g['beats_min']:4d} | tilt max med {g['tilt_max_deg_median']:5.2f} deg | "
                      f"drift med {g['drift_final_mm_median']:6.1f} mm", flush=True)
    return agg


def cmd_mc(a):
    """opt/hover_mc.py on the device: seeds x noise scales x scenarios as batched flights, one batch per
    (dt, flight length)."""
    import jax
    from hover import mjx_flight as F
    from hover import mjx_sensors as MS
    from hover import sensors as Sn
    t0, rows = time.time(), []
    for dt in a.dt_ms:
        fl, bird, rel, fp = flight(a, dt)
        beat_s = 2 * math.pi / fp.control.omega
        ctls = mc_controllers(a, fp)
        c0 = ctls[a.noise_scales[0]]
        noise = Sn.SensorNoise(scale=a.noise_scales[0], pos_scale=a.pos_noise_scale, pos_latency_ms=a.pos_latency_ms)
        mlp = fp.mlp if c0 is None and not a.open_loop else None
        sp = MS.SensorPipeline(fl, noise, att_mode=a.att_mode, ctl=c0, state_mlp=mlp)
        by_len = {}
        for name, kicks, nb in mc_scenarios(a):
            by_len.setdefault(nb, []).append((name, kicks))
        for nb, scen in by_len.items():
            jobs = [(ns, seed, name, kicks) for ns in a.noise_scales for seed in range(a.seed, a.seed + a.seeds)
                    for name, kicks in scen]
            kick_arr = np.stack([kick_array(k, nb) for _, _, _, k in jobs])
            ids = np.array([seed * 100003 + int(round(ns * 1000)) + 7919 * i for i, (ns, seed, _, _) in enumerate(jobs)])
            keys = np.asarray(jax.vmap(jax.random.PRNGKey)(jax.numpy.asarray(ids % (2 ** 31))))
            scales = np.array([ns for ns, _, _, _ in jobs])
            ctl_b = {k: np.stack([np.asarray(MS.ctl_arrays(ctls[ns])[k]) for ns, _, _, _ in jobs])
                     for k in MS.ctl_arrays(c0)}
            f = MS.batch_sensor_flyer(fl, sp, nb)
            t = time.time()
            res = run_sensor_batched(f, fl.arrays, kick_arr, keys, scales, ctl_b, chunk=a.chunk)
            sec = time.time() - t
            print(f"dt {dt}: {len(jobs)} flights of {nb} wingbeats in {sec:.1f} s with compile "
                  f"({len(jobs) * nb / sec:.0f} wingbeats/s) | {knobs(fl, a)}", flush=True)
            Sg, Yg, Ug, Ag = res
            for i, (ns, seed, name, _) in enumerate(jobs):
                n = int(F.lost_at(Ag[i]))
                s = summarize(Sg[i][:n + 1], nb, beat_s, Yg[i][:n + 1])
                ok = s["flew_all"] and (not name.startswith("sweep") or s["tilt_last10_deg"] < 2.0)
                rows.append({"dt_ms": dt, "noise_scale": ns, "seed": seed, "scenario": name, "pass": bool(ok), **s})
    agg = aggregate(rows, a)
    wall = time.time() - t0
    print(f"{len(rows)} flights in {wall:.0f} s")
    dump(a, {"tool": "hover_gpu mc", "aggregate": agg, "wall_s": wall, "runs": rows})
    return agg


def run_sensor_batched(f, ar, kicks, keys, scales, ctl, chunk=None):
    """run_batched for batch_sensor_flyer (keys stay uint32, the controller dict is batched per flight)."""
    import jax.numpy as jnp
    from hover import mjx_flight as F
    B = len(kicks)
    chunk = min(chunk or B, B)
    outs = []
    for s in range(0, B, chunk):
        sl = slice(s, s + chunk)
        part = [kicks[sl], keys[sl], scales[sl]]
        cpart = {k: v[sl] for k, v in ctl.items()}
        n = len(part[0])
        if n < chunk:
            pad = lambda x: np.concatenate([x, np.repeat(x[-1:], chunk - n, 0)])
            part = [pad(p) for p in part]; cpart = {k: pad(v) for k, v in cpart.items()}
        res = f(ar, jnp.asarray(part[0], F.fdtype()), jnp.asarray(part[1], jnp.uint32), jnp.asarray(part[2], F.fdtype()),
                {k: jnp.asarray(v, F.fdtype()) for k, v in cpart.items()})
        outs.append([np.asarray(r)[:n] for r in res])
    return [np.concatenate(o) for o in zip(*outs)]


def dump(a, obj):
    if getattr(a, "json", None):
        a.json.parent.mkdir(parents=True, exist_ok=True)
        obj["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
        a.json.write_text(json.dumps(obj, indent=1, default=lambda x: np.asarray(x).tolist()))
        print(f"-> {a.json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tg", type=Path, default=FREE_HOVER_TG)
    ap.add_argument("--control", type=Path, default=FREE_HOVER_TG.parent / "control.tg")
    dev = argparse.ArgumentParser(add_help=False)
    g = dev.add_argument_group("device (MJX) knobs")
    g.add_argument("--x64", action="store_true", help="float64 on device (consumer GPUs run FP64 at 1/64 rate)")
    g.add_argument("--iterations", type=int, default=None, help="solver iterations (default: the model's, as the CPU)")
    g.add_argument("--ls-iterations", type=int, default=8,
                   help="line-search iterations (default 8: CPU parity as with the model's 50, 2-6x faster batched)")
    g.add_argument("--contact", dest="no_contact", action="store_false",
                   help="keep contacts on device (off by default: the free bird never touches itself; ncon = 0 on CPU)")
    g.add_argument("--chunk", type=int, default=None, help="flights per device call (default: all at once)")
    g.add_argument("--dt-ms", type=float, nargs="+", default=[0.13], help="timestep(s), snapped per wingbeat")
    g.add_argument("--json", type=Path, default=None)
    pol = argparse.ArgumentParser(add_help=False)
    g = pol.add_argument_group("policy (default: the control's closed loop)")
    g.add_argument("--open-loop", action="store_true")
    g.add_argument("--lqr", type=Path, default=None, help="fly the LQR gain K of this npz")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity", parents=[dev], help="CPU vs device free flight from the same released state")
    p.add_argument("--beats", type=int, default=120)
    p.add_argument("--lqr", type=Path, default=FREE_HOVER_TG.parent / "lqr.npz")
    p.add_argument("--cpu-ref", type=Path, default=None, help="npz cache of the CPU reference flights")
    p.add_argument("--cpu-only", action="store_true", help="only compute (and cache) the CPU reference")
    p.add_argument("--save-traces", type=Path, default=None, help="npz of the device traces")
    p = sub.add_parser("linearize", parents=[dev], help="per-wingbeat (A, B) + LQR on device (hover.lqr's finite differences)")
    p.add_argument("--ref", type=Path, default=None, help="a hover.lqr npz to compare with")
    p.add_argument("--jacfwd", action="store_true", help="also forward-mode AD of the one-wingbeat map")
    p.add_argument("--out", type=Path, default=None)
    p = sub.add_parser("verify", parents=[dev, pol], help="opt/verify_hover.py on device")
    p.add_argument("--beats", type=int, default=120)
    sub.add_parser("sweep", parents=[dev, pol], help="opt/push_sweep.py on device")
    p = sub.add_parser("trim", parents=[dev], help="hover.trim (Gauss-Newton on the cradle wrench) on device")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--jacobian", type=Path, default=None)
    p = sub.add_parser("bench", parents=[dev, pol], help="device throughput vs batch width")
    p.add_argument("--batch", type=int, nargs="+", default=[1, 64, 512, 2048, 8192])
    p.add_argument("--beats", type=int, default=4)
    p = sub.add_parser("derive", parents=[dev], help="[trim ->] linearize -> distill -> verify + sweep, all on device")
    p.add_argument("--out", type=Path, required=True, help="output dir: control.tg, lqr.npz, mlp_closed_loop.npz")
    p.add_argument("--trim", action="store_true", help="re-trim the open loop on device first (hover.trim)")
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--ref", type=Path, default=None, help="a hover.lqr npz to compare the linear model with")
    p.add_argument("--beats", type=int, default=120)
    p.add_argument("--cpu-check", action="store_true", help="finish with the CPU opt/verify_hover.py (the reference)")
    p = sub.add_parser("mc", parents=[dev], help="opt/hover_mc.py on device: sensor-noise Monte Carlo (hover.mjx_sensors)")
    p.add_argument("--beats", type=int, default=120)
    p.add_argument("--open-loop", action="store_true")
    p.add_argument("--lqr", type=Path, default=None, help="lqr.npz with the sensor model (hover.lqr); with --obs: design per noise scale")
    p.add_argument("--ctrl", type=Path, default=None, help="a saved hover.reduced controller")
    p.add_argument("--obs", default=None)
    p.add_argument("--mode", default="lqg", choices=("sof", "lqg"))
    p.add_argument("--process-scale", type=float, default=1.0)
    p.add_argument("--noise-scales", type=float, nargs="+", default=[1.0])
    p.add_argument("--design-noise-scale", type=float, default=None)
    p.add_argument("--pos-noise-scale", type=float, default=1.0)
    p.add_argument("--pos-latency-ms", type=float, default=0.0)
    p.add_argument("--att-mode", default="filter", choices=("filter", "truth"))
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenarios", nargs="+", default=["none", "verify"], choices=("none", "verify", "sweep"))
    a = ap.parse_args()
    if a.x64:
        import jax
        jax.config.update("jax_enable_x64", True)
    {"parity": cmd_parity, "linearize": cmd_linearize, "verify": cmd_verify, "sweep": cmd_sweep, "mc": cmd_mc, "trim": cmd_trim, "derive": cmd_derive, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
