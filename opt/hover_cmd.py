#!/usr/bin/env python3
"""Command-following flight of the free bird (HOVER / FWD / REV / UP / DOWN), learned and evaluated on the GPU.

The policy (hover.command): the device-linearized hover model -> LP steady-state feedforward tables per command
axis (two tilt maps) + an LQR servo with a position integral (handed off as the reference decelerates and gated
for 8 wingbeats after a stop: hover.command.integral_keep / integral_gate; attitude / feedforward references
lead the velocity reference by one wingbeat) -> one batched ILC / envelope pass of
constant-command flights refines the tables -> evaluated on random command sequences (hover.mjx_command: one
compiled program, B flights x 30 wingbeats, chained). No force of any kind acts after release: the bird is flown
only through the 24 harmonic depths of its four cranks.

    P=".venv/bin/python opt/hover_cmd.py"         # on a GPU box (gpu-box skill); every subcommand takes --help
    $P learn --out opt/runs_cmd/R                  # the budgeted run: release, x64 linearize, design, 2 ILC /
                                                   # continuation passes, eval 120 (-> policy.npz metrics.json
                                                   # timings.json traces.npz), then untimed: eval to 240, verify
    $P dt --run opt/runs_cmd/R --dt-ms 0.10 0.16   # the fixed policy at other timesteps (64 flights x 240 beats)
    $P sensors --run opt/runs_cmd/R                # mocap-position (P) and IMU-only (N) observers, noise x1 / x3
    $P cpu --run opt/runs_cmd/R                    # CPU reference (hover.launch.fly, contacts off as the device)
                                                   # of 16 eval flights (JAX_PLATFORMS=cpu)
    $P all --out opt/runs_cmd/R                    # learn -> dt -> sensors, cpu alongside
    uv run python opt/hover_cmd.py export --run opt/runs_cmd/R --out designs/hummer/free-hover/command.tg
                                                   # -> a flight control .tg (command_loop / command_observer) that
                                                   # src/view_free_hover.py and the GUI fly with live commands

Device knobs: float32 flights unless --x64, solver iterations 8 (a tumbling flight otherwise stalls the whole
vmapped batch; the change is below the float32 run-to-run noise), --ls-iterations 8, contacts off (the scene's
floor is 0.25 m below the release COM: descending flights pass through it); the linearization runs in x64.
Results and dead ends: opt/runs_cmd/NOTES.md. Set JAX_COMPILATION_CACHE_DIR for the persistent XLA cache (every run-specific quantity is a traced
array, so a second run is all cache hits). Timings: timings.json (per stage; compile / cache load per shape).
"""
from __future__ import annotations

import time

T_START = time.time()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]

import numpy as np  # noqa: E402

import hover_gpu as HG  # noqa: E402
from design_paths import FREE_HOVER_TG  # noqa: E402
from hover import command as C  # noqa: E402



def log(msg):
    print(f"[{time.time() - T_START:7.1f} s] {msg}", flush=True)


def setup_jax(a):
    import jax
    if a.x64:
        jax.config.update("jax_enable_x64", True)
    if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    return jax


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda x: np.asarray(x).tolist()))


class Clock:
    def __init__(self):
        self.t = time.time()
        self.stages = {}

    def stage(self, name):
        now = time.time()
        self.stages[name] = now - self.t
        self.t = now
        log(f"-- {name}: {self.stages[name]:.1f} s (since process start {now - T_START:.1f} s)")


# ---------------------------------------------------------------------------------------------- building blocks

def vop(a):
    """Fixed-magnitude command speeds (FWD/REV, UP, DOWN)."""
    return (a.v_op_h, a.v_op_z, a.v_op_down if a.v_op_down is not None else a.v_op_z)


def ref_kw(a, lead=None):
    """build_refs keywords of the integral handoff / gate, the stop rate and the feedforward lead (``lead`` overrides
    a.ff_lead: the ILC passes learn without it)."""
    return {"zeta_handoff": a.zeta_handoff, "zeta_gate": a.zeta_gate, "stop_rate": a.stop_rate,
            "ff_lead": a.ff_lead if lead is None else lead}


def pol_ref_args(pol):
    g = int(pol.get("zeta_gate", -1))
    return argparse.Namespace(zeta_handoff=bool(pol.get("zeta_handoff", False)), zeta_gate=None if g < 0 else g,
                              stop_rate=float(pol.get("stop_rate", 1.0)), ff_lead=int(pol.get("ff_lead", 0)))


def gate_arg(a):
    """--zeta-gate -1 = no gate column."""
    if a.zeta_gate is not None and a.zeta_gate < 0:
        a.zeta_gate = None
    return a


def xref_width(a):
    return 14 if a.zeta_gate is not None else 13 if a.zeta_handoff else 12


def rates(a, B):
    return np.broadcast_to(np.array([a.v_op_h, a.v_op_h, a.v_op_z]) / a.rate_beats, (B, 3))


def cell_pass_inputs(a, tabs, e_f, T, B, rng, lead=None):
    """The ILC / envelope pass: flight i flies the constant command of cell i % n_cells (2 maps x 64 cells);
    flights >= n_cells are dithered copies. Returns (cells, cell_of, v_cmd, x_ref, u_ff, dither)."""
    cells = C.cell_list(len(tabs), tabs[0]["speeds"].shape[1])
    nc = len(cells)
    cell_of = np.arange(B) % nc
    v_cmd = np.zeros((B, a.call_beats, 3)); rate = np.zeros((B, 3))
    for i in range(B):
        m, ax, j = cells[cell_of[i]]
        v = C.cell_velocity(tabs[m], e_f, ax, j)
        v_cmd[i] = v
        vop = a.v_op_h if ax < 2 else a.v_op_z
        rate[i] = max(vop, np.linalg.norm(v) / 2) / a.rate_beats        # ramp in at most 2 rate_beats
    x_ref = np.zeros((B, a.call_beats + 1, xref_width(a))); u_ff = np.zeros((B, a.call_beats + 1, 24))
    for m, tab in enumerate(tabs):
        sel = np.array([cells[c][0] == m for c in cell_of])
        x_ref[sel], u_ff[sel], _ = C.build_refs(v_cmd[sel], tab, e_f, T, rate[sel], **ref_kw(a, lead))
    dither = np.zeros((B, a.call_beats + 1, 24))
    if B > nc:
        dither[nc:] = rng.normal(0, a.dither, (B - nc, a.call_beats + 1, 24))
    return cells, cell_of, v_cmd, x_ref, u_ff, dither


def run_cell_pass(chain, fl, a, tabs, e_f, T, B, seed, clock=None, label="ILC pass", Ka=None, lead=None):
    from hover import mjx_command as MC
    cells, cell_of, v_cmd, x_ref, u_ff, dither = cell_pass_inputs(a, tabs, e_f, T, B, np.random.default_rng(seed), lead)
    st, c_rest = MC.release_carry(fl, B)
    log(f"{label}: {B} flights x {a.call_beats} wingbeats ({len(cells)} clean cells + {B - len(cells)} dithered copies)")
    S, U, A, _, secs = chain.run(st, c_rest, np.zeros((B, a.call_beats, 12)), x_ref, u_ff, dither, np.repeat((a.Ka if Ka is None else Ka)[None], B, 0), 1,
                                 log=log, lost_tilt=math.radians(a.pass_tilt))
    win = tuple(a.window)
    rows = []
    for ci, (m, ax, j) in enumerate(cells):
        vc = C.cell_velocity(tabs[m], e_f, ax, j)
        r = C.cell_pass(S, U, A, [ci], vc, win, T, a.u_max) if ci < B else {"passed": False, "alive": False, "sat": 1.0}
        sp = float(np.linalg.norm(vc))
        err = abs(r["along"]) if "along" in r else r.get("v", np.inf)
        r["good"] = bool(r["alive"] and r["sat"] <= a.good_sat and err <= max(0.05, 0.5 * sp)
                         and r.get("cross", 0.0) <= a.good_cross)
        rows.append(r)
    env = C.envelope(rows, cells, tabs[0]["speeds"], len(tabs))
    return {"cells": cells, "cell_of": cell_of, "S": S, "U": U, "A": A, "u_ff": u_ff[:, :a.call_beats], "dither": dither[:, :a.call_beats],
            "x_ref": x_ref, "rows": rows, "env": env, "secs": secs}


def diag_local_models(p, A_, B_, Ka, window):
    out = []
    for ci, cell in enumerate(p["cells"]):
        fl = np.flatnonzero(p["cell_of"] == ci)
        r = C.local_models(p["S"], p["U"], p["A"], fl[1:], window, A_, B_, Ka)
        if r is not None:
            out.append({"cell": cell, **r})
    return out


def eval_inputs(a, tab, e_f, T, B, NB, vmax_pre, K_hover):
    """Eval batch: 75% servo at V_op (the last quarter of them with one verify-sized push at their second
    segment start + 5), 12.5% servo with magnitudes U[0.1, V_max_pre], 12.5% baseline LQR on the schedules of
    the first 12.5%."""
    rng = np.random.default_rng(a.seed)
    n_vop = int(0.75 * B); n_uni = B // 8; n_base = B - n_vop - n_uni
    sv = C.random_schedules(rng, n_vop, NB, e_f, speeds=vop(a))
    su = C.random_schedules(rng, n_uni, NB, e_f, mag_mode="uniform", vmax={k: max(0.1, v) for k, v in vmax_pre.items()})
    sched = {k: np.concatenate([sv[k], su[k], sv[k][:n_base]]) for k in ("v_cmd", "cmd")}
    ms = max(sv["seg"].shape[1], su["seg"].shape[1])
    pad = lambda s: np.concatenate([s, -np.ones((len(s), ms - s.shape[1], 4))], 1)
    sched["seg"] = np.concatenate([pad(sv["seg"]), pad(su["seg"]), pad(sv["seg"][:n_base])])
    arm = np.array(["vop"] * n_vop + ["uniform"] * n_uni + ["baseline"] * n_base)
    x_ref, u_ff, v_ref = C.build_refs(sched["v_cmd"], tab, e_f, T, rates(a, B), a.descend_after_stop, a.descend_delay,
                                      **ref_kw(a))
    x_ref[arm == "baseline", :, :12] = C.baseline_refs(v_ref[arm == "baseline"], T)
    u_ff[arm == "baseline"] = 0.0
    Ka = np.repeat(a.Ka_fly[None], B, 0)
    Ka[arm == "baseline"] = C.hover_ka(K_hover)
    kicks = np.zeros((B, NB, 12))
    push = -np.ones(B, int)
    pv = list(HG.VERIFY_PUSHES.values())
    for n, i in enumerate(range(n_vop - n_vop // 4, n_vop)):
        k = int(sched["seg"][i, 1, 0]) + 5
        if k < NB:
            kicks[i, k] = pv[n % len(pv)]
            push[i] = k
    return sched, arm, x_ref, u_ff, Ka, kicks, push


def arm_metrics(S, U, A, sched, x_ref, T, e_f, arm, push, u_max, n_beats):
    """summarize() per arm (vop = all V_op servo flights; vop_nopush / vop_push; uniform; baseline; servo_twin =
    the V_op flights the baseline shares schedules with)."""
    S, U, A, x_ref = S[:, :n_beats + 1], U[:, :n_beats], A[:, :n_beats + 1], x_ref[:, :n_beats + 1]
    segs, fls = C.segment_metrics(S, U, A, sched, x_ref, T, e_f=e_f, u_max=u_max)
    n_base = int((arm == "baseline").sum())
    groups = {"vop": arm == "vop", "vop_nopush": (arm == "vop") & (push < 0), "vop_push": (arm == "vop") & (push >= 0),
              "uniform": arm == "uniform", "baseline": arm == "baseline",
              "servo_twin": np.isin(np.arange(len(arm)), np.flatnonzero(arm == "vop")[:n_base])}
    out = {}
    for g, sel in groups.items():
        ids = set(np.flatnonzero(sel).tolist())
        if ids:
            out[g] = C.summarize([s for s in segs if s["flight"] in ids], [f for f in fls if f["flight"] in ids])
    return out, segs, fls


def G(name, value, op, thr):
    ok = bool(value == value and (value <= thr if op == "<=" else value >= thr))
    return {"gate": name, "value": float(value), "op": op, "threshold": thr, "pass": ok}


def gates_learn(m, env_post, chosen):
    v, vn, vp, b, tw = m["vop"], m["vop_nopush"], m.get("vop_push", {}), m["baseline"], m["servo_twin"]
    rows = [G("G1 survival V_op servo (all)", v["survival"], ">=", 0.99),
            G("G1 survival no-push (target 1.0)", vn["survival"], ">=", 1.0),
            G("G1 survival push flights", vp.get("survival", float("nan")), ">=", 0.99),
            G("G2 follow rate (all segments)", v["follow_rate"], ">=", 0.95)]
    rows += [G(f"G2 follow rate {c}", d["pass_rate"], ">=", 0.90) for c, d in v["per_cmd"].items()]
    rows += [G("G3 |along| median", v["along_abs_median"], "<=", 0.02), G("G3 |along| p95", v["along_abs_p95"], "<=", 0.05),
             G("G3 cross p95", v["cross_p95"], "<=", 0.05),
             G("G4 hover hold p95 max|p-p_ref|", v["hover_hold_p95"], "<=", 0.05),
             G("G4 hover |v| median", v["hover_v_median"], "<=", 0.01), G("G4 hover |v| p95", v["hover_v_p95"], "<=", 0.02),
             G("G4 hover drift p95", v["hover_drift_p95"], "<=", 0.02),
             G("G5 RMS |p-p_ref| median", v["rms_dev_median"], "<=", 0.04), G("G5 max |p-p_ref| p95", v["max_dev_p95"], "<=", 0.15),
             G("G5 rise time median [beats]", v["rise_median"], "<=", 8),
             G("G6 tilt deviation p95 [deg]", v["max_tilt_dev_p95_deg"], "<=", 15),
             G("G6 max tilt of survivors [deg]", v["max_tilt_survivors_deg"], "<=", 45),
             G("G6 saturated settle beats", v["sat_frac_mean"], "<=", 0.05)]
    if env_post is not None:
        rows += [G(f"G7 post-ILC V_max {k} (map {chosen})", env_post[chosen][k], ">=", 0.3) for k in C.AXES]
    rows.append(G("G8 servo/baseline |along| median", tw["along_abs_median"] / max(b["along_abs_median"], 1e-12), "<=", 0.5))
    return rows


def print_gates(rows):
    for r in rows:
        log(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['gate']:40s} {r['value']:10.4g} {r['op']} {r['threshold']}")


def brief(sm):
    return (f"survival {sm['survival'] * 100:.1f}% | follow {sm['follow_rate'] * 100:.1f}% of {sm['n_segments']} | |along| med "
            f"{sm['along_abs_median']:.4f} p95 {sm['along_abs_p95']:.4f} | cross p95 {sm['cross_p95']:.4f} | hover hold p95 "
            f"{sm['hover_hold_p95'] * 100:.1f} cm, |v| med {sm['hover_v_median']:.4f} | rms dev med {sm['rms_dev_median'] * 100:.1f} cm | "
            f"rise med {sm['rise_median']:.0f} | sat {sm['sat_frac_mean'] * 100:.1f}%")


# ---------------------------------------------------------------------------------------------- learn

def device_linearize(a, bird, rel, fp, fl):
    """hover.lqr's one-wingbeat central differences as one batched device call. --lin x64 (default) switches JAX to
    float64 for this call only (a float64 Flight of the same released state, the model's own solver settings, as
    the CPU reference: matches hover.lqr to round-off; 67 flights x 1 wingbeat take ~1 s on an RTX 5070 Ti) and
    back to float32 for the flights. Returns (A, B, drift, compile / cache-load seconds)."""
    import jax
    import jax.numpy as jnp
    from hover import mjx_flight as F
    to64 = a.lin == "x64" and not jax.config.jax_enable_x64
    if to64:
        jax.config.update("jax_enable_x64", True)
    try:
        fl_l = F.Flight(bird, rel, fp.theta, fp.K, k_phi=fp.phase_lock, ls_iterations=a.ls_iterations,
                        disable_contact=a.no_contact) if a.lin == "x64" else fl
        S_ = fl_l.static
        perts, dus = F.lqr_tasks(fp.K)
        flin = jax.jit(jax.vmap(lambda ar, p, du: (F.one_beat(S_, ar, p, du),), in_axes=(None, 0, 0)))
        args = (fl_l.arrays, jnp.asarray(perts, F.fdtype()), jnp.asarray(dus, F.fdtype()))
        t = time.time()
        comp = flin.lower(*args).compile()
        tl = time.time() - t
        (o,) = comp(*args)
        A_, B_, drift = F.ab_from_tasks(np.asarray(o, float), fp.K)
    finally:
        if to64:
            jax.config.update("jax_enable_x64", False)
    return A_, B_, drift, tl


def cmd_learn(a):
    gate_arg(a)
    jax = setup_jax(a)
    import jax.numpy as jnp
    from hover import mjx_command as MC
    from hover import mjx_flight as F
    from hover.lqr import lqr
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    clock = Clock()
    clock.stages["imports"] = clock.t - T_START
    timings = {"first_call": {}, "calls": {}}
    # 1. release
    fl, bird, rel, fp = HG.flight(a, 0.13)
    S_, K = fl.static, fp.K
    T = S_.n_step * S_.dt
    e_f = C.forward_axis(rel.q_ref)
    log(f"release: dt {S_.dt * 1e3:.5f} ms, {S_.n_step} steps/wingbeat, T {T:.5f} s, e_f {np.round(e_f, 4).tolist()} | "
        f"{HG.knobs(fl, a)} | backend {jax.default_backend()}")
    clock.stage("release")
    # 2. device linearize (x64 by default: the float32 FD model is 17% / 12% off in A / B, systematically)
    A_, B_, drift, tl = device_linearize(a, bird, rel, fp, fl)
    timings["first_call"][f"linearize {len(F.lqr_tasks(K)[0])}x1 ({a.lin})"] = tl
    K_hover = lqr(A_, B_)
    eig = lambda M: float(np.abs(np.linalg.eigvals(M)).max())
    lin = {"open_eig": eig(A_), "hover_closed_eig": eig(A_ - B_ @ K_hover), "drift": drift}
    ref = HG.compare_ab(A_, B_, np.load(a.ref_lqr)) if a.ref_lqr and Path(a.ref_lqr).exists() else None
    log(f"linearize: open |eig| {lin['open_eig']:.3f}, hover LQR closed {lin['hover_closed_eig']:.3f}"
        + (f" (vs {a.ref_lqr}: closed {ref['closed_eig_max_ref']:.3f}, A err {ref['A_rel_err']:.3f}, B err {ref['B_rel_err']:.3f})" if ref else ""))
    clock.stage("linearize")
    # 3. host design
    tabs = [C.steady_state_tables(A_, B_, drift, e_f, tm, a.u_max, C.SPEEDS_H[:a.n_speeds], C.SPEEDS_Z[:a.n_speeds], log=log)
            for tm in a.tilt_maps]
    Ka, rho = C.servo_gains(A_, B_, a.q_int, weights=weights(a))
    a.Ka = Ka
    fly_w = {**weights(a), **({"q_vel": a.fly_q_vel} if a.fly_q_vel else {})}
    a.Ka_fly, rho_fly = (C.servo_gains(A_, B_, a.fly_q_int or a.q_int, weights=fly_w) if (a.fly_q_int or a.fly_q_vel)
                         else (Ka, rho))
    if a.Ka_fly is not Ka:
        log(f"flight gain (q_int {a.fly_q_int or a.q_int:g}, q_vel {fly_w['q_vel']:g}): rho {rho_fly:.3f}; the ILC passes use "
            f"the learning gain (q_int {a.q_int:g}): its integral settles within the ILC window")
    a.H = C.servo_hessian(A_, B_, a.q_int, weights=weights(a)) if a.alloc_iters else None
    log(f"servo: |eig| of the augmented closed loop {rho:.3f} (hover LQR {lin['hover_closed_eig']:.3f}); input allocation "
        + (f"in the Q-function metric, {a.alloc_iters} FISTA iterations" if a.alloc_iters else "clip"))
    clock.stage("design")
    # 4. linear-model sanity sim
    sch = C.random_schedules(np.random.default_rng(a.seed + 1), 200, 120, e_f, speeds=vop(a))
    xr, uf, _ = C.build_refs(sch["v_cmd"], tabs[0], e_f, T, rates(a, 200), a.descend_after_stop, a.descend_delay,
                             **ref_kw(a))
    Sl, Ul = C.linear_sim(A_, B_, drift, a.Ka_fly, xr, uf, u_max=a.u_max, zeta_max=a.zeta_max, H=a.H, iters=a.alloc_iters)
    lin_sim = C.summarize(*C.segment_metrics(Sl, Ul, np.ones(Sl.shape[:2], bool), sch, xr, T, e_f=e_f, u_max=a.u_max))
    log("linear-model sim (200 x 120 beats, tilt map 0): " + brief(lin_sim))
    clock.stage("linear_sim")
    # 5. ILC / envelope pass
    chain = MC.Chain(fl, a.batch, a.call_beats, a.u_max, a.zeta_max, H=a.H, alloc_iters=a.alloc_iters)
    passes, tabs_used = [], tabs
    for it in range(a.ilc_iters):
        fly_gain = it >= a.ilc_iters - a.ilc_fly_passes          # the last --ilc-fly-passes passes fly the flight gain
        p = run_cell_pass(chain, fl, a, tabs_used, e_f, T, a.batch, a.seed + 100 + it,
                          label=f"ILC pass {it + 1}" + (" (flight gain)" if fly_gain else ""), Ka=a.Ka_fly if fly_gain else None,
                          lead=a.learn_ff_lead)
        if it == 0:
            timings["first_call"][f"flight program {a.batch}x{a.call_beats}"] = chain.compile_s
            env_pre = p["env"]
            log(f"pre-ILC envelope (V_max per axis): " + "; ".join(f"tilt {tabs[m]['tilt_max_deg']:g}: {env_pre[m]}" for m in env_pre))
        good = np.array([r["good"] for r in p["rows"]])
        meas, info = C.ilc_update(tabs_used, p["S"], p["U"], p["A"], p["u_ff"], p["dither"], p["cell_of"], p["cells"],
                                  e_f, T, window=tuple(a.window), u_max=a.u_max, good=good)
        tabs_used, xinfo = C.extrapolate_tables(meas, tabs, good, p["cells"], max_factor=a.extrap)
        p.update(ilc_info=info, extrap=xinfo, good=good)
        passes.append(p)
        log(f"ILC {it + 1}: envelope " + "; ".join(f"tilt {tabs[m]['tilt_max_deg']:g}: {p['env'][m]}" for m in p["env"])
            + f" | good cells {int(good.sum())}/{len(good)}, max |du| {max(r.get('du_inf', 0) for r in info):.4f}; good speeds "
            + "; ".join(f"{k} <= {max(v['good'], default=0):g}" for k, v in xinfo.items()))
    env_last = passes[-1]["env"]
    chosen = max(range(len(tabs)), key=lambda m: (min(env_last[m].values()), -m))     # ties -> map 0 (5 deg)
    log(f"chosen tilt map {tabs[chosen]['tilt_max_deg']:g} deg (last-pass envelope {env_last[chosen]})")
    timings["calls"]["ilc_pass"] = [s for p in passes for s in p["secs"]]
    diag = diag_local_models(passes[0], A_, B_, Ka, tuple(a.window))
    clock.stage("ilc_pass")
    # 6. evaluation
    tab = tabs_used[chosen]
    NB = max(a.eval_beats, a.continue_to or 0)
    sched, arm, x_ref, u_ff, Ka_b, kicks, push = eval_inputs(a, tab, e_f, T, a.batch, NB, env_last[chosen], K_hover)
    st, c_rest = MC.release_carry(fl, a.batch)
    n1 = a.eval_beats // a.call_beats
    log(f"eval: {a.batch} flights ({(arm == 'vop').sum()} V_op servo incl. {(push >= 0).sum()} pushed, {(arm == 'uniform').sum()} "
        f"uniform, {(arm == 'baseline').sum()} baseline) x {a.eval_beats} wingbeats = {n1} calls")
    H_b = eval_H(a, arm)
    S1, U1, A1, st, secs = chain.run(st, c_rest, kicks, x_ref, u_ff, np.zeros_like(u_ff), Ka_b, n1, log=log, H=H_b)
    timings["calls"]["eval"] = secs
    clock.stage("eval")
    m120, _, _ = arm_metrics(S1, U1, A1, sched, x_ref, T, e_f, arm, push, a.u_max, a.eval_beats)
    for g in ("vop", "vop_push", "uniform", "baseline"):
        if g in m120:
            log(f"eval {a.eval_beats} beats, {g:9s}: " + brief(m120[g]))
    C.save_policy(out / "policy.npz", tabs=tabs_used, chosen=chosen, Ka=a.Ka_fly, K_hover=K_hover, e_f=e_f, T=T,
                  v_op=list(vop(a)), u_max=a.u_max, zeta_max=a.zeta_max, rate_beats=a.rate_beats, A=A_, B=B_, drift=drift,
                  extra={"tabs_lp_u": np.stack([t["u"] for t in tabs]), "tabs_lp_att": np.stack([t["att"] for t in tabs]),
                         "weights": np.array(json.dumps({**weights(a), "q_int": a.q_int})),
                         "descend_after_stop": a.descend_after_stop, "descend_delay": a.descend_delay, "alloc_iters": a.alloc_iters, "Ka_learn": Ka,
                         "zeta_handoff": a.zeta_handoff, "zeta_gate": -1 if a.zeta_gate is None else a.zeta_gate, "stop_rate": a.stop_rate,
                         "ff_lead": a.ff_lead,
                         **({"H": a.H} if a.H is not None else {})})
    tr = {"S": S1.astype(np.float32), "U": U1.astype(np.float32), "alive": A1, "v_cmd": sched["v_cmd"], "seg": sched["seg"],
          "arm": arm, "push": push, "kicks_at": np.array([kicks[i, k] if k >= 0 else np.zeros(12) for i, k in enumerate(push)]),
          "x_ref": x_ref.astype(np.float32), "u_ff": u_ff.astype(np.float32), "Ka": np.stack([a.Ka_fly, C.hover_ka(K_hover)]),
          "ilc_S": passes[0]["S"].astype(np.float32), "ilc_A": passes[0]["A"], "ilc_cell_of": passes[0]["cell_of"]}
    np.savez(out / "traces.npz", **tr)
    np.savez(out / "ilc_passes.npz", **{f"p{i}_{k}": np.asarray(p[k]) for i, p in enumerate(passes)
                                        for k in ("S", "U", "A", "u_ff", "x_ref", "cell_of", "good")})
    metrics = {"eval_120": m120, "linear_sim": lin_sim, "linearize": lin, "linearize_vs_ref": ref, "servo_rho": rho,
               "lp_envelope": {tabs[m]["tilt_max_deg"]: tabs[m]["lp_envelope"] for m in range(len(tabs))},
               "table_notes": [t["note"] for t in tabs], "env_pre": env_pre, "env_per_pass": [p["env"] for p in passes],
               "good_per_pass": [p["good"] for p in passes], "extrap_per_pass": [p["extrap"] for p in passes], "chosen_map": chosen,
               "chosen_tilt_deg": tabs[chosen]["tilt_max_deg"], "ilc_pass_rows": passes[0]["rows"],
               "ilc_info": [p["ilc_info"] for p in passes], "local_models": diag, "config": cfg(a)}
    jdump(out / "metrics.json", metrics)
    clock.stage("metrics_write")
    budget = time.time() - T_START
    timings.update(stages=clock.stages, budget_wall_s=budget, B=a.batch, eval_beats=a.eval_beats,
                   shapes_on_budget_path=list(timings["first_call"]))
    jdump(out / "timings.json", timings)
    log(f"== BUDGET: learn -> eval-{a.eval_beats} wall {budget:.1f} s from process start (G10 <= 240 s: "
        f"{'PASS' if budget <= 240 else 'FAIL'})")
    # 8. untimed continuation
    env_post = None
    if a.continue_to and a.continue_to > a.eval_beats:
        n2 = (a.continue_to - a.eval_beats) // a.call_beats
        S2, U2, A2, st, secs2 = chain.run(st, c_rest, kicks, x_ref, u_ff, np.zeros_like(u_ff), Ka_b, n2, start=a.eval_beats, log=log,
                                          H=H_b)
        S_all = np.concatenate([S1[:, :-1], S2], 1); U_all = np.concatenate([U1, U2], 1); A_all = np.concatenate([A1[:, :-1], A2], 1)
        timings["calls"]["continue"] = secs2
        clock.stage("continue_to_%d" % a.continue_to)
        m240, segs, fls = arm_metrics(S_all, U_all, A_all, sched, x_ref, T, e_f, arm, push, a.u_max, a.continue_to)
        for g in ("vop", "vop_nopush", "vop_push", "uniform", "baseline", "servo_twin"):
            if g in m240:
                log(f"eval {a.continue_to} beats, {g:10s}: " + brief(m240[g]))
        metrics[f"eval_{a.continue_to}"] = m240
        tr.update(S=S_all.astype(np.float32), U=U_all.astype(np.float32), alive=A_all)
        np.savez(out / "traces.npz", **tr)
    if not a.no_verify:
        pv = run_cell_pass(chain, fl, a, tabs_used, e_f, T, a.batch, a.seed + 200, label="envelope verify (post-ILC)", Ka=a.Ka_fly)
        env_post = pv["env"]
        log("post-ILC envelope: " + "; ".join(f"tilt {tabs[m]['tilt_max_deg']:g}: {env_post[m]}" for m in env_post))
        metrics.update(env_post=env_post, verify_pass_rows=pv["rows"], local_models_post=diag_local_models(pv, A_, B_, Ka, tuple(a.window)))
        _, info2 = C.ilc_update(tabs_used, pv["S"], pv["U"], pv["A"], pv["u_ff"], pv["dither"], pv["cell_of"], pv["cells"], e_f, T,
                                window=tuple(a.window), u_max=a.u_max)
        metrics["verify_residual_du_inf"] = [r.get("du_inf") for r in info2]
        clock.stage("envelope_verify")
    mg = metrics.get(f"eval_{a.continue_to}", m120)
    gates = gates_learn(mg, env_post, chosen)
    gates.append(G("G10 warm budget wall [s]", budget, "<=", 240))
    gates += [G(f"G10 first call {k} [s]", v, "<=", 10) for k, v in timings["first_call"].items()]
    gates.append(G("G10 compiled shapes on the budget path", len(timings["first_call"]), "<=", 2))
    metrics["gates"] = gates
    metrics["gates_on"] = f"eval_{a.continue_to}" if f"eval_{a.continue_to}" in metrics else "eval_120"
    log(f"gates (on {metrics['gates_on']}):")
    print_gates(gates)
    timings["stages"] = clock.stages
    jdump(out / "timings.json", timings)
    jdump(out / "metrics.json", metrics)
    log(f"-> {out}")


def eval_H(a, arm):
    """Per-flight allocation metric: the servo's H, and I (= plain clip) for the baseline arm."""
    if a.H is None:
        return None
    H = np.repeat(a.H[None], len(arm), 0)
    H[arm == "baseline"] = np.eye(a.H.shape[0])
    return H


def weights(a):
    return {"q_pos": a.q_pos, "q_att": a.q_att, "q_vel": a.q_vel, "q_w": a.q_w, "r": a.r}


def cfg(a):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k not in ("Ka", "Ka_fly", "H")}


# ---------------------------------------------------------------------------------------------- dt robustness

def cmd_dt(a):
    setup_jax(a)
    from hover import mjx_command as MC
    run = Path(a.run)
    pol = C.load_policy(run / "policy.npz")
    tr = np.load(run / "traces.npz")
    sel = np.flatnonzero((tr["arm"] == "vop") & (tr["push"] < 0))[:a.flights]
    NB = a.beats
    sched = {"v_cmd": tr["v_cmd"][sel, :NB], "cmd": None, "seg": tr["seg"][sel]}
    e_f, T = pol["e_f"], float(pol["T"])
    vop = pol["v_op"]
    rate = np.array([vop[0], vop[0], vop[1]]) / float(pol["rate_beats"])
    x_ref, u_ff, _ = C.build_refs(sched["v_cmd"], pol["tab"], e_f, T, rate, float(pol.get("descend_after_stop", 0.0)),
                                  int(pol.get("descend_delay", 0)), **ref_kw(pol_ref_args(pol)))
    res = {}
    for dt in a.dt_ms:
        t0 = time.time()
        fl, bird, rel, fp = HG.flight(a, dt)
        Tn = fl.static.n_step * fl.static.dt
        chain = MC.Chain(fl, len(sel), a.call_beats, float(pol["u_max"]), float(pol["zeta_max"]), H=pol.get("H"),
                         alloc_iters=int(pol.get("alloc_iters", 0)))
        st, c_rest = MC.release_carry(fl, len(sel))
        log(f"dt {dt}: snapped {fl.static.dt * 1e3:.5f} ms, {fl.static.n_step} steps/wingbeat (T {Tn:.5f} vs policy {T:.5f})")
        S, U, A, _, secs = chain.run(st, c_rest, np.zeros((len(sel), NB, 12)), x_ref, u_ff, np.zeros_like(u_ff),
                                     np.repeat(pol["Ka"][None], len(sel), 0), NB // a.call_beats, log=log)
        segs, fls = C.segment_metrics(S, U, A, sched, x_ref, T, e_f=e_f, u_max=float(pol["u_max"]))
        sm = C.summarize(segs, fls)
        log(f"dt {dt} ({len(sel)} flights x {NB} beats): " + brief(sm))
        gates = [G(f"G9 dt {dt} survival", sm["survival"], ">=", 1.0), G(f"G9 dt {dt} follow rate", sm["follow_rate"], ">=", 0.9),
                 G(f"G9 dt {dt} |along| median", sm["along_abs_median"], "<=", 0.03)]
        print_gates(gates)
        res[str(dt)] = {"summary": sm, "gates": gates, "compile_s": chain.compile_s, "call_s": secs, "wall_s": time.time() - t0}
    jdump(run / "dt.json", res)
    log(f"-> {run / 'dt.json'}")


# ---------------------------------------------------------------------------------------------- sensors

def design_n(a, T, log_=log):
    """The IMU-only observer (att, w, acc [+ bias states]) from the x64 sensor model, at --n-design-noise."""
    from hover import reduced as Rd
    from hover import sensors as Sn
    model = Rd.load_model(a.sensor_lqr)
    noise = Sn.SensorNoise(scale=a.n_design_noise)
    tried = []
    for bias in ([a.n_bias] if a.n_bias else []) + ["acc", "acc,att_xy", None]:
        try:
            ctl = Rd.design(model, a.n_obs, "lqg", noise=noise, beat_s=T, bias=bias)
        except Exception as e:                       # noqa: BLE001 (undetectable bias sets raise)
            tried.append({"bias": bias, "error": str(e)[:200]})
            continue
        rho = ctl["info"]["rho_closed_loop"]
        tried.append({"bias": bias, "rho": rho})
        if rho < 1:
            log_(f"N observer: obs {a.n_obs}, bias {bias}, design noise x{a.n_design_noise}: closed-loop rho {rho:.3f}")
            return ctl, tried
    raise RuntimeError(f"no stable IMU-only LQG: {tried}")


def cmd_sensors(a):
    """Sensor ladder: P = mocap position (5 ms latency) + IMU attitude filter + rate: an LQG on pos,att,w with
    attitude-bias states (the filter's tilt error changes with the flight condition) + the servo with a soft
    position integral (--p-q-int) on the estimate; N = IMU only (att, w: the Prompt 1 LQG N_lqgV1 with its own gain).
    The true-state tables are relearned IN ESTIMATE COORDINATES by --sensor-ilc passes of constant-command sensor
    flights; N's cells also move their trim by the linear trim sensitivity x their TRUE mean-motion velocity error
    (--n-vel-ilc: learned in simulation, the policy reads only the IMU). P's observer predicts with the per-beat
    offset that makes the reference its model's own trajectory (--p-cz-seq, hover.command.trim_consistent_cz) and
    flies the true-state tables (--p-tables true). Then 2 variants x noise scales x seeds random command flights."""
    jax = setup_jax(a)
    import jax.numpy as jnp
    from hover import mjx_command as MC
    from hover import mjx_flight as F
    from hover import reduced as Rd
    from hover import sensors as Sn
    run = Path(a.run)
    pol = C.load_policy(run / "policy.npz")
    e_f, T, tab = pol["e_f"], float(pol["T"]), pol["tab"]
    A_, B_ = pol["A"], pol["B"]
    fl, bird, rel, fp = HG.flight(a, 0.13)
    if a.p_obs:
        P = Rd.design(Rd.load_model(a.sensor_lqr), a.p_obs, "lqg", beat_s=T, bias=a.p_bias or None, bias_walk=a.p_bias_walk,
                      noise=Sn.SensorNoise(scale=a.p_design_noise, pos_latency_ms=a.pos_latency_ms),
                      process_scale=a.p_process_scale)
        log(f"P observer: designed on {a.p_obs} (bias {a.p_bias}, walk {a.p_bias_walk:g}/beat, noise x{a.p_design_noise:g}, "
            f"process x{a.p_process_scale:g}, latency {a.pos_latency_ms} ms): "
            f"closed-loop rho {P['info']['rho_closed_loop']:.3f}")
    else:
        P = Rd.load_controller(a.p_ctl)
        if str(P["kind"]) != "lqg" or "dr" in P or list(P["obs"]) != list(Sn.parse_obs("pos,att,w")):
            raise SystemExit(f"{a.p_ctl} is not a plain pos,att,w LQG")
    if str(P["kind"]) != "lqg" or "dr" in P or not set(Sn.parse_obs("pos")) <= set(np.asarray(P["obs"]).tolist()):
        raise SystemExit("the P observer must be a plain LQG that reads pos")
    Ka_P, cp_P = pol["Ka"], 1.0
    if a.p_q_int >= 0 and not a.p_own_gain:        # a softer position integral for the estimate-fed servo
        wts = json.loads(str(pol["weights"])) if "weights" in pol else {}
        Ka_P, rho_P = C.servo_gains(A_, B_, a.p_q_int, weights={k: v for k, v in wts.items() if k != "q_int"})
        log(f"P: servo gain with q_int {a.p_q_int:g} (rho {rho_P:.3f})")
    elif a.p_own_gain:                             # the observer's own LQR gain on its design states, no integral
        Ka_P = np.zeros((24, 15)); Ka_P[:, np.asarray(P["xs"], int)] = P["K"]; cp_P = 0.0
        log("P: the observer's own LQR gain (no position integral) instead of the servo gain")
    if a.n_variant == "att_w":
        # F6: IMU attitude + rates only (the Prompt 1 LQG, unchanged, with its own gain on the estimate of att, vel,
        # w); no velocity servo: the command enters through the (sensor-ILC) att / u_ff references only
        N, n_tried = Rd.load_controller(a.n_ctl), []
        Ka_N = np.zeros((24, 15)); Ka_N[:, np.asarray(N["xs"], int)] = N["K"]
        if a.n_no_vel:
            Ka_N[:, 6:9] = 0.0
        rho_N = float("nan")
        n_cv = 0.0
        log(f"N observer: {a.n_ctl} (obs {Sn.obs_names(N['obs'])}), its own gain, no velocity servo")
    else:
        N, n_tried = design_n(a, T)
        wts = json.loads(str(pol["weights"])) if "weights" in pol else {}
        Ka_N, rho_N = C.servo_gains_vel(A_, B_, T, weights={k: v for k, v in wts.items() if k != "q_int"})
        n_cv = 1.0
        log(f"velocity servo (N): augmented closed-loop rho {rho_N:.3f}")
    nz = max(P["Az"].shape[0], N["Az"].shape[0])
    obsP, obsN = MC.pad_observer(P, nz, 24), MC.pad_observer(N, nz, 24)
    names = ("P", "N")
    variants = {"P": (obsP, Ka_P, cp_P, 0.0, Sn.n_non_imu(P["obs"])), "N": (obsN, Ka_N, 0.0, n_cv, Sn.n_non_imu(N["obs"]))}
    noise = Sn.SensorNoise(pos_latency_ms=a.pos_latency_ms)
    alloc = int(pol.get("alloc_iters", 0)) if "H" in pol else 0
    sp = MC.CmdSensorPipeline(fl, noise, nz, att_mode="filter", u_max=float(pol["u_max"]), zeta_max=float(pol["zeta_max"]),
                              alloc_iters=alloc)
    H_of = {"P": pol["H"] if alloc else None, "N": np.eye(24) if alloc else None}
    progs = {}

    def fly(var_of, x_ref, u_ff, scales, seed):
        n, NB = x_ref.shape[0], x_ref.shape[1]
        ctl = {k: [] for k in list(obsP) + ["Ka", "x_ref", "u_ff", "cp", "cv", "T", "cz_seq"] + (["H"] if alloc else [])}
        for i, v in enumerate(var_of):
            ob, Ka_, cp, cv, _ = variants[v]
            for k in ob:
                ctl[k].append(ob[k])
            if v == "P" and a.p_cz_seq:
                cs = np.zeros((NB, nz)); cs[:, :P["Az"].shape[0]] = C.trim_consistent_cz(P, x_ref[i], u_ff[i])
            else:
                cs = np.tile(ob["cz"], (NB, 1))
            ctl["cz_seq"].append(cs)
            ctl["Ka"].append(Ka_); ctl["x_ref"].append(x_ref[i]); ctl["u_ff"].append(u_ff[i])
            ctl["cp"].append(cp); ctl["cv"].append(cv); ctl["T"].append(T)
            if alloc:
                ctl["H"].append(H_of[v])
        ctl = {k: jnp.asarray(np.array(v), F.fdtype()) for k, v in ctl.items()}
        keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray((np.arange(n) * 7919 + seed) % (2 ** 31)))
        args = (fl.arrays, jnp.zeros((n, NB, 12), F.fdtype()), keys, jnp.asarray(scales, F.fdtype()), ctl)
        if NB not in progs:
            t = time.time()
            progs[NB] = MC.batch_cmd_sensor_flyer(fl, sp, NB).lower(*args).compile()
            log(f"sensor program {n} x {NB}: compile / cache load {time.time() - t:.1f} s")
        t = time.time()
        out = [np.asarray(x) for x in progs[NB](*args)]
        log(f"  {n} sensor flights x {NB} beats: {time.time() - t:.1f} s")
        return out
    res = {"n_variant": a.n_variant, "n_tried": n_tried,
           "N_info": {k: v for k, v in N["info"].items() if k in ("rho_closed_loop", "bias", "observability")},
           "P_info": {k: P["info"].get(k) for k in ("rho_closed_loop",)}, "rho_vel_servo": rho_N, "rows": {}, "ilc": []}
    # sensor ILC: the tables in estimate coordinates, per variant (map index = variant)
    vo = [float(x) for x in pol["v_op"]]
    a.v_op_h, a.v_op_z, a.v_op_down = vo[0], vo[1], vo[2] if len(vo) > 2 else vo[1]
    a.zeta_handoff, a.zeta_gate, a.stop_rate, a.ff_lead = vars(pol_ref_args(pol)).values()
    a.rate_beats, a.call_beats, a.dither = float(pol["rate_beats"]), a.ilc_beats, 0.0
    tabs_s = [{k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in tab.items()} for _ in names]
    for it in range(a.sensor_ilc):
        cells, cell_of, v_cmd, xr, uf, _ = cell_pass_inputs(a, tabs_s, e_f, T, a.batch, np.random.default_rng(a.seed + 300 + it),
                                                            lead=0)
        var_of = [names[cells[c][0]] for c in cell_of]
        nb = a.ilc_beats
        S_, Y_, X_, U_, A_l = fly(var_of, xr[:, :nb], uf[:, :nb], np.full(a.batch, a.ilc_noise), a.seed + 1000 * (it + 1))
        Sc = S_.copy(); Sc[:, :nb, 3:12] = X_[:, :, 3:12]
        win = tuple(a.window)
        rows = []
        for ci, (m, ax, j) in enumerate(cells):
            vc = C.cell_velocity(tabs_s[m], e_f, ax, j)
            r = (C.cell_pass(S_, U_, A_l, [ci], vc, win, T, float(pol["u_max"])) if ci < a.batch
                 else {"passed": False, "alive": False, "sat": 1.0})
            sp_ = float(np.linalg.norm(vc))
            err = abs(r["along"]) if "along" in r else r.get("v", np.inf)
            r["good"] = bool(r["alive"] and r["sat"] <= a.sensor_good_sat and (m == 1 and a.n_ilc_any or err <= max(0.05, 0.5 * sp_)))
            rows.append(r)
        good = np.array([r["good"] for r in rows])
        env = C.envelope(rows, cells, tabs_s[0]["speeds"], 2)
        meas, info = C.ilc_update(tabs_s, Sc, U_, A_l, uf[:, :nb], np.zeros_like(uf[:, :nb]), cell_of, cells, e_f, T,
                                  window=win, u_max=float(pol["u_max"]), good=good, dv_clip=(0.2, a.n_dv_clip))
        for m_v, g_v in ((0, a.p_vel_ilc), (1, a.n_vel_ilc)):
            if g_v:
                meas, vinfo = C.velocity_ilc(meas, S_, A_l, cell_of, cells, e_f, T, *C.trim_sensitivity(A_, B_), maps=(m_v,),
                                             window=win, gain=g_v, u_max=float(pol["u_max"]))
                log(f"sensor ILC {it + 1}: {names[m_v]} velocity ILC on {len(vinfo)} cells, median |true velocity error| "
                    f"{np.median([np.linalg.norm(r['dv']) for r in vinfo]) if vinfo else float('nan'):.3f} m/s")
        tabs_s, xinfo = C.extrapolate_tables(meas, [tab, tab], good, cells, max_factor=a.extrap)
        log(f"sensor ILC {it + 1} (noise x{a.ilc_noise:g}): envelope P {env[0]} | N {env[1]} | good cells {int(good.sum())}/{len(good)}")
        res["ilc"].append({"env": {names[m]: env[m] for m in env}, "good": good, "extrap": xinfo})
    if a.p_tables == "true":
        tabs_s[0] = tab
        log("P flies the true-state tables (its sensor-ILC tables are discarded)")
    res["p_cz_seq"], res["p_tables"] = bool(a.p_cz_seq), a.p_tables
    # evaluation: variants x noise scales x seeds on random V_op schedules
    rng = np.random.default_rng(a.seed + 7)
    NB = a.beats
    sch = C.random_schedules(rng, a.seeds, NB, e_f, speeds=tuple(float(x) for x in pol["v_op"]))
    rate = np.array([pol["v_op"][0], pol["v_op"][0], pol["v_op"][1]]) / float(pol["rate_beats"])
    das = float(pol.get("descend_after_stop", 0.0))
    refs = {v: C.build_refs(sch["v_cmd"], tabs_s[m], e_f, T, rate, das, int(pol.get("descend_delay", 0)), **ref_kw(a),
                            climb_after_stop=a.p_climb_after_stop if v == "P" else 0.0)[:2]
            for m, v in enumerate(names)}
    jobs = [(v, ns, i) for v in names for ns in a.noise_scales for i in range(a.seeds)]
    x_ref = np.stack([refs[v][0][i, :NB] for v, _, i in jobs]); u_ff = np.stack([refs[v][1][i, :NB] for v, _, i in jobs])
    t = time.time()
    Sg, Yg, Xg, Ug, Ag = fly([v for v, _, _ in jobs], x_ref, u_ff, np.array([ns for _, ns, _ in jobs]), a.seed)
    res["flights_s"] = time.time() - t
    gates = []
    for v in names:
        for ns in a.noise_scales:
            sel = np.array([j for j, (vv, nn, _) in enumerate(jobs) if vv == v and nn == ns])
            fi = np.array([jobs[j][2] for j in sel])
            sub = {"v_cmd": sch["v_cmd"][fi], "seg": sch["seg"][fi]}
            xr_true = refs[v][0][fi]
            segs, fls = C.segment_metrics(Sg[sel], Ug[sel], Ag[sel], sub, xr_true, T, e_f=e_f, u_max=float(pol["u_max"]))
            sm = C.summarize(segs, fls)
            la = C.lost_at(Ag[sel])
            ok = la == NB
            vb = (Xg[sel][:, 10:, 6:9] - Sg[sel][:, 10:NB, 6:9])[ok]
            dev = Sg[sel][:, :, :3] - xr_true[:, :NB + 1, :3]
            drift = np.linalg.norm(dev[:, 120 if NB >= 120 else NB] - dev[:, 60], axis=1)[ok] if NB >= 60 else np.array([])
            sm.update(n_non_imu=variants[v][4], vel_est_bias_mean=vb.mean((0, 1)).tolist() if vb.size else None,
                      vel_est_err_rms=float(np.sqrt((vb ** 2).sum(-1).mean())) if vb.size else None,
                      pos_drift_60_120_median=float(np.median(drift)) if drift.size else None,
                      pos_drift_60_120_p95=float(np.quantile(drift, 0.95)) if drift.size else None)
            res["rows"][f"{v}x{ns:g}"] = sm
            log(f"variant {v} ({variants[v][4]} non-IMU channels) noise x{ns:g}: " + brief(sm)
                + f" | dir-follow {sm['direction_follow'] * 100:.1f}% | vel-est bias {np.round(sm['vel_est_bias_mean'] or [], 4)} | "
                  f"drift 60-120 med {sm['pos_drift_60_120_median']}")
            if v == "P" and ns == 1:
                gates += [G("G12 P x1 survival", sm["survival"], ">=", 0.99), G("G12 P x1 follow rate", sm["follow_rate"], ">=", 0.9),
                          G("G12 P x1 |along| p95", sm["along_abs_p95"], "<=", 0.08), G("G12 P x1 hover hold p95", sm["hover_hold_p95"], "<=", 0.08)]
            if v == "P" and ns == 3:
                gates.append(G("G12 P x3 survival", sm["survival"], ">=", 0.95))
            if v == "N" and ns == 1:
                # direction_follow (the sign along the command) is logged, not gated: it passes a bird sliding
                # sideways at 1.4 m/s under FWD (IMU-only att,w); gate N on the same follow rate as P
                gates += [G("G12 N x1 survival", sm["survival"], ">=", 0.99), G("G12 N x1 follow rate", sm["follow_rate"], ">=", 0.9)]
    res["gates"] = gates
    print_gates(gates)
    jdump(run / "sensors.json", res)
    np.savez(run / "sensor_traces.npz", S=Sg.astype(np.float32), Xh=Xg.astype(np.float32), U=Ug.astype(np.float32), alive=Ag,
             jobs=np.array([f"{v}|{ns}|{i}" for v, ns, i in jobs]))
    log(f"-> {run / 'sensors.json'}")


# ---------------------------------------------------------------------------------------------- CPU reference

def _cpu_job(job):
    from hover import launch as L
    tg, control, dt, n_beats, Ka, x_ref, u_ff, kicks, u_max, zeta_max, contacts, H, iters = job
    bird, rel, fp = HG.load(tg, control, dt)
    if not contacts:          # as the device (--contact off): the scene's floor is 0.25 m below the release COM
        import mujoco
        bird.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    pol = C.CommandPolicy(Ka, x_ref, u_ff, u_max=u_max, zeta_max=zeta_max, H=H, iters=iters)
    import mujoco
    warn = []                 # MuJoCo resets the state on a bad QACC and only warns: count that as the loss
    mujoco.set_mju_user_warning(lambda msg: warn.append((len(pol.U), str(msg))))
    t = time.time()
    try:
        tr = L.fly(bird, rel, fp.theta, n_beats, policy=pol, kicks=kicks)
    finally:
        mujoco.set_mju_user_warning(None)
    U = np.array(pol.U)
    if warn:                  # a warning inside beat k-1 (k = len(U) then): the strobe k is the lost one
        k = warn[0][0]
        tr, U = tr[:k + 1], U[:k]
    return tr, U, time.time() - t, (warn[0] if warn else None)


def cmd_cpu(a):
    run = Path(a.run)
    pol = C.load_policy(run / "policy.npz")
    tr = np.load(run / "traces.npz")
    arm, push = tr["arm"], tr["push"]
    NB = min(a.beats, tr["S"].shape[1] - 1)
    vop_np = np.flatnonzero((arm == "vop") & (push < 0))
    sel = np.r_[vop_np[:a.n_plain], np.flatnonzero((arm == "vop") & (push >= 0))[:a.n_push], np.flatnonzero(arm == "baseline")[:a.n_base]]
    Kas = {"vop": pol["Ka"], "baseline": C.hover_ka(pol["K_hover"])}
    jobs = []
    for i in sel:
        kicks = {int(push[i]): tr["kicks_at"][i]} if push[i] >= 0 else None
        jobs.append((a.tg, a.control, 0.13, NB, Kas["baseline" if arm[i] == "baseline" else "vop"], tr["x_ref"][i].astype(float),
                     tr["u_ff"][i].astype(float), kicks, float(pol["u_max"]), float(pol["zeta_max"]), a.cpu_contacts,
                     (np.eye(24) if arm[i] == "baseline" else pol["H"]) if "H" in pol else None, int(pol.get("alloc_iters", 0))))
    log(f"CPU reference: {len(jobs)} flights x {NB} wingbeats on {a.procs} processes")
    t = time.time()
    with Pool(a.procs) as p:
        res = p.map(_cpu_job, jobs, chunksize=1)
    log(f"  {time.time() - t:.0f} s")
    e_f, T = pol["e_f"], float(pol["T"])
    rows, agree, dv = [], [], []
    for k, i in enumerate(sel):
        cpu, Uc, sec, warn = res[k]
        gpu_full = tr["S"][i][:NB + 1].astype(float)
        ag = tr["alive"][i][:NB + 1]
        gpu = gpu_full[:int(C.lost_at(ag)) + 1]
        sub = {"v_cmd": tr["v_cmd"][i:i + 1, :NB], "seg": tr["seg"][i:i + 1]}
        n_c = len(cpu)
        Sc = np.zeros((1, NB + 1, 12)); Sc[0, :n_c] = cpu
        Ac = np.zeros((1, NB + 1), bool); Ac[0, :n_c] = True
        if n_c <= NB or warn:                                 # the CPU stops at the lost strobe (which it returns)
            Ac[0, n_c - 1] = False
            Sc[0, n_c:] = cpu[-1]
        Ucp = np.zeros((1, NB, 24)); Ucp[0, :len(Uc)] = Uc
        segc, _ = C.segment_metrics(Sc, Ucp, Ac, sub, tr["x_ref"][i:i + 1].astype(float), T, e_f=e_f)
        segg, _ = C.segment_metrics(gpu_full[None], tr["U"][i:i + 1, :NB].astype(float), ag[None], sub, tr["x_ref"][i:i + 1].astype(float), T, e_f=e_f)
        for sc, sg in zip(segc, segg):
            agree.append(sc["passed"] == sg["passed"])
            if sc["alive"] and sg["alive"]:
                vc = sc.get("along", sc.get("v_seg")); vg = sg.get("along", sg.get("v_seg"))
                dv.append(abs(vc - vg))
        err, _ = HG.state_error(cpu[:121], gpu[:121], windows=(120,))
        rows.append({"flight": int(i), "arm": str(arm[i]), "push": int(push[i]), "cpu_flew": n_c == NB + 1 and not warn,
                     "mujoco_warning": list(warn) if warn else None, "gpu_flew": len(gpu) == NB + 1,
                     "err_120": err.get("err_120", err["err_all"]), "cpu_s": sec})
        log(f"  flight {i:4d} {arm[i]:8s} push {push[i]:4d}: CPU {'flew all' if rows[-1]['cpu_flew'] else 'LOST %d' % (n_c - 1)}{' (MuJoCo warning)' if warn else ''} | GPU "
            f"{'flew all' if len(gpu) == NB + 1 else 'LOST %d' % (len(gpu) - 1)} | err/scale 0-120 {rows[-1]['err_120']:.3f} | {sec:.0f} s")
    same = all(r["cpu_flew"] == r["gpu_flew"] for r in rows)
    dv = np.array(dv)
    gates = [G("G11 identical survival", float(same), ">=", 1.0), G("G11 segment verdict agreement", float(np.mean(agree)), ">=", 0.95),
             G("G11 |v_seg cpu - gpu| median", float(np.median(dv)) if dv.size else float("nan"), "<=", 0.005), G("G11 |v_seg cpu - gpu| max", float(dv.max()) if dv.size else float("nan"), "<=", 0.02),
             G("G11 state error / ERR_SCALE (0-120, max over flights)", float(np.nanmax([r["err_120"] for r in rows])), "<=", 1.0)]
    print_gates(gates)
    jdump(run / "cpu.json", {"rows": rows, "gates": gates, "n_segments": len(agree), "dv": dv})
    np.savez(run / "cpu_traces.npz", flights=sel, **{f"S{i}": res[k][0] for k, i in enumerate(sel)})
    log(f"-> {run / 'cpu.json'}")


def cmd_export(a):
    """Package a run's policy as a flight control: <out>.tg = the --control drive + command_loop (the policy npz +
    the sensor-flight servo gain, climb shaping and measured envelope) + command_observer (the P observer: an LQG
    on pos,att,w, designed here from --sensor-lqr exactly as `sensors` designs it)."""
    from hover import reduced as Rd
    from hover import sensors as Sn
    from hover.policy import load_flight_policy, write_flight_control
    run, out = Path(a.run), Path(a.out)
    z = dict(np.load(run / "policy.npz"))
    pol = C.load_policy(run / "policy.npz")
    m = json.loads((run / "metrics.json").read_text())
    env = m["env_post"][str(int(pol["chosen_map"]))]
    T = float(pol["T"])
    wts = {k: v for k, v in json.loads(str(pol["weights"])).items() if k != "q_int"}
    Ka_P, rho_P = C.servo_gains(pol["A"], pol["B"], a.p_q_int, weights=wts)
    P = Rd.design(Rd.load_model(a.sensor_lqr), a.p_obs, "lqg", beat_s=T, bias=a.p_bias or None, bias_walk=a.p_bias_walk,
                  noise=Sn.SensorNoise(scale=a.p_design_noise, pos_latency_ms=a.pos_latency_ms), process_scale=a.p_process_scale)
    z.update(sensor_Ka=Ka_P, sensor_q_int=a.p_q_int, sensor_climb_after_stop=a.p_climb_after_stop,
             sensor_pos_latency_ms=a.pos_latency_ms, envelope=np.array([env[k] for k in C.AXES]))
    stem = out.stem
    rel = lambda q: os.path.relpath(Path(q).resolve(), Path(__file__).resolve().parents[1])
    np.savez(out.parent / f"{stem}_policy.npz", **z)
    Rd.save_controller(out.parent / f"{stem}_observer.npz", P)
    fp = load_flight_policy(a.control)
    K = fp.K
    write_flight_control(out, a0=fp.control.a0, depth_a=fp.theta[1:1 + 4 * K].reshape(4, K),
                         depth_b=fp.theta[1 + 4 * K:].reshape(4, K), attitude=fp.attitude, phase_lock=fp.phase_lock,
                         command_loop=f"{stem}_policy.npz", command_observer=f"{stem}_observer.npz",
                         header=f"Command-following flight control (HOVER / FWD / REV / UP / DOWN; hover.command.LiveCommand).\n"
                                f"Drive + trim = {rel(a.control)}; policy = {rel(run)}/policy.npz (opt/hover_cmd.py learn), exported by\n"
                                f"opt/hover_cmd.py export. Envelope (m/s): " + ", ".join(f"{k} {env[k]:g}" for k in C.AXES) + ".\n"
                                f"Fly it: src/view_free_hover.py --control {rel(out)} [--sensors]; or the GUI's Viewer tab.")
    log(f"sensor servo: q_int {a.p_q_int:g}, rho {rho_P:.3f}; observer rho {P['info']['rho_closed_loop']:.3f}")
    log(f"-> {out}, {out.parent / (stem + '_policy.npz')}, {out.parent / (stem + '_observer.npz')}")


def cmd_all(a):
    cmd_learn(a)
    a.run = a.out
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    cpu = subprocess.Popen([sys.executable, __file__, "cpu", "--run", str(a.out)], env=env,
                           stdout=open(Path(a.out) / "cpu.log", "w"), stderr=subprocess.STDOUT)
    cmd_dt(argparse.Namespace(**{**vars(a), "dt_ms": [0.10, 0.16], "flights": 64, "beats": 240}))
    cmd_sensors(argparse.Namespace(**{**vars(a), "beats": 120}))
    cpu.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tg", type=Path, default=FREE_HOVER_TG)
    ap.add_argument("--control", type=Path, default=FREE_HOVER_TG.parent / "control.tg")
    dev = argparse.ArgumentParser(add_help=False)
    g = dev.add_argument_group("device (MJX) knobs (as opt/hover_gpu.py)")
    g.add_argument("--x64", action="store_true")
    g.add_argument("--iterations", type=int, default=8, help="solver iterations of the flights (default 8: as the model's "
                   "100 within float32 noise, and a tumbling flight no longer stalls the vmapped batch; never 1)")
    g.add_argument("--lin", default="x64", choices=("x64", "f32"), help="precision of the device linearization")
    g.add_argument("--ls-iterations", type=int, default=8)
    g.add_argument("--contact", dest="no_contact", action="store_false")
    pol = argparse.ArgumentParser(add_help=False)
    g = pol.add_argument_group("policy")
    g.add_argument("--v-op-h", type=float, default=C.V_OP_H, help="FWD / REV operating speed [m/s]")
    g.add_argument("--v-op-z", type=float, default=C.V_OP_Z, help="UP / DOWN operating speed [m/s]")
    g.add_argument("--v-op-down", type=float, default=None, help="DOWN operating speed (default --v-op-z)")
    g.add_argument("--descend-after-stop", type=float, default=0.05, help="references: a descent starts only once the "
                   "horizontal reference speed is below this [m/s] (0 = off)")
    g.add_argument("--descend-delay", type=int, default=8, help="references: wingbeats of extra hold after the horizontal "
                   "stop before a descent starts")
    g.add_argument("--zeta-handoff", action=argparse.BooleanOptionalAction, default=True, help="scale the position "
                   "integral by hover.command.integral_keep as the reference decelerates (x_ref gets a 13th column)")
    g.add_argument("--zeta-gate", type=int, default=8, help="integral gate (hover.command.integral_gate): no "
                   "integration while the command is HOVER and the reference stopped less than this many wingbeats ago "
                   "(x_ref gets a 14th column; -1 = no gate)")
    g.add_argument("--stop-rate", type=float, default=1.0, help="reference rate x this while the command is HOVER")
    g.add_argument("--ff-lead", type=int, default=1, help="att / w / u_ff references lead v_ref by this many wingbeats "
                   "in flight (the attitude lags a velocity change; 2 already overshoots FWD)")
    g.add_argument("--learn-ff-lead", type=int, default=0, help="the lead of the ILC passes (learning with lead 1 moved "
                   "the FWD 0.20 / 0.25 entries so the extrapolated FWD 0.30 one saturated 67%% of FWD settle beats)")
    g.add_argument("--u-max", type=float, default=C.U_MAX)
    g.add_argument("--zeta-max", type=float, default=C.ZETA_MAX)
    g.add_argument("--q-int", type=float, default=100.0, help="LQR weight of the position integral")
    g.add_argument("--fly-q-int", type=float, default=50.0, help="q_int of the FLIGHT gain (0 = --q-int). The ILC passes "
                   "keep --q-int, whose integral settles inside the ILC window")
    g.add_argument("--fly-q-vel", type=float, default=0.15, help="q_vel of the flight gain (0 = --q-vel)")
    g.add_argument("--q-pos", type=float, default=0.02, help="LQR tolerances (hover.lqr.cost): position [m]")
    g.add_argument("--q-att", type=float, default=0.05, help="attitude [rad]")
    g.add_argument("--q-vel", type=float, default=0.1, help="velocity [m/s]")
    g.add_argument("--q-w", type=float, default=0.5, help="rate [rad/s]")
    g.add_argument("--r", type=float, default=0.05, help="input [harmonic depth]")
    g.add_argument("--pass-tilt", type=float, default=35.0, help="loss tilt [deg] in the ILC / envelope passes (eval: 60)")
    g.add_argument("--good-sat", type=float, default=0.7, help="an ILC cell is measured when alive, saturated <= this "
                   "fraction of the window (0.6 flipped DOWN 0.30 in and out between runs at 9-10 of 15 beats; 0.8 let in FWD "
                   "0.30 at 12 of 15 with cross 0.095 and spoiled FWD; 1 = no saturation test: the 93-100%% saturated FWD 0.30 "
                   "flight writes a 9 deg yaw reference), "
                   "|along| <= max(0.05, speed / 2) and cross <= --good-cross; the others are extrapolated")
    g.add_argument("--good-cross", type=float, default=0.08, help="an ILC cell is measured only when its cross-track "
                   "mean-motion speed is <= this [m/s] (DOWN 0.30 flies 0.045-0.047; inf = no cross test)")
    g.add_argument("--extrap", type=float, default=1.5, help="continuation: extrapolate at most this x the largest good speed")
    g.add_argument("--rate-beats", type=float, default=8.0, help="v_ref reaches V_op in this many wingbeats (spec: 4)")
    g.add_argument("--tilt-maps", type=float, nargs="+", default=list(C.TILT_MAPS))
    g.add_argument("--window", type=int, nargs=2, default=[15, 30], help="ILC / envelope window (beats)")
    g.add_argument("--dither", type=float, default=0.01)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--n-speeds", type=int, default=len(C.SPEEDS_H), help="speed grid points per axis (15)")
    g.add_argument("--call-beats", type=int, default=30, help="wingbeats per device call (the flight-program shape)")
    g.add_argument("--alloc-iters", type=int, default=0, help="input allocation: FISTA iterations of the projection onto "
                   "the input box in the servo's Q-function metric (hover.command.servo_hessian); 0 = clip (default: the "
                   "allocation flew worse, NOTES.md 14)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("learn", "all"):
        p = sub.add_parser(name, parents=[dev, pol])
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--batch", type=int, default=512, help="B: flights per device call (the one program shape)")
        p.add_argument("--eval-beats", type=int, default=120)
        p.add_argument("--continue-to", type=int, default=240, help="untimed: extend the eval flights (0 = no)")
        p.add_argument("--ilc-iters", type=int, default=2, help="ILC / continuation passes before the eval")
        p.add_argument("--ilc-fly-passes", type=int, default=0, help="the last N ILC passes fly the flight gain (--fly-q-int / "
                       "--fly-q-vel) instead of the learning gain: FWD 0.30 saturates under the learning gain, not the flight gain")
        p.add_argument("--no-verify", action="store_true", help="skip the post-ILC envelope verify pass")
        p.add_argument("--ref-lqr", type=Path, default=Path("opt/runs_gpu/ref/lqr.npz"), help="CPU x64 hover.lqr npz to compare with")
        if name == "all":
            p.add_argument("--p-ctl", type=Path, default=Path("opt/runs_imu/noise/ctl/P_lqgV1.npz"))
            p.add_argument("--sensor-lqr", type=Path, default=Path("opt/runs_gpu/ref/lqr_sensor.npz"))
            p.add_argument("--n-obs", default="att,w,acc")
            p.add_argument("--n-bias", default=None)
            p.add_argument("--n-design-noise", type=float, default=1.0)
            p.add_argument("--noise-scales", type=float, nargs="+", default=[1.0, 3.0])
            p.add_argument("--seeds", type=int, default=128)
            p.add_argument("--pos-latency-ms", type=float, default=5.0)
            p.add_argument("--sensor-ilc", type=int, default=2)
            p.add_argument("--n-variant", default="att_w", choices=("att_w", "acc_bias"))
            p.add_argument("--n-ctl", type=Path, default=Path("opt/runs_imu/noise/ctl/N_lqgV1.npz"))
            p.add_argument("--ilc-beats", type=int, default=30)
            p.add_argument("--ilc-noise", type=float, default=1.0)
            p.add_argument("--p-obs", default="pos,att,w")
            p.add_argument("--p-bias", default="att_xy")
            p.add_argument("--p-design-noise", type=float, default=1.0)
            p.add_argument("--p-bias-walk", type=float, default=9e-4)
            p.add_argument("--p-own-gain", action="store_true")
            p.add_argument("--p-q-int", type=float, default=3.0)
            p.add_argument("--n-dv-clip", type=float, default=1.0)
            p.add_argument("--n-ilc-any", action=argparse.BooleanOptionalAction, default=True)
            p.add_argument("--n-vel-ilc", type=float, default=1.0)
            p.add_argument("--n-no-vel", action="store_true")
            p.add_argument("--sensor-good-sat", type=float, default=0.6)
            p.add_argument("--p-vel-ilc", type=float, default=0.0)
            p.add_argument("--p-cz-seq", action=argparse.BooleanOptionalAction, default=True)
            p.add_argument("--p-process-scale", type=float, default=3.0)
            p.add_argument("--p-climb-after-stop", type=float, default=0.05)
            p.add_argument("--p-tables", default="true", choices=("true", "sensor"))
    p = sub.add_parser("dt", parents=[dev])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--call-beats", type=int, default=30)
    p.add_argument("--dt-ms", type=float, nargs="+", default=[0.10, 0.16])
    p.add_argument("--flights", type=int, default=64)
    p.add_argument("--beats", type=int, default=240)
    p = sub.add_parser("sensors", parents=[dev])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--p-ctl", type=Path, default=Path("opt/runs_imu/noise/ctl/P_lqgV1.npz"))
    p.add_argument("--sensor-lqr", type=Path, default=Path("opt/runs_gpu/ref/lqr_sensor.npz"), help="x64 hover.lqr npz with C_now")
    p.add_argument("--n-obs", default="att,w,acc")
    p.add_argument("--n-bias", default=None, help="bias channels of the IMU-only LQG (default: try acc, acc+att_xy, none)")
    p.add_argument("--n-design-noise", type=float, default=1.0)
    p.add_argument("--noise-scales", type=float, nargs="+", default=[1.0, 3.0])
    p.add_argument("--seeds", type=int, default=128)
    p.add_argument("--beats", type=int, default=120)
    p.add_argument("--pos-latency-ms", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sensor-ilc", type=int, default=2, help="ILC passes of sensor flights (tables in estimate coordinates)")
    p.add_argument("--n-variant", default="att_w", choices=("att_w", "acc_bias"),
                   help="IMU-only variant: att_w = the Prompt 1 att,w LQG + its own gain, no velocity servo (F6); "
                        "acc_bias = att,w,acc LQG with acc-bias states + velocity servo (unstable on CPU even at hover)")
    p.add_argument("--n-ctl", type=Path, default=Path("opt/runs_imu/noise/ctl/N_lqgV1.npz"))
    p.add_argument("--ilc-beats", type=int, default=30)
    p.add_argument("--ilc-noise", type=float, default=1.0, help="noise scale of the sensor ILC flights")
    p.add_argument("--p-obs", default="pos,att,w", help="design the P observer on these channels (must include pos) from "
                   "--sensor-lqr ('' = load --p-ctl, the Prompt 1 pos,att,w LQG)")
    p.add_argument("--p-bias", default="att_xy", help="bias channels of the designed P observer: att_xy estimates the "
                   "attitude filter's tilt error, which changes with the flight condition")
    p.add_argument("--p-design-noise", type=float, default=1.0)
    p.add_argument("--p-own-gain", action="store_true", help="P flies its observer's own LQR gain (no integral)")
    p.add_argument("--p-q-int", type=float, default=3.0, help="P flies the servo gain re-solved with this position-integral "
                   "weight (the learn weights otherwise; < 0: the policy's flight gain). The full integral loses P flights "
                   "(the attitude estimate is biased at speed)")
    p.add_argument("--p-bias-walk", type=float, default=9e-4, help="random walk per wingbeat of the P bias states "
                   "(hover.reduced.BIAS_WALK = 3e-4; 3x it with --p-process-scale 3: NOTES 20)")
    p.add_argument("--n-dv-clip", type=float, default=1.0, help="sensor ILC of N: clip of the estimate-minus-true "
                   "velocity offset along the command, as a fraction of the speed")
    p.add_argument("--n-ilc-any", action=argparse.BooleanOptionalAction, default=True, help="sensor ILC of N: update every alive, unsaturated cell "
                   "(not only those within 50%% of the commanded speed)")
    p.add_argument("--n-vel-ilc", type=float, default=1.0, help="sensor ILC of N: also move each cell's trim by this gain x "
                   "the linear trim sensitivity x its TRUE mean-motion velocity error (hover.command.velocity_ilc; 0 = off)")
    p.add_argument("--n-no-vel", action="store_true", help="N: zero the velocity columns of its gain (the IMU-only velocity "
                   "estimate is model prediction only)")
    p.add_argument("--sensor-good-sat", type=float, default=0.6, help="sensor ILC: a cell is measured when saturated <= "
                   "this fraction of its window")
    p.add_argument("--p-vel-ilc", type=float, default=0.0, help="sensor ILC of P: also move each cell's trim by this gain "
                   "x the trim sensitivity x its TRUE mean-motion velocity error (as --n-vel-ilc; 0 = off)")
    p.add_argument("--p-cz-seq", action=argparse.BooleanOptionalAction, default=True, help="P's observer predicts "
                   "with the per-beat offset that makes the reference its model's own trajectory (hover.command."
                   "trim_consistent_cz) instead of the constant hover drift (NOTES 19-20)")
    p.add_argument("--p-process-scale", type=float, default=3.0, help="P observer process noise x hover.reduced."
                   "PROCESS_STD (a larger one trusts mocap position more: faster velocity estimate in transitions)")
    p.add_argument("--p-climb-after-stop", type=float, default=0.05, help="P references: a climb starts only once the "
                   "horizontal v_ref <= this (FWD -> UP was 6 of 6 P x1 losses; 0 = off)")
    p.add_argument("--p-tables", default="true", choices=("true", "sensor"), help="P flies the true-state tables "
                   "(its estimate is trim-consistent) or its own sensor-ILC tables (estimate coordinates)")
    p.add_argument("--batch", type=int, default=512, help="sensor ILC pass width")
    p.add_argument("--window", type=int, nargs=2, default=[15, 30])
    p.add_argument("--good-sat", type=float, default=0.6)
    p.add_argument("--extrap", type=float, default=1.5)
    p = sub.add_parser("cpu")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--beats", type=int, default=240)
    p.add_argument("--n-plain", type=int, default=8)
    p.add_argument("--n-push", type=int, default=4)
    p.add_argument("--n-base", type=int, default=4)
    p.add_argument("--procs", type=int, default=16)
    p.add_argument("--cpu-contacts", action="store_true", help="keep MuJoCo contacts on CPU (the floor 0.25 m below the "
                   "release; the device flies with contacts off, so off by default for the comparison)")
    p = sub.add_parser("export", help="package a run's policy as a flight control .tg (live commands: viewer / GUI)")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="the .tg to write (its npz files go beside it)")
    p.add_argument("--sensor-lqr", type=Path, default=Path("opt/runs_gpu/ref/lqr_sensor.npz"))
    p.add_argument("--p-obs", default="pos,att,w")
    p.add_argument("--p-bias", default="att_xy")
    p.add_argument("--p-bias-walk", type=float, default=9e-4)
    p.add_argument("--p-design-noise", type=float, default=1.0)
    p.add_argument("--p-process-scale", type=float, default=3.0)
    p.add_argument("--p-q-int", type=float, default=3.0)
    p.add_argument("--p-climb-after-stop", type=float, default=0.05)
    p.add_argument("--pos-latency-ms", type=float, default=5.0)
    a = ap.parse_args()
    {"learn": cmd_learn, "dt": cmd_dt, "sensors": cmd_sensors, "cpu": cmd_cpu, "all": cmd_all, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
