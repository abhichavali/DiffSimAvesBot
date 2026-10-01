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
    $P trims --policy P.npz --out D               # measured edge trims (the learn --ff-trims input; ~24 min, x64)
    JAX_PLATFORMS=cpu $P torque --run opt/runs_cmd/R --limit 0.325   # crank actuator cost vs hover (CPU)
    uv run python opt/hover_cmd.py export --run opt/runs_cmd/R --out designs/hummer/free-hover/command.tg
                                                   # -> a flight control .tg (command_loop / command_observer) that
                                                   # src/view_free_hover.py and the GUI fly with live commands

Carried-lane continuation (X3', docs/history/command_flight_faster_notes_2026-10.md): `learn --ilc-iters 1 --ilc-carry 1` replaces the second ILC pass
by one call that keeps flying the first pass's lanes (good lanes step up one grid speed, lost ones re-seed at the
measured frontier), so the edge cells are measured 20-60 wingbeats into a flight; on top of --lin-rigid-rotation
--tilt-maps 5 15 --climb-gain reg, with --carry-ok-update
--carry-dither 0 --climb-refit --call-beats 20 --window 10 20 --climb-window 6 20 --pass-ramp-beats 8 --batch 384 and
per-axis --v-op-h 0.6 --v-op-rev 0.5 it is the round-3 policy (x3_gb). All these flags default to the old behaviour.

Edge trims and crank speed (X8, docs/history/command_flight_faster_notes_2026-10.md): `--ff-trims trims.npz` splices measured moving-cradle trims
(`trims`, precomputed, untimed) into the 15-deg FWD / REV entries from --ff-trims-from
(--ff-trims-keep: kept through the ILC); `--g-down G` adds a DOWN crank-speed feedforward flown with variable-length
wingbeats (hover.mjx_command.beat_var; CommandPolicy / LiveCommand return theta[0] = ln(n0 / n)).

Presets: `learn --preset fast` = round 6's config K as the defaults (V_op FWD 0.75 / REV 0.65 / UP 0.5 / DOWN 0.45, the
carried-lane flags above + --climb-speeds 0.3 0.6 --climb-chatter-max 1000 --rate-beats-z 6 + --ff-trims
designs/hummer/free-hover/command_trims.npz --ff-trims-keep); explicit flags override it. No --preset (= --preset r18)
keeps every old default: `learn --out R` is still the r18 config.

Device knobs: float32 flights unless --x64, solver iterations 8 (a tumbling flight otherwise stalls the whole
vmapped batch; the change is below the float32 run-to-run noise), --ls-iterations 8, contacts off (the scene's
floor is 0.25 m below the release COM: descending flights pass through it); the linearization runs in x64.
Results and dead ends: docs/history/command_flight_notes_2026-09.md (r18), docs/history/command_flight_faster_notes_2026-10.md (--preset fast). Set JAX_COMPILATION_CACHE_DIR for the persistent XLA cache (every run-specific quantity is a traced
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
    """Fixed-magnitude command speeds (FWD/REV, UP, DOWN[, REV]: the 4th only with --v-op-rev)."""
    v = (a.v_op_h, a.v_op_z, a.v_op_down if a.v_op_down is not None else a.v_op_z)
    return v + (a.v_op_rev,) if getattr(a, "v_op_rev", None) is not None else v


def ref_kw(a, lead=None):
    """build_refs keywords of the integral handoff / gate, the stop rate and the feedforward lead (``lead`` overrides
    a.ff_lead: the ILC passes learn without it); ``gain_v`` only when set (a policy with a scheduled forward gain)."""
    kw = {"zeta_handoff": a.zeta_handoff, "zeta_gate": a.zeta_gate, "stop_rate": a.stop_rate,
          "ff_lead": a.ff_lead if lead is None else lead}
    if getattr(a, "gain_v", None) is not None:
        kw["gain_v"] = a.gain_v
    return kw


def pol_ref_args(pol):
    g = int(pol.get("zeta_gate", -1))
    return argparse.Namespace(zeta_handoff=bool(pol.get("zeta_handoff", False)), zeta_gate=None if g < 0 else g,
                              stop_rate=float(pol.get("stop_rate", 1.0)), ff_lead=int(pol.get("ff_lead", 0)),
                              gain_v=C.pol_gain_v(pol))


def gate_arg(a):
    """--zeta-gate -1 = no gate column."""
    if a.zeta_gate is not None and a.zeta_gate < 0:
        a.zeta_gate = None
    return a


def xref_width(a):
    return (14 if a.zeta_gate is not None else 13 if a.zeta_handoff else 12) + (getattr(a, "gain_v", None) is not None)


def rates(a, B):
    return np.broadcast_to(np.array([a.v_op_h / a.rate_beats, a.v_op_h / a.rate_beats,
                                     a.v_op_z / (getattr(a, "rate_beats_z", None) or a.rate_beats)]), (B, 3))


def pass_rate(a, vop_axis, speed):
    """Reference rate of an ILC / carry lane: V_op / rate_beats, faster for a cell above 2 V_op (it ramps in at most
    2 rate_beats), and with --pass-ramp-beats in at most that many wingbeats."""
    r = max(vop_axis, speed / 2) / a.rate_beats
    prb = getattr(a, "pass_ramp_beats", None)
    return max(r, speed / prb) if prb else r


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
        rate[i] = pass_rate(a, vop, np.linalg.norm(v))
    x_ref = np.zeros((B, a.call_beats + 1, xref_width(a))); u_ff = np.zeros((B, a.call_beats + 1, 24))
    v_ref = np.zeros((B, a.call_beats + 1, 3))
    nbeat = None if g_n0(a) is None else np.zeros((B, a.call_beats + 1), int)
    for m, tab in enumerate(tabs):
        sel = np.array([cells[c][0] == m for c in cell_of])
        x_ref[sel], u_ff[sel], v_ref[sel] = C.build_refs(v_cmd[sel], tab, e_f, T, rate[sel], **ref_kw(a, lead))
        if nbeat is not None:
            x_ref[sel], nbeat[sel] = g_retime(a, tab, e_f, T, x_ref[sel], v_ref[sel], ref_kw(a, lead)["ff_lead"])
    dither = np.zeros((B, a.call_beats + 1, 24))
    if B > nc:
        dither[nc:] = rng.normal(0, a.dither, (B - nc, a.call_beats + 1, 24))
    return cells, cell_of, v_cmd, x_ref, u_ff, dither, v_ref, nbeat


def g_n0(a):
    """The wingbeat length n0 (steps) of the crank-speed feedforward (--g-down), or None (off: the fixed wingbeat)."""
    return getattr(a, "g_n0", None)


def g_retime(a, tab, e_f, T, x_ref, v_ref, lead):
    """--g-down: per-beat wingbeat lengths of the references and x_ref with p_ref re-integrated over them."""
    nb = C.g_beats(tab, e_f, v_ref, a.g_n0, lead)
    return C.retime_refs(x_ref, v_ref, T, nb, a.g_n0), nb


def run_cell_pass(chain, fl, a, tabs, e_f, T, B, seed, clock=None, label="ILC pass", Ka=None, lead=None):
    from hover import mjx_command as MC
    cells, cell_of, v_cmd, x_ref, u_ff, dither, v_ref, nbeat = cell_pass_inputs(a, tabs, e_f, T, B, np.random.default_rng(seed), lead)
    Ka = a.Ka if Ka is None else Ka
    st, c_rest = MC.release_carry(fl, B, C.servo_nz(Ka))
    log(f"{label}: {B} flights x {a.call_beats} wingbeats ({len(cells)} clean cells + {B - len(cells)} dithered copies)")
    S, U, A, st_end, secs = chain.run(st, c_rest, np.zeros((B, a.call_beats, 12)), x_ref, u_ff, dither, np.repeat(Ka[None], B, 0), 1,
                                 log=log, lost_tilt=math.radians(a.pass_tilt), nbeat=nbeat)
    Tb = C.beat_times(T, nbeat, g_n0(a), a.call_beats)
    win = tuple(a.window)
    rows = []
    for ci, (m, ax, j) in enumerate(cells):
        vc = C.cell_velocity(tabs[m], e_f, ax, j)
        r = C.cell_pass(S, U, A, [ci], vc, win, Tb, a.u_max) if ci < B else {"passed": False, "alive": False, "sat": 1.0}
        r["good"] = lane_good(a, r, float(np.linalg.norm(vc)))
        rows.append(r)
    env = C.envelope(rows, cells, tabs[0]["speeds"], len(tabs))
    out = {"cells": cells, "cell_of": cell_of, "S": S, "U": U, "A": A, "u_ff": u_ff[:, :a.call_beats], "dither": dither[:, :a.call_beats],
           "x_ref": x_ref, "rows": rows, "env": env, "secs": secs, "Tb": Tb, "nbeat": nbeat}
    if getattr(a, "ilc_carry", 0):                     # what a carried continuation (run_carry_call) starts from
        out.update(state=st_end, v_ref=v_ref, c_rest=c_rest, lane_ok=lanes_ok(a, out, tabs, e_f, T))
    return out


def lane_good(a, r, sp):
    """An ILC measurement is usable (hover.command.cell_pass row ``r`` of a lane at speed ``sp``)."""
    err = abs(r["along"]) if "along" in r else r.get("v", np.inf)
    return bool(r["alive"] and r["sat"] <= a.good_sat and err <= max(0.05, 0.5 * sp) and r.get("cross", 0.0) <= a.good_cross)


def lanes_ok(a, p, tabs, e_f, T):
    """Per lane (flight) of a pass / carry call: its own cell_pass row is a usable measurement (lane_good)."""
    ok = np.zeros(len(p["cell_of"]), bool)
    for i, ci in enumerate(p["cell_of"]):
        m, ax, j = p["cells"][ci]
        vc = C.cell_velocity(tabs[m], e_f, ax, j)
        ok[i] = lane_good(a, C.cell_pass(p["S"], p["U"], p["A"], [i], vc, tuple(a.window), p.get("Tb", T), a.u_max),
                          float(np.linalg.norm(vc)))
    return ok


def carry_frontier(good_cum, cells, n_maps, n_speeds):
    """{(map, axis): the largest speed index j whose cells 1..j are all measured (good in some pass / carry call)}."""
    fr = {}
    for m in range(n_maps):
        for ax in range(4):
            j = 0
            while j + 1 < n_speeds and good_cum[cells.index((m, ax, j + 1))]:
                j += 1
            fr[(m, ax)] = j
    return fr


def run_carry_call(chain, a, prev, tabs, e_f, T, B, good_cum, seed, Ka, label):
    """--ilc-carry: one more call of the ILC flights that CONTINUES the previous call's lanes (its end carry, alive,
    position integral and references) instead of re-releasing them. A lane whose last measurement was usable
    (lanes_ok) steps its constant command up by --carry-step grid speeds, ramping from its current reference; a
    lost or unusable lane is re-seeded from the release onto the cells just above its axis' measured frontier
    (round robin); no lane goes above frontier + --carry-ahead. The hover lanes stay at hover. Tables are the
    previous update's (bootstrapped above the frontier by extrapolation). The carried lanes are measured 30-60+
    wingbeats into their flight, where the edge cells departed (X3' STEP 0), at the cost of one pass."""
    import jax
    import jax.numpy as jnp
    cells, n_sp = prev["cells"], tabs[0]["speeds"].shape[1]
    nc = len(cells)
    fr = carry_frontier(good_cum, cells, len(tabs), n_sp)
    ok = prev["lane_ok"]
    cell_of = np.empty(B, int); reseed = np.zeros(B, bool); rr = {}
    for i in range(B):
        m, ax, j = cells[prev["cell_of"][i]]
        cap = min(fr[(m, ax)] + a.carry_ahead, n_sp - 1)
        if (ax, j) == (0, 0):
            jn, reseed[i] = 0, not ok[i]
        elif ok[i]:
            jn = min(j + a.carry_step, cap)
        else:
            reseed[i] = True
            tg = list(range(min(fr[(m, ax)] + 1, cap), cap + 1))
            k = rr.get((m, ax), 0); rr[(m, ax)] = k + 1
            jn = tg[k % len(tg)]
        cell_of[i] = cells.index((m, ax, jn))
    N = a.call_beats
    v_cmd = np.zeros((B, N, 3)); rate = np.zeros((B, 3))
    for i in range(B):
        m, ax, j = cells[cell_of[i]]
        v = C.cell_velocity(tabs[m], e_f, ax, j)
        v_cmd[i] = v
        vop_ = a.v_op_h if ax < 2 else a.v_op_z
        rate[i] = pass_rate(a, vop_, np.linalg.norm(v))
    keep = ~reseed[:, None]
    v0 = prev["v_ref"][:, -1] * keep
    p0 = prev["x_ref"][:, -1, :3] * keep
    x_ref = np.zeros((B, N + 1, xref_width(a))); u_ff = np.zeros((B, N + 1, 24)); v_ref = np.zeros((B, N + 1, 3))
    nbeat = None if g_n0(a) is None else np.zeros((B, N + 1), int)
    for m, tab in enumerate(tabs):
        sel = np.array([cells[c][0] == m for c in cell_of])
        x_ref[sel], u_ff[sel], v_ref[sel] = C.build_refs(v_cmd[sel], tab, e_f, T, rate[sel], **ref_kw(a, a.learn_ff_lead),
                                                         v0=v0[sel], p0=p0[sel])
        if nbeat is not None:
            x_ref[sel], nbeat[sel] = g_retime(a, tab, e_f, T, x_ref[sel], v_ref[sel], ref_kw(a, a.learn_ff_lead)["ff_lead"])
    dither = np.zeros((B, N + 1, 24))
    cd = a.dither if a.carry_dither is None else a.carry_dither
    if B > nc and cd:
        dither[nc:] = np.random.default_rng(seed).normal(0, cd, (B - nc, N + 1, 24))
    c, alive, zeta = prev.pop("state")                  # (device memory: one carried state at a time)
    c_rest = prev["c_rest"]
    msk = jnp.asarray(reseed)
    c = jax.tree_util.tree_map(lambda x, y: jnp.where(msk.reshape((-1,) + (1,) * (jnp.ndim(x) - 1)), jnp.asarray(y, x.dtype), x),
                               c, c_rest)
    alive = jnp.asarray(alive) | msk
    zeta = jnp.where(msk[:, None], jnp.zeros_like(zeta), zeta)
    n_carried = int((~reseed).sum())
    log(f"{label}: {B} lanes x {N} wingbeats: {n_carried} carried (step +{a.carry_step}), {int(reseed.sum())} re-seeded; frontier "
        + "; ".join(f"{tabs[m]['tilt_max_deg']:g}/{C.AXES[ax]} {tabs[m]['speeds'][ax, j]:g}" for (m, ax), j in fr.items()))
    S, U, A, st_end, secs = chain.run((c, alive, zeta), c_rest, np.zeros((B, N, 12)), x_ref, u_ff, dither,
                                      np.repeat(Ka[None], B, 0), 1, log=log, lost_tilt=math.radians(a.pass_tilt), nbeat=nbeat)
    Tb = C.beat_times(T, nbeat, g_n0(a), N)
    out = {"cells": cells, "cell_of": cell_of, "S": S, "U": U, "A": A, "u_ff": u_ff[:, :N], "dither": dither[:, :N], "x_ref": x_ref,
           "secs": secs, "state": st_end, "v_ref": v_ref, "c_rest": c_rest, "reseed": reseed, "frontier": fr, "Tb": Tb, "nbeat": nbeat}
    out["lane_ok"] = lanes_ok(a, out, tabs, e_f, T)
    rows = []
    for ci, (m, ax, j) in enumerate(cells):
        ln = np.flatnonzero(cell_of == ci)
        if not len(ln):
            rows.append(None)
            continue
        vc = C.cell_velocity(tabs[m], e_f, ax, j)
        rs = [C.cell_pass(S, U, A, [i], vc, tuple(a.window), Tb, a.u_max) for i in ln]
        rows.append({"n": len(ln), "n_carried": int((~reseed[ln]).sum()), "passed": bool(np.mean([r["passed"] for r in rs]) >= 0.5),
                     "good": bool(out["lane_ok"][ln].any() if a.carry_ok_update else out["lane_ok"][ln].mean() >= 0.5),
                     "n_ok": int(out["lane_ok"][ln].sum()), "alive": int(sum(r["alive"] for r in rs)),
                     "sat": float(np.mean([r["sat"] for r in rs if r["alive"]] or [1.0]))})
    out["rows"] = rows
    return out


def du_chatter(a, p, tabs, window):
    """Median per-wingbeat |u_k - u_{k-1}|_inf of a pass / carry call's clean (undithered) lanes alive through the
    window: UP (at V_op z .. 2 V_op z: a climb gain can chatter only above the operating speed, x3_carry1: UP
    0.36-0.60), MOVE (FWD / REV / DOWN at <= their operating speed) and HOVER. The chatter check compares UP with
    MOVE: the hover lanes' |du| (~0.0007 in the eval) is so small that 3x it rejects the clean certified gains too
    (x2c_climb_b / x2c_z04c: UP / HOVER 3.5 in eval; the chattering x2c_z04b: 34)."""
    lo, hi = window
    cells = p["cells"]
    nc = len(cells)
    lim = {0: a.v_op_h, 1: getattr(a, "v_op_rev", None) or a.v_op_h, 2: a.v_op_z, 3: a.v_op_down if a.v_op_down is not None else a.v_op_z}
    res = {}
    for name, axes in (("UP", (2,)), ("MOVE", (0, 1, 3)), ("HOVER", ())):
        ln = []
        for i in range(min(nc, len(p["cell_of"]))):
            m, ax, j = cells[p["cell_of"][i]]
            hov = (ax, j) == (0, 0)
            sp = tabs[m]["speeds"][ax, j]
            ok_sp = lim[ax] - 1e-9 <= sp <= 2 * lim[ax] + 1e-9 if name == "UP" else sp <= lim[ax] + 1e-9
            if (hov if not axes else (not hov and ax in axes and ok_sp)) \
                    and bool(p["A"][i][:hi + 1].all()):
                ln.append(i)
        du = [np.abs(np.diff(p["U"][i, lo:hi], axis=0)).max(-1) for i in ln]
        res[name] = float(np.median(np.concatenate(du))) if du else float("nan")
    return res


def eval_du(U, A, seg, arm, push, skip=10):
    """Median per-wingbeat |u_k - u_{k-1}|_inf of the eval's V_op servo flights (no push) per command segment
    (beats ``skip``.. of the segment, alive through it): opt/runs_fast/scratch/du_chatter.py on the learn's arrays."""
    dus = {n: [] for n in C.CMDS}
    for i in np.flatnonzero((arm == "vop") & (push < 0)):
        for k0, k1, c, _ in seg[i]:
            if k0 < 0:
                continue
            k0, k1 = int(k0) + skip, min(int(k1), U.shape[1])
            if k1 - k0 < 3 or not A[i, :k1 + 1].all():
                continue
            dus[C.CMDS[int(c)]].append(np.abs(np.diff(U[i, k0:k1], axis=0)).max(-1))
    med = {n: float(np.median(np.concatenate(v))) if v else float("nan") for n, v in dus.items()}
    move = float(np.nanmedian([med["FWD"], med["REV"], med["DOWN"]]))
    return {**med, "MOVE": move, "ratio": med["UP"] / max(move, 1e-12)}


def climb_gain(a, passes, tabs, A_, B_, K_hover):
    """--climb-gain reg: the FLIGHT gain re-solved (DARE, flight weights) on a per-wingbeat model regressed from the
    ILC passes' dithered climb flights (UP cells of every map at --climb-speeds, beats --climb-window, only flights
    alive through its end; hover.command.regress_ab; --climb-reg fixB: A only, B = the hover model's, A shrunk
    toward the hover model's by --climb-ridge). The hover model's small-perturbation gain lets a sustained
    climb grow a ~8-wingbeat roll oscillation under dither / pushes that the moving-cradle linearization at climb
    does not show (docs/history/command_flight_faster_notes_2026-10.md X2'). Sets a.Ka_fly; returns the diagnostics (None: kept the hover gain)."""
    lo_v, hi_v = a.climb_speeds
    fl_all, S_l, U_l, A_l = [], [], [], []
    off = 0
    for p in passes:
        nc = len(p["cells"])
        sel = [i for i in range(len(p["cell_of"])) if i >= nc and p["cells"][p["cell_of"][i]][1] == 2
               and lo_v - 1e-9 <= tabs[0]["speeds"][2][p["cells"][p["cell_of"][i]][2]] <= hi_v + 1e-9]
        fl_all += [off + i for i in sel]
        S_l.append(p["S"]); U_l.append(p["U"]); A_l.append(p["A"])
        off += len(p["cell_of"])
    S_c, U_c, A_c = np.concatenate(S_l), np.concatenate(U_l), np.concatenate(A_l)
    fixb = a.climb_reg == "fixB"
    fly_w = {**weights(a), **({"q_vel": a.fly_q_vel} if a.fly_q_vel else {})}
    # fixB: raise the ridge (x3 per step) until the climb gain also holds the hover model within --climb-rho-max
    ridges = [a.climb_ridge * 3 ** j for j in range(4)] if fixb else [0.0]
    info = None
    for rg in ridges:
        r = C.regress_ab(S_c, U_c, A_c, fl_all, tuple(a.climb_window), B_fixed=B_ if fixb else None, A_prior=A_, ridge=rg)
        if r is None:
            log(f"climb gain: too few climb flights alive through beat {a.climb_window[1]}: keeping the hover flight gain")
            return None
        A_r, B_r, X = r
        rho_old = float(np.abs(np.linalg.eigvals(C.servo_closed(A_r, B_r, a.Ka_fly))).max())
        try:
            Ka_c, rho_c = C.servo_gains(A_r, B_r, a.fly_q_int or a.q_int, weights=fly_w, **aug_kw(a))
        except (AssertionError, np.linalg.LinAlgError, ValueError) as e:
            log(f"climb gain: DARE on the regressed climb model (ridge {rg:g}) failed ({e!r})")
            continue
        rho_h = float(np.abs(np.linalg.eigvals(C.servo_closed(A_, B_, Ka_c))).max())
        info = {"n_flights": int(sum(bool(np.asarray(A_c[i][:a.climb_window[1] + 1]).all()) for i in fl_all)), "n_samples": len(X),
                "cond": float(np.linalg.cond(X)), "ridge": rg, "rho_hover_gain_on_climb": rho_old, "rho_climb_gain_on_climb": rho_c,
                "rho_climb_gain_on_hover": rho_h, "dKa_rel": float(np.linalg.norm(Ka_c - a.Ka_fly) / np.linalg.norm(a.Ka_fly)),
                "A_climb": A_r, "B_climb": B_r, "accepted": bool(rho_h <= a.climb_rho_max)}
        log(f"climb gain: regressed climb model (ridge {rg:g}) from {info['n_flights']} flights ({len(X)} samples, cond "
            f"{info['cond']:.0f}): hover flight gain rho {rho_old:.3f} on it; the climb gain rho {rho_c:.3f} (on the hover model "
            f"{rho_h:.3f}), |dKa|/|Ka| {info['dKa_rel']:.2f}" + (" -> flies every axis" if info["accepted"] else ""))
        if info["accepted"]:
            a.Ka_fly = Ka_c
            return info
    log(f"climb gain: no regressed gain holds the hover model within rho {a.climb_rho_max:g}: keeping the hover flight gain")
    return info


def fwd_gain(a, runs, tabs, chosen, A_, B_):
    """--fwd-gain reg: a speed-scheduled FORWARD gain (X4 round 4, FWD 0.70 departs by a slow yaw divergence at 0%
    saturation). As climb_gain: the flight gain re-solved (DARE, flight weights) on a model regressed from the
    learn's own FWD flights (pass + carry lanes of the chosen map at --fwd-speeds, beats --fwd-window, alive through
    its end), A only with B fixed from hover, A shrunk toward the hover A by --fwd-ridge (x3 steps until the gain
    holds the hover model within --climb-rho-max). The eval / policy then flies Ka + w (Ka_fwd - Ka), w =
    hover.command.gain_weight of the reference forward speed between --fwd-blend lo / hi (a 15th x_ref column, the
    same in build_refs, RefStream and LiveCommand). Returns (Ka_fwd or None, diagnostics)."""
    lo_v, hi_v = a.fwd_speeds
    fl_all, S_l, U_l, A_l = [], [], [], []
    off = 0
    for p in runs:
        for i in range(len(p["cell_of"])):
            m, ax, j = p["cells"][p["cell_of"][i]]
            if m == chosen and ax == 0 and lo_v - 1e-9 <= tabs[m]["speeds"][0][j] <= hi_v + 1e-9:
                fl_all.append(off + i)
        S_l.append(p["S"]); U_l.append(p["U"]); A_l.append(p["A"])
        off += len(p["cell_of"])
    S_c, U_c, A_c = np.concatenate(S_l), np.concatenate(U_l), np.concatenate(A_l)
    fly_w = {**weights(a), **({"q_vel": a.fly_q_vel} if a.fly_q_vel else {})}
    win = tuple(a.fwd_window or a.climb_window)
    info = None
    for rg in [a.fwd_ridge * 3 ** j for j in range(4)]:
        r = C.regress_ab(S_c, U_c, A_c, fl_all, win, B_fixed=B_, A_prior=A_, ridge=rg)
        if r is None:
            log(f"fwd gain: too few FWD flights alive through beat {win[1]}: no scheduled gain")
            return None, None
        A_r, B_r, X = r
        rho_old = float(np.abs(np.linalg.eigvals(C.servo_closed(A_r, B_r, a.Ka_fly))).max())
        try:
            Ka_f, rho_f = C.servo_gains(A_r, B_r, a.fly_q_int or a.q_int, weights=fly_w, **aug_kw(a))
        except (AssertionError, np.linalg.LinAlgError, ValueError) as e:
            log(f"fwd gain: DARE on the regressed FWD model (ridge {rg:g}) failed ({e!r})")
            continue
        rho_h = float(np.abs(np.linalg.eigvals(C.servo_closed(A_, B_, Ka_f))).max())
        n_fl = int(sum(bool(np.asarray(A_c[i][:win[1] + 1]).all()) for i in fl_all))
        info = {"n_flights": n_fl, "n_candidates": len(fl_all), "n_samples": len(X), "cond": float(np.linalg.cond(X)), "ridge": rg,
                "rho_base_gain_on_fwd": rho_old, "rho_fwd_gain_on_fwd": rho_f, "rho_fwd_gain_on_hover": rho_h,
                "dKa_rel": float(np.linalg.norm(Ka_f - a.Ka_fly) / np.linalg.norm(a.Ka_fly)),
                "yaw_rows_dKa_rel": float(np.linalg.norm((Ka_f - a.Ka_fly)[:, [5, 11]]) / max(np.linalg.norm(a.Ka_fly[:, [5, 11]]), 1e-12)),
                "blend": list(a.fwd_blend), "accepted": bool(rho_h <= a.climb_rho_max)}
        log(f"fwd gain: regressed FWD model (ridge {rg:g}) from {n_fl}/{len(fl_all)} flights ({len(X)} samples, cond {info['cond']:.0f}): "
            f"base flight gain rho {rho_old:.3f} on it; the FWD gain rho {rho_f:.3f} (on the hover model {rho_h:.3f}), |dKa|/|Ka| "
            f"{info['dKa_rel']:.2f}" + (f" -> scheduled by the reference forward speed {a.fwd_blend[0]:g}..{a.fwd_blend[1]:g} m/s"
                                        if info["accepted"] else ""))
        if info["accepted"]:
            return Ka_f, info
    log(f"fwd gain: no regressed gain holds the hover model within rho {a.climb_rho_max:g}: no scheduled gain")
    return None, info


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
    nbeat = None
    if g_n0(a) is not None:
        x_ref, nbeat = g_retime(a, tab, e_f, T, x_ref, v_ref, a.ff_lead)
        nbeat[arm == "baseline"] = a.g_n0                 # the baseline LQR arm flies the fixed crank speed
    x_ref[arm == "baseline", :, :12] = C.baseline_refs(v_ref[arm == "baseline"], T)
    u_ff[arm == "baseline"] = 0.0
    Kp = a.Ka_fly if getattr(a, "Ka_pol", None) is None else a.Ka_pol     # --fwd-gain: the 30-wide scheduled gain
    Ka = np.repeat(Kp[None], B, 0)
    Ka[arm == "baseline"] = C.pad_ka(C.hover_ka(K_hover), Kp.shape[1])
    kicks = np.zeros((B, NB, 12))
    push = -np.ones(B, int)
    pv = list(HG.VERIFY_PUSHES.values())
    for n, i in enumerate(range(n_vop - n_vop // 4, n_vop)):
        k = int(sched["seg"][i, 1, 0]) + 5
        if k < NB:
            kicks[i, k] = pv[n % len(pv)]
            push[i] = k
    return sched, arm, x_ref, u_ff, Ka, kicks, push, nbeat


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

def splice_trims(a, tabs):
    """--ff-trims: the measured moving-cradle trims (`trims`: hover.mjx_cradle.trim_at_speed
    + orbit_entries; axis 0 = FWD, 1 = REV) replace the FWD / REV entries (u, att, dv, w) of the largest-tilt map at
    grid speeds >= --ff-trims-from (FWD, REV), the edge where the ILC-learned tables were extrapolations. Spliced into
    the LP base tables before the first ILC pass, so the passes / carry fly them and the ILC refines them like any
    other entry (and extrapolates relative to them); --ff-trims-keep also re-splices them after every update (the
    trims themselves are flown). Returns (tables, info)."""
    z = a.ff_trims_data
    out = [dict(t) for t in tabs]
    m = max(range(len(tabs)), key=lambda k: tabs[k]["tilt_max_deg"])
    t = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tabs[m].items()}
    nu = t["u"].shape[-1]
    info = {"map": m}
    for ax in (0, 1):
        sp = t["speeds"][ax]
        done = []
        for i in np.flatnonzero(z["axis"] == ax):
            j = int(np.argmin(np.abs(sp - z["speed"][i])))
            if abs(sp[j] - z["speed"][i]) > 1e-6 or sp[j] < a.ff_trims_from[ax] - 1e-6:
                continue
            t["u"][ax, j] = np.clip(z["x"][i][:nu], -0.9 * a.u_max, 0.9 * a.u_max)
            t["att"][ax, j], t["dv"][ax, j], t["w"][ax, j] = z["att"][i], z["dv"][i], z["w_ent"][i]
            done.append(round(float(sp[j]), 3))
        info[C.AXES[ax]] = done
    out[m] = t
    return out, info


def cpu_bg(bird, rel, fp, n0, steps):
    """One-wingbeat central difference of the strobed state in the crank speed g (CPU, hover.launch.fly): wingbeats of
    n0 -/+ ``steps`` steps (g = ln(n0 / n), quantized as the flights fly it). Returns B_g (12,) per unit g."""
    from hover import launch as L
    x = {}
    for n in (n0 - steps, n0 + steps):
        dg = float(C.g_of_steps(n, n0))
        th = np.array(fp.theta, float).copy()
        th[0] += dg
        x[n] = (dg, L.fly(bird, rel, th, 1)[1])
    (dp, xp), (dm, xm) = x[n0 - steps], x[n0 + steps]
    return (xp - xm) / (dp - dm)


def device_linearize(a, bird, rel, fp, fl):
    """hover.lqr's one-wingbeat central differences as one batched device call. --lin x64 (default) switches JAX to
    float64 for this call only (a float64 Flight of the same released state, the model's own solver settings, as
    the CPU reference: matches hover.lqr to round-off; 67 flights x 1 wingbeat take ~1 s on an RTX 5070 Ti) and
    back to float32 for the flights. --lin-rigid-rotation: rotate_vel in the attitude perturbations. Returns (A, B, drift,
    compile / cache-load seconds)."""
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
        rv = getattr(a, "lin_rigid_rotation", False)
        flin = jax.jit(jax.vmap(lambda ar, p, du: (F.one_beat(S_, ar, p, du, rv),), in_axes=(None, 0, 0)))
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
    a.ff_trims_data = None
    if a.ff_trims:
        z = np.load(a.ff_trims)
        a.ff_trims_data = {k: z[k] for k in ("axis", "speed", "x", "att", "dv", "w_ent", "residual")}
        a.ff_trims_data["axis"] = a.ff_trims_data["axis"].astype(int)
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
    a.g_n0 = a.g_nmax = a.g_B = None
    if a.g_down:
        # --g-down: a DOWN crank-speed feedforward (variable-length wingbeats, hover.mjx_command.beat_var); its one-beat
        # effect B_g from a CPU central difference at the release (n0 -/+ --g-fd-steps steps), the DOWN LP entries
        # re-solved with it, then learned by the ILC as usual
        t = time.time()
        a.g_n0, a.g_B = S_.n_step, cpu_bg(bird, rel, fp, S_.n_step, a.g_fd_steps)
        tabs = [C.g_table(t_, A_, B_, drift, a.g_B, e_f, a.g_down, a.g_down_full, a.u_max) for t_ in tabs]
        a.g_nmax = int(C.beat_steps(min(t_["g"].min() for t_ in tabs), a.g_n0))
        log(f"crank-speed feedforward: DOWN g = {a.g_down * 100:+.1f}% x min(1, s / {a.g_down_full:g}) -> wingbeats {a.g_n0}..{a.g_nmax} "
            f"steps; B_g (CPU FD, {time.time() - t:.1f} s) vel {np.round(a.g_B[6:9], 3).tolist()} w {np.round(a.g_B[9:12], 2).tolist()}; "
            + "; ".join(f"tilt {t_['tilt_max_deg']:g} DOWN |u_ff|inf @ " + ", ".join(f"{t_['speeds'][3, j]:.2f} {np.abs(t_['u'][3, j]).max():.3f}"
                                                                                    for j in (5, 7, 8, 10)) for t_ in tabs))
    if a.ff_trims:
        tabs, sinfo = splice_trims(a, tabs)
        log(f"--ff-trims {a.ff_trims}: trims spliced into the base tables (map {sinfo['map']}): FWD {sinfo['FWD']}, REV {sinfo['REV']}"
            + (" (kept through the ILC)" if a.ff_trims_keep else " (the ILC refines them)"))
    Ka, rho = C.servo_gains(A_, B_, a.q_int, weights=weights(a))
    a.Ka = Ka
    fly_w = {**weights(a), **({"q_vel": a.fly_q_vel} if a.fly_q_vel else {})}
    aug = aug_kw(a)
    a.Ka_fly, rho_fly = (C.servo_gains(A_, B_, a.fly_q_int or a.q_int, weights=fly_w, **aug)
                         if (a.fly_q_int or a.fly_q_vel or aug) else (Ka, rho))
    if a.Ka_fly is not Ka:
        log(f"flight gain (q_int {a.fly_q_int or a.q_int:g}, q_vel {fly_w['q_vel']:g}): rho {rho_fly:.3f}; the ILC passes use "
            f"the learning gain (q_int {a.q_int:g}): its integral settles within the ILC window")
    if aug:
        # --du-penalty / --yaw-int: the flight (and climb) gains are augmented (hover.command.servo_gains_aug); the
        # learning gain of the ILC passes stays the plain one, zero-padded to the same program shape
        a.Ka = C.pad_ka(Ka, a.Ka_fly.shape[1])
        p2 = lambda M: max([abs(z) for z in np.linalg.eigvals(M) if z.real < -0.2], default=0.0)
        Ka_plain, _ = C.servo_gains(A_, B_, a.fly_q_int or a.q_int, weights=fly_w)
        log(f"augmented flight gain ({aug}): Ka {a.Ka_fly.shape}, period-2 mode on the hover model |z| "
            f"{p2(C.servo_closed(A_, B_, a.Ka_fly)):.3f} (plain gain {p2(C.servo_closed(A_, B_, Ka_plain)):.3f})")
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
    chain = MC.Chain(fl, a.batch, a.call_beats, a.u_max, a.zeta_max, H=a.H, alloc_iters=a.alloc_iters, n_max=a.g_nmax)
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
                                  e_f, p["Tb"], window=tuple(a.window), u_max=a.u_max, good=good)
        tabs_used, xinfo = C.extrapolate_tables(meas, tabs, good, p["cells"], max_factor=a.extrap, u_max=a.u_max)
        if a.ff_trims and a.ff_trims_keep:
            tabs_used, _ = splice_trims(a, tabs_used)
        p.update(ilc_info=info, extrap=xinfo, good=good)
        passes.append(p)
        log(f"ILC {it + 1}: envelope " + "; ".join(f"tilt {tabs[m]['tilt_max_deg']:g}: {p['env'][m]}" for m in p["env"])
            + f" | good cells {int(good.sum())}/{len(good)}, max |du| {max(r.get('du_inf', 0) for r in info):.4f}; good speeds "
            + "; ".join(f"{k} <= {max(v['good'], default=0):g}" for k, v in xinfo.items()))
    env_last = passes[-1]["env"]
    Ka_fly_hover = a.Ka_fly
    climb = climb_gain(a, passes, tabs, A_, B_, K_hover) if a.climb_gain != "off" else None
    carries = []
    if a.ilc_carry:
        # carried-lane continuation (run_carry_call): the measured set / envelope accumulate over the calls
        good_cum = np.array([r["good"] for r in passes[-1]["rows"]])
        passed_cum = [r["passed"] for r in passes[-1]["rows"]]
        Ka_c = a.Ka_fly if a.carry_gain == "fly" else a.Ka
        prev = passes[-1]
        for it in range(a.ilc_carry):
            q = run_carry_call(chain, a, prev, tabs_used, e_f, T, a.batch, good_cum, a.seed + 150 + it, Ka_c,
                               f"ILC carry {it + 1} ({a.carry_gain} gain)")
            good_now = np.array([bool(r and r["good"]) for r in q["rows"]])
            # --carry-ok-update: a cell with any usable lane is measured, from its usable lanes only
            upd_of = np.where(q["lane_ok"], q["cell_of"], -1) if a.carry_ok_update else q["cell_of"]
            meas, info = C.ilc_update(tabs_used, q["S"], q["U"], q["A"], q["u_ff"], q["dither"], upd_of, q["cells"],
                                      e_f, q["Tb"], window=tuple(a.window), u_max=a.u_max, good=good_now)
            good_cum = good_cum | good_now
            tabs_used, xinfo = C.extrapolate_tables(meas, tabs, good_cum, q["cells"], max_factor=a.extrap, u_max=a.u_max)
            if a.ff_trims and a.ff_trims_keep:
                tabs_used, _ = splice_trims(a, tabs_used)
            for ci, r in enumerate(q["rows"]):
                if r is not None:
                    passed_cum[ci] = r["passed"]
            q["env"] = C.envelope([{"passed": x} for x in passed_cum], q["cells"], tabs[0]["speeds"], len(tabs))
            q.update(ilc_info=info, extrap=xinfo, good=good_now, good_cum=good_cum.copy())
            carries.append(q)
            prev = q
            log(f"ILC carry {it + 1}: envelope " + "; ".join(f"tilt {tabs[m]['tilt_max_deg']:g}: {q['env'][m]}" for m in q["env"])
                + f" | cells measured now {int(good_now.sum())}, in all {int(good_cum.sum())}/{len(good_cum)}; lanes ok "
                f"{int(q['lane_ok'].sum())}/{a.batch}; good speeds " + "; ".join(f"{k} <= {max(v['good'], default=0):g}" for k, v in xinfo.items()))
        env_last = carries[-1]["env"]
        if a.climb_refit and a.climb_gain != "off":
            # re-solve the climb gain on the pass AND carry flights: the carried UP lanes add sustained-climb data
            # 38-60 wingbeats into the flight, where the roll mode grows (one pass alone gave a weaker gain: X3')
            Ka_first = a.Ka_fly
            a.Ka_fly = Ka_fly_hover
            climb2 = climb_gain(a, passes + carries, tabs, A_, B_, K_hover)
            if climb2 is None or not climb2.get("accepted"):
                a.Ka_fly = Ka_first
                log("climb refit: not accepted, keeping the first climb gain")
            else:
                climb2["first"] = {k: v for k, v in (climb or {}).items() if k not in ("A_climb", "B_climb")}
                climb = climb2
        if climb and climb.get("accepted") and a.carry_gain == "fly":        # logged always; rejects only with --chatter-max
            ch = du_chatter(a, carries[-1], tabs, tuple(a.window))
            ratio = ch["UP"] / max(ch["MOVE"], 1e-12)
            climb["chatter"] = {**ch, "ratio": ratio, "max": a.chatter_max}
            log(f"chatter check (carry {len(carries)}, clean lanes): median per-beat |du|inf UP (V_op..2 V_op) {ch['UP']:.4f}, FWD/REV/DOWN "
                f"{ch['MOVE']:.4f}, HOVER {ch['HOVER']:.4f}; UP / MOVE {ratio:.2f} (max {a.chatter_max:g})" + (" -> REJECT the climb gain: the hover flight gain flies"
                                                                 if a.chatter_max and ratio > a.chatter_max else ""))
            if a.chatter_max and ratio > a.chatter_max:
                a.Ka_fly = Ka_fly_hover
                climb["accepted"] = False
    chosen = max(range(len(tabs)), key=lambda m: (min(env_last[m].values()), -m))     # ties -> map 0 (5 deg)
    if a.ff_trims:
        # the trims live in the largest-tilt map only: flying another map at the raised V_op loses the learn (x8_cg_b
        # chose the 5-deg map on a noisy 15-deg UP carry envelope of 0.30 and survived 1.7%)
        m_tr = max(range(len(tabs)), key=lambda k: tabs[k]["tilt_max_deg"])
        if chosen != m_tr:
            log(f"--ff-trims: map {tabs[m_tr]['tilt_max_deg']:g} deg (the trims') replaces the envelope's choice "
                f"{tabs[chosen]['tilt_max_deg']:g} deg ({env_last[chosen]} vs {env_last[m_tr]})")
        chosen = m_tr
    if a.merge_axes:
        mt, axis_maps = C.merge_axis_tables(tabs_used, env_last, chosen)
        mlp, _ = C.merge_axis_tables(tabs, env_last, chosen)
        env_last = {**env_last, len(tabs): {ax: env_last[m][ax] for ax, m in zip(C.AXES, axis_maps)}}
        tabs_used, tabs = tabs_used + [mt], tabs + [mlp]
        log(f"merged per-axis table (map {len(tabs) - 1}): {mt['note']}")
        chosen = len(tabs) - 1
    log(f"chosen tilt map {tabs[chosen]['tilt_max_deg']:g} deg (last-pass envelope {env_last[chosen]})")
    a.Ka_pol, a.gain_v, fwd = None, None, None
    if a.fwd_gain != "off":
        Ka_fwd, fwd = fwd_gain(a, passes + carries, tabs, chosen, A_, B_)
        if Ka_fwd is not None:
            a.Ka_pol, a.gain_v = np.hstack([a.Ka_fly, Ka_fwd - a.Ka_fly]), tuple(a.fwd_blend)
            # the scheduled gain is a new program shape (x_ref 15, Ka 30): the eval and verify fly their own chain
            chain = MC.Chain(fl, a.batch, a.call_beats, a.u_max, a.zeta_max, H=a.H, alloc_iters=a.alloc_iters, n_max=a.g_nmax)
    timings["calls"]["ilc_pass"] = [s for p in passes for s in p["secs"]]
    if carries:
        timings["calls"]["ilc_carry"] = [s for q in carries for s in q["secs"]]
    diag = diag_local_models(passes[0], A_, B_, Ka, tuple(a.window))
    clock.stage("ilc_pass")
    # 6. evaluation
    tab = tabs_used[chosen]
    NB = max(a.eval_beats, a.continue_to or 0)
    sched, arm, x_ref, u_ff, Ka_b, kicks, push, nbeat = eval_inputs(a, tab, e_f, T, a.batch, NB, env_last[chosen], K_hover)
    st, c_rest = MC.release_carry(fl, a.batch, C.servo_nz(Ka_b[0]))
    n1 = a.eval_beats // a.call_beats
    log(f"eval: {a.batch} flights ({(arm == 'vop').sum()} V_op servo incl. {(push >= 0).sum()} pushed, {(arm == 'uniform').sum()} "
        f"uniform, {(arm == 'baseline').sum()} baseline) x {a.eval_beats} wingbeats = {n1} calls")
    H_b = eval_H(a, arm)
    lt_b = None if a.eval_base_tilt is None else np.where(arm == "baseline", math.radians(a.eval_base_tilt), F.LOST_TILT)
    S1, U1, A1, st, secs = chain.run(st, c_rest, kicks, x_ref, u_ff, np.zeros_like(u_ff), Ka_b, n1, log=log, H=H_b, lost_tilt=lt_b, nbeat=nbeat)
    timings["calls"]["eval"] = secs
    if a.Ka_pol is not None:
        timings["first_call"][f"flight program {a.batch}x{a.call_beats} (scheduled gain)"] = chain.compile_s
    chat = None
    if a.climb_chatter_max and climb and climb.get("accepted"):
        # --climb-chatter-max: the climb gain's own V_op flights (eval UP segments) against FWD / REV / DOWN; a
        # chattering gain is rejected and the eval re-flown with the hover flight gain (the first climb gain is no
        # fallback: in x4_z_a it chattered 6x worse than the refit; round 4 NOTES)
        cands = [("hover flight gain", Ka_fly_hover)]
        chat = {"tried": []}
        while True:
            du = eval_du(U1, A1, sched["seg"], arm, push)
            ok = du["ratio"] <= a.climb_chatter_max
            chat["tried"].append({**du, "ok": bool(ok)})
            log(f"climb chatter gate (eval-{a.eval_beats} V_op flights): median per-beat |du|inf UP {du['UP']:.4f}, FWD/REV/DOWN "
                f"{du['MOVE']:.4f}; UP / MOVE {du['ratio']:.2f} (max {a.climb_chatter_max:g}) -> {'PASS' if ok else 'REJECT'}")
            if ok or not cands:
                break
            name, a.Ka_fly = cands.pop(0)
            climb["accepted"] = False
            climb["chatter_fallback"] = name
            if a.Ka_pol is not None:
                w_ = a.Ka_fly.shape[1]
                a.Ka_pol = np.hstack([a.Ka_fly, a.Ka_pol[:, :w_] + a.Ka_pol[:, w_:] - a.Ka_fly])
            sched, arm, x_ref, u_ff, Ka_b, kicks, push, nbeat = eval_inputs(a, tab, e_f, T, a.batch, NB, env_last[chosen], K_hover)
            st, c_rest = MC.release_carry(fl, a.batch, C.servo_nz(Ka_b[0]))
            log(f"  re-flying the eval with the {name}")
            S1, U1, A1, st, secs = chain.run(st, c_rest, kicks, x_ref, u_ff, np.zeros_like(u_ff), Ka_b, n1, log=log, H=H_b, lost_tilt=lt_b, nbeat=nbeat)
            timings["calls"].setdefault("eval_redo", []).extend(secs)
        chat["final"] = chat["tried"][-1]
    clock.stage("eval")
    Tb = C.beat_times(T, nbeat, g_n0(a))
    m120, _, _ = arm_metrics(S1, U1, A1, sched, x_ref, Tb, e_f, arm, push, a.u_max, a.eval_beats)
    for g in ("vop", "vop_push", "uniform", "baseline"):
        if g in m120:
            log(f"eval {a.eval_beats} beats, {g:9s}: " + brief(m120[g]))
    C.save_policy(out / "policy.npz", tabs=tabs_used, chosen=chosen, Ka=a.Ka_fly, K_hover=K_hover, e_f=e_f, T=T,
                  v_op=list(vop(a)), u_max=a.u_max, zeta_max=a.zeta_max, rate_beats=a.rate_beats, A=A_, B=B_, drift=drift,
                  extra={"tabs_lp_u": np.stack([t["u"] for t in tabs]), "tabs_lp_att": np.stack([t["att"] for t in tabs]),
                         "weights": np.array(json.dumps({**weights(a), "q_int": a.q_int})),
                         "descend_after_stop": a.descend_after_stop, "descend_delay": a.descend_delay, "alloc_iters": a.alloc_iters, "Ka_learn": Ka,
                         "zeta_handoff": a.zeta_handoff, "zeta_gate": -1 if a.zeta_gate is None else a.zeta_gate, "stop_rate": a.stop_rate,
                         "ff_lead": a.ff_lead, **({"rate_beats_z": a.rate_beats_z} if a.rate_beats_z else {}),
                         **({"H": a.H} if a.H is not None else {}),
                         **({"climb_A": climb["A_climb"], "climb_B": climb["B_climb"]} if climb and climb["accepted"] else {}),
                         **({"Ka_fwd": a.Ka_pol[:, :a.Ka_fly.shape[1]] + a.Ka_pol[:, a.Ka_fly.shape[1]:],
                             "fwd_gain_v": np.asarray(a.gain_v, float)} if a.Ka_pol is not None else {}),
                         **({"du_penalty": a.du_penalty, "yaw_int": a.yaw_int} if aug else {}),
                         **({"g_n0": a.g_n0, "g_B": a.g_B, "g_down": a.g_down, "g_down_full": a.g_down_full} if a.g_n0 else {})})
    tr = {"S": S1.astype(np.float32), "U": U1.astype(np.float32), "alive": A1, "v_cmd": sched["v_cmd"], "seg": sched["seg"],
          "arm": arm, "push": push, "kicks_at": np.array([kicks[i, k] if k >= 0 else np.zeros(12) for i, k in enumerate(push)]),
          "x_ref": x_ref.astype(np.float32), "u_ff": u_ff.astype(np.float32), "Ka": np.stack([a.Ka_fly if a.Ka_pol is None else a.Ka_pol,
                                                                                         C.pad_ka(C.hover_ka(K_hover), (a.Ka_fly if a.Ka_pol is None else a.Ka_pol).shape[1])]),
          "ilc_S": passes[0]["S"].astype(np.float32), "ilc_A": passes[0]["A"], "ilc_cell_of": passes[0]["cell_of"],
          **({"nbeat": nbeat} if nbeat is not None else {})}
    np.savez(out / "traces.npz", **tr)
    np.savez(out / "ilc_passes.npz", **{f"p{i}_{k}": np.asarray(p[k]) for i, p in enumerate(passes)
                                        for k in ("S", "U", "A", "u_ff", "x_ref", "cell_of", "good")},
             **{f"c{i}_{k}": np.asarray(q[k]) for i, q in enumerate(carries)
                for k in ("S", "U", "A", "u_ff", "x_ref", "cell_of", "good", "good_cum", "reseed", "lane_ok")})
    metrics = {"eval_120": m120, "linear_sim": lin_sim, "linearize": lin, "linearize_vs_ref": ref, "servo_rho": rho,
               "lp_envelope": {tabs[m]["tilt_max_deg"]: tabs[m]["lp_envelope"] for m in range(len(tabs))},
               "table_notes": [t["note"] for t in tabs], "env_pre": env_pre, "env_per_pass": [p["env"] for p in passes],
               "good_per_pass": [p["good"] for p in passes], "extrap_per_pass": [p["extrap"] for p in passes], "chosen_map": chosen,
               "chosen_tilt_deg": tabs[chosen]["tilt_max_deg"], "ilc_pass_rows": passes[0]["rows"],
               "ilc_info": [p["ilc_info"] for p in passes], "local_models": diag, "config": cfg(a),
               "climb_gain": {k: v for k, v in climb.items() if k not in ("A_climb", "B_climb")} if climb else None}
    if fwd is not None:
        metrics["fwd_gain"] = fwd
    if chat is not None:
        metrics["climb_chatter_gate"] = chat
    if carries:
        metrics["carry"] = [{"frontier": {f"{m}/{C.AXES[ax]}": float(tabs[m]["speeds"][ax, j]) for (m, ax), j in q["frontier"].items()},
                             "env": q["env"], "n_carried": int((~q["reseed"]).sum()), "lanes_ok": int(q["lane_ok"].sum()),
                             "good": q["good"], "good_cum": q["good_cum"], "extrap": q["extrap"], "rows": q["rows"],
                             "ilc_info": q["ilc_info"], "secs": q["secs"]} for q in carries]
    if a.merge_axes:
        metrics["merged_axis_maps"] = axis_maps
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
                                          H=H_b, lost_tilt=lt_b, nbeat=nbeat)
        S_all = np.concatenate([S1[:, :-1], S2], 1); U_all = np.concatenate([U1, U2], 1); A_all = np.concatenate([A1[:, :-1], A2], 1)
        timings["calls"]["continue"] = secs2
        clock.stage("continue_to_%d" % a.continue_to)
        m240, segs, fls = arm_metrics(S_all, U_all, A_all, sched, x_ref, Tb, e_f, arm, push, a.u_max, a.continue_to)
        for g in ("vop", "vop_nopush", "vop_push", "uniform", "baseline", "servo_twin"):
            if g in m240:
                log(f"eval {a.continue_to} beats, {g:10s}: " + brief(m240[g]))
        metrics[f"eval_{a.continue_to}"] = m240
        tr.update(S=S_all.astype(np.float32), U=U_all.astype(np.float32), alive=A_all)
        np.savez(out / "traces.npz", **tr)
    if not a.no_verify:
        pv = run_cell_pass(chain, fl, a, tabs_used, e_f, T, a.batch, a.seed + 200, label="envelope verify (post-ILC)",
                           Ka=a.Ka_fly if a.Ka_pol is None else a.Ka_pol)
        env_post = pv["env"]
        log("post-ILC envelope: " + "; ".join(f"tilt {tabs[m]['tilt_max_deg']:g}: {env_post[m]}" for m in env_post))
        metrics.update(env_post=env_post, verify_pass_rows=pv["rows"], local_models_post=diag_local_models(pv, A_, B_, Ka, tuple(a.window)))
        if climb and climb.get("accepted"):
            vch = du_chatter(a, pv, tabs_used, tuple(a.window))
            metrics["verify_chatter"] = {**vch, "ratio": vch["UP"] / max(vch["MOVE"], 1e-12)}
            log(f"verify chatter (flight gain, clean lanes): median per-beat |du|inf UP (V_op..2 V_op) {vch['UP']:.4f}, FWD/REV/DOWN "
                f"{vch['MOVE']:.4f}, HOVER {vch['HOVER']:.4f}")
        _, info2 = C.ilc_update(tabs_used, pv["S"], pv["U"], pv["A"], pv["u_ff"], pv["dither"], pv["cell_of"], pv["cells"], e_f, pv["Tb"],
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


def aug_kw(a):
    """servo_gains keywords of the augmented flight servo (--yaw-int, --du-penalty); {} = the plain servo."""
    kw = {}
    if getattr(a, "yaw_int", 0.0):
        kw["q_yi"] = a.yaw_int
    if getattr(a, "du_penalty", 0.0):
        kw["r_d"] = a.du_penalty
    return kw


def cfg(a):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k not in ("Ka", "Ka_fly", "Ka_pol", "H", "ff_trims_data")}


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
    rate = C.ref_rate(pol)
    x_ref0, u_ff, v_ref = C.build_refs(sched["v_cmd"], pol["tab"], e_f, T, rate, float(pol.get("descend_after_stop", 0.0)),
                                       int(pol.get("descend_delay", 0)), **ref_kw(pol_ref_args(pol)))
    res = {}
    for dt in a.dt_ms:
        t0 = time.time()
        fl, bird, rel, fp = HG.flight(a, dt)
        Tn = fl.static.n_step * fl.static.dt
        x_ref, nbeat, Tb = x_ref0, None, T
        if "g_n0" in pol:            # crank-speed feedforward: this dt's own wingbeat lengths of the same g
            nbeat = C.g_beats(pol["tab"], e_f, v_ref, fl.static.n_step, int(pol.get("ff_lead", 0)))
            x_ref = C.retime_refs(x_ref0, v_ref, T, nbeat, fl.static.n_step)
            Tb = C.beat_times(T, nbeat, fl.static.n_step)
        chain = MC.Chain(fl, len(sel), a.call_beats, float(pol["u_max"]), float(pol["zeta_max"]), H=pol.get("H"),
                         alloc_iters=int(pol.get("alloc_iters", 0)), n_max=None if nbeat is None else int(nbeat.max()))
        st, c_rest = MC.release_carry(fl, len(sel), C.servo_nz(C.policy_ka(pol)))
        log(f"dt {dt}: snapped {fl.static.dt * 1e3:.5f} ms, {fl.static.n_step} steps/wingbeat (T {Tn:.5f} vs policy {T:.5f})")
        S, U, A, _, secs = chain.run(st, c_rest, np.zeros((len(sel), NB, 12)), x_ref, u_ff, np.zeros_like(u_ff),
                                     np.repeat(C.policy_ka(pol)[None], len(sel), 0), NB // a.call_beats, log=log, nbeat=nbeat)
        segs, fls = C.segment_metrics(S, U, A, sched, x_ref, Tb, e_f=e_f, u_max=float(pol["u_max"]))
        sm = C.summarize(segs, fls)
        log(f"dt {dt} ({len(sel)} flights x {NB} beats): " + brief(sm))
        gates = [G(f"G9 dt {dt} survival", sm["survival"], ">=", 1.0), G(f"G9 dt {dt} follow rate", sm["follow_rate"], ">=", 0.9),
                 G(f"G9 dt {dt} |along| median", sm["along_abs_median"], "<=", 0.03)]
        print_gates(gates)
        res[str(dt)] = {"summary": sm, "gates": gates, "compile_s": chain.compile_s, "call_s": secs, "wall_s": time.time() - t0}
    jdump(run / "dt.json", res)
    log(f"-> {run / 'dt.json'}")


# ---------------------------------------------------------------------------------------------- sensors

def p_servo_gain(a, pol):
    """The P variant's servo gain: the DARE with a soft position integral (--p-q-int) on the policy's hover model
    (--p-gain hover) or on its regressed climb model (--p-gain climb: the model the flight gain of a --climb-gain
    policy is solved on). --p-gain auto (the default) = climb when the policy carries a climb model, else hover (r18):
    on the hover model P loses ~35% of its flights in sustained UP 0.5 climbs (docs/history/command_flight_faster_notes_2026-10.md X9)."""
    wts = {k: v for k, v in (json.loads(str(pol["weights"])) if "weights" in pol else {}).items() if k != "q_int"}
    pg = getattr(a, "p_gain", "hover")
    climb = pg == "climb" or (pg == "auto" and "climb_A" in pol)
    if climb and "climb_A" not in pol:
        raise SystemExit("--p-gain climb: the policy has no regressed climb model (learn --climb-gain reg)")
    A_m, B_m = (pol["climb_A"], pol["climb_B"]) if climb else (pol["A"], pol["B"])
    Ka_P, rho_P = C.servo_gains(A_m, B_m, a.p_q_int, weights=wts)
    if climb:
        rho_h = float(np.abs(np.linalg.eigvals(C.servo_closed(pol["A"], pol["B"], Ka_P))).max())
        log(f"P: servo gain on the climb model (rho {rho_P:.3f}; on the hover model {rho_h:.3f})")
    return Ka_P, rho_P


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
        Ka_P, rho_P = p_servo_gain(a, pol)
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
    ra = pol_ref_args(pol)
    a.zeta_handoff, a.zeta_gate, a.stop_rate, a.ff_lead = ra.zeta_handoff, ra.zeta_gate, ra.stop_rate, ra.ff_lead
    a.rate_beats, a.call_beats, a.dither = float(pol["rate_beats"]), a.ilc_beats, 0.0
    tabs_s = [{k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in tab.items()} for _ in names]
    for it in range(a.sensor_ilc):
        cells, cell_of, v_cmd, xr, uf, *_ = cell_pass_inputs(a, tabs_s, e_f, T, a.batch, np.random.default_rng(a.seed + 300 + it),
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
    rate = C.ref_rate(pol)
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
    tg, control, dt, n_beats, Ka, x_ref, u_ff, kicks, u_max, zeta_max, contacts, H, iters = job[:13]
    g = job[13] if len(job) > 13 else None
    bird, rel, fp = HG.load(tg, control, dt)
    if not contacts:          # as the device (--contact off): the scene's floor is 0.25 m below the release COM
        import mujoco
        bird.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    pol = C.CommandPolicy(Ka, x_ref, u_ff, u_max=u_max, zeta_max=zeta_max, H=H, iters=iters, g=g)
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
    Kas = {"vop": C.policy_ka(pol), "baseline": C.pad_ka(C.hover_ka(pol["K_hover"]), C.policy_ka(pol).shape[1])}
    jobs = []
    for i in sel:
        kicks = {int(push[i]): tr["kicks_at"][i]} if push[i] >= 0 else None
        jobs.append((a.tg, a.control, 0.13, NB, Kas["baseline" if arm[i] == "baseline" else "vop"], tr["x_ref"][i].astype(float),
                     tr["u_ff"][i].astype(float), kicks, float(pol["u_max"]), float(pol["zeta_max"]), a.cpu_contacts,
                     (np.eye(24) if arm[i] == "baseline" else pol["H"]) if "H" in pol else None, int(pol.get("alloc_iters", 0)))
                    + ((C.g_of_steps(tr["nbeat"][i], int(pol["g_n0"])),) if "nbeat" in tr.files else ()))
    log(f"CPU reference: {len(jobs)} flights x {NB} wingbeats on {a.procs} processes")
    t = time.time()
    with Pool(a.procs) as p:
        res = p.map(_cpu_job, jobs, chunksize=1)
    log(f"  {time.time() - t:.0f} s")
    e_f, T = pol["e_f"], float(pol["T"])
    Tb_all = C.beat_times(T, tr["nbeat"], int(pol["g_n0"])) if "nbeat" in tr.files else None
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
        Ti = T if Tb_all is None else Tb_all[i:i + 1]
        segc, _ = C.segment_metrics(Sc, Ucp, Ac, sub, tr["x_ref"][i:i + 1].astype(float), Ti, e_f=e_f)
        segg, _ = C.segment_metrics(gpu_full[None], tr["U"][i:i + 1, :NB].astype(float), ag[None], sub, tr["x_ref"][i:i + 1].astype(float), Ti, e_f=e_f)
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


def cmd_trims(a):
    """Measured steady-flight trims at the FWD / REV edge: the `learn --ff-trims` input (the shipped
    designs/hummer/free-hover/command_trims.npz). hover.mjx_cradle.trim_at_speed per cell on the moving cradle (device,
    x64 unless --f32; x = 24 harmonic depths + tilt_x, tilt_y, yaw; crank speed g and a0 fixed). FWD and REV advance
    together one grid speed per stage: stage 0 is seeded from --policy's table entries (u, att) at that speed, stage 1
    from stage 0 + the table's own slope, later stages by linear extrapolation of the two previous trims;
    --seed-trims instead replays the starting points stored in an earlier trims.npz (x0). Then the command-table
    entries (att, dv, w; hover.mjx_cradle.orbit_entries) of every trim relative to the hover cradle.
    Writes <out>/trims.npz (+ trims.json: per-stage residual history, wall time). Untimed (~24 min on an RTX 5070 Ti)."""
    import jax
    if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    from hover import mjx_flight as F
    if not a.f32:
        F.enable_x64()
    from hover import launch as L
    from hover import mjx_cradle as MCR
    from hover.policy import load_flight_policy
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fp = load_flight_policy(a.control)
    nu = 8 * fp.K
    pol = C.load_policy(a.policy)
    tab = pol["tabs"][int(pol["chosen_map"]) if a.map is None else a.map]
    e_f = np.asarray(pol["e_f"], float)
    seed = None
    if a.seed_trims is not None:
        z = np.load(a.seed_trims)
        seed = {(int(ax), round(float(s), 6)): x0 for ax, s, x0 in zip(z["axis"], z["speed"], z["x0"])}
    bird = L.load_bird(a.tg, control=a.control)
    cr = MCR.Cradle(bird, fp.K, k_phi=fp.phase_lock, ls_iterations=8, disable_contact=True)
    log(f"trims: cradle ready ({'float32' if a.f32 else 'float64'}); seed policy {a.policy}, map {tab['tilt_max_deg']:g} deg, "
        f"e_f {e_f.round(4)}" + (f"; starting points from {a.seed_trims}" if seed else ""))
    sp = tab["speeds"]

    def tab_x(axis, s):
        return np.r_[C._interp(sp[axis], tab["u"][axis], np.array([s]))[0], C._interp(sp[axis], tab["att"][axis], np.array([s]))[0]]

    dirs = {0: e_f, 1: -e_f}
    lanes = [(0, a.fwd), (1, a.rev)]
    trims = {0: [], 1: []}
    res = {"config": {k: str(v) for k, v in vars(a).items()}, "stages": []}
    X_h = None
    for k in range(max(len(a.fwd), len(a.rev))):
        vs, x0s, who = [], [], []
        for ax, speeds in lanes:
            if k >= len(speeds):
                continue
            s, prev = speeds[k], trims[ax]
            if seed is not None and (ax, round(s, 6)) in seed:
                x0 = np.array(seed[(ax, round(s, 6))], float)
            elif k == 0:
                x0 = tab_x(ax, s)
            elif k == 1:
                x0 = prev[-1]["x"] + (tab_x(ax, s) - tab_x(ax, speeds[0]))
            else:
                x0 = prev[-1]["x"] + (prev[-1]["x"] - prev[-2]["x"]) * (s - speeds[k - 1]) / (speeds[k - 1] - speeds[k - 2])
            x0[:nu] = np.clip(x0[:nu], -0.135, 0.135)
            vs.append(s * dirs[ax]); x0s.append(x0); who.append((ax, s))
        t = time.time()
        log(f"stage {k}: {[(C.AXES[ax], s) for ax, s in who]}")
        res_k, X_h = MCR.trim_at_speed(cr, fp, np.array(vs), x0s, iters=a.iters, trust=a.trust, ang_trust=math.radians(a.ang_trust_deg),
                                       n_cradle=a.n_cradle, n_avg=a.n_avg, log=log, min_iters=a.min_iters,
                                       ang_box=np.radians([a.tilt_max_deg, a.tilt_max_deg, a.yaw_max_deg]))
        for (ax, s), o, v in zip(who, res_k, vs):
            o.update(axis=ax, speed=s, v=v, entries=MCR.orbit_entries(o["X"], X_h, v))
            trims[ax].append(o)
            x, ent = o["x"], o["entries"]
            log(f"  {C.AXES[ax]} {s:.2f}: res {o['residual']:.4f} |u|inf {np.abs(x[:nu]).max():.4f} tilt "
                f"{np.round(np.degrees(x[nu:nu + 2]), 2).tolist()} yaw {np.degrees(x[nu + 2]):.2f} deg | entries att "
                f"{np.round(np.degrees(ent['att']), 2).tolist()} deg, dv {np.round(ent['dv'], 4).tolist()}, w {np.round(ent['w'], 4).tolist()}")
        res["stages"].append({"k": k, "wall_s": time.time() - t, "cells": [{"axis": C.AXES[ax], "speed": s, "residual": o["residual"],
                                                                           "history": o["history"]} for (ax, s), o in zip(who, res_k)]})
        log(f"stage {k}: {time.time() - t:.1f} s")
        jdump(out / "trims.json", res)
    rows = [o for ax in (0, 1) for o in trims[ax]]
    np.savez(out / "trims.npz", axis=np.array([o["axis"] for o in rows]), speed=np.array([o["speed"] for o in rows]),
             v=np.array([o["v"] for o in rows]), x=np.array([o["x"] for o in rows]), w=np.array([o["w"] for o in rows]),
             residual=np.array([o["residual"] for o in rows]), X=np.array([o["X"] for o in rows]), X_hover=X_h,
             att=np.array([o["entries"]["att"] for o in rows]), dv=np.array([o["entries"]["dv"] for o in rows]),
             w_ent=np.array([o["entries"]["w"] for o in rows]),
             x0=np.array([o["history"][0]["x"] for o in rows]), w0=np.array([o["history"][0]["w"] for o in rows]),
             theta0=fp.theta, att0=np.asarray(fp.attitude, float), policy=str(a.policy))
    res["wall_s"] = time.time() - T_START
    jdump(out / "trims.json", res)
    log(f"-> {out / 'trims.npz'} (wall {res['wall_s']:.0f} s); install: cp {out / 'trims.npz'} {TRIMS_FAST}")


class _CrankRec:
    """A pass-through hover.launch.fly ``sensor``: records the cranks' actuator_force / actuator_velocity / ctrl at every
    step (the policy still reads the true strobed state)."""

    def __init__(self, ids):
        self.ids, self.f, self.v, self.c = ids, [], [], []

    def start(self, bird, rel, sim, theta):
        pass

    def step(self, d):
        self.f.append(d.actuator_force[self.ids].copy()); self.v.append(d.actuator_velocity[self.ids].copy())
        self.c.append(d.ctrl[self.ids].copy())

    def read(self, d, s):
        return s


_ACT0 = {}


def _torque_job(job):
    """One CPU flight (hover.launch.fly, float64, the model's own solver, contacts off) of a CommandPolicy on given
    references, recording every step's crank actuator force. ``limit`` [N m]: the cranks forcelimited to +-limit
    (default: the compiled model's own actuator settings). Returns per-step arrays + the strobed states."""
    import mujoco
    from hover import launch as L
    bird, rel, fp = HG.load(job["tg"], job["control"], 0.13)
    m = bird.model
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    ids = list(rel.sim.crank_ids)
    key = id(m)
    if key not in _ACT0:
        _ACT0[key] = (m.actuator_forcelimited[ids].copy(), m.actuator_forcerange[ids].copy())
    m.actuator_forcelimited[ids], m.actuator_forcerange[ids] = _ACT0[key]
    if job["limit"]:
        m.actuator_forcelimited[ids] = 1
        m.actuator_forcerange[ids] = [-job["limit"], job["limit"]]
    pol = C.CommandPolicy(job["Ka"], job["x_ref"], job["u_ff"], u_max=job["u_max"], zeta_max=job["zeta_max"], H=job["H"],
                          iters=job["iters"], g=job["g"])
    rec = _CrankRec(ids)
    warn = []
    mujoco.set_mju_user_warning(lambda msg: warn.append((len(pol.U), str(msg))))
    t = time.time()
    try:
        tr = L.fly(bird, rel, fp.theta, job["n_beats"], policy=pol, kicks=job["kicks"], sensor=rec)
    finally:
        mujoco.set_mju_user_warning(None)
    n = int(round(2 * math.pi / (bird.omega * math.exp(fp.theta[0])) / m.opt.timestep))
    return {"S": tr, "f": np.array(rec.f), "v": np.array(rec.v), "c": np.array(rec.c), "U": np.array(pol.U), "n": n,
            "warn": warn[0] if warn else None, "sec": time.time() - t}


def crank_stats(r, w0, skip, g=None):
    """Crank actuator statistics of one recorded flight over wingbeats [w0, end): peak_all = max |actuator_force| over
    every step (beat-boundary impulses included), peak_smooth = the same without the first ``skip`` steps of each
    wingbeat, RMS per crank, mean mechanical power (sum over cranks of mean force x velocity), ctrl peak."""
    f, v, c = r["f"], r["v"], r["c"]
    nb = len(r["U"])
    lens = np.rint(r["n"] * np.exp(-(np.zeros(nb) if g is None else np.asarray(g)[:nb]))).astype(int)
    edges = np.r_[0, np.cumsum(lens)]
    nb = int(min(nb, np.searchsorted(edges, len(f), "right") - 1))
    beats = [np.abs(f[edges[k]:edges[k + 1]]) for k in range(w0, nb)]
    if not beats:
        return None
    pk = np.array([b.max() for b in beats])
    sm = np.array([b[skip:].max() for b in beats])
    sl = slice(edges[w0], edges[nb])
    return {"beats": [w0, nb], "peak_all_mNm": float(pk.max() * 1e3), "peak_all_beat": int(w0 + pk.argmax()),
            "peak_smooth_mNm": float(sm.max() * 1e3), "boundary_spike_median_mNm": float(np.median([b[:skip].max() for b in beats]) * 1e3),
            "rms_mNm_per_crank": (np.sqrt((f[sl] ** 2).mean(0)) * 1e3).round(1).tolist(),
            "power_W": float((f[sl] * v[sl]).mean(0).sum()), "ctrl_peak": float(np.abs(c[sl]).max())}


def cmd_torque(a):
    """Crank actuator cost on CPU (hover.launch.fly, float64, the model's own solver, contacts off): constant-command
    flights (HOVER, every axis at the policy's V_op and at its exported envelope, + --cells) of --beats wingbeats measured
    over beats --window-from.., and with --run also --eval-flights of its random V_op eval flights (traces.npz; command
    changes included) over beats 10... Reports the full-step peak |actuator_force| (peak_all: every step, the one-step
    wingbeat-boundary impulses included) and the smooth peak (the first --skip steps of each wingbeat left out), both
    as a ratio to HOVER's; FLAG = either ratio > --flag-ratio. --limit L re-flies every flight with the cranks
    forcelimited to +-L N m (the compiled model has no torque limit: kv-1 velocity servos) and reports whether they still
    fly and follow. Writes <run>/torque.json (or --out)."""
    import mujoco
    pol_path = Path(a.policy) if a.policy else Path(a.run) / "policy.npz"
    pol = C.load_policy(pol_path)
    e_f, T = np.asarray(pol["e_f"], float), float(pol["T"])
    vo = np.asarray(pol["v_op"], float)
    v_op = {"FWD": vo[0], "UP": vo[1], "DOWN": vo[2] if len(vo) > 2 else vo[1], "REV": vo[3] if len(vo) > 3 else vo[0]}
    cells = [("HOVER", 0.0)] + [(ax, float(v_op[ax])) for ax in C.AXES]
    if "envelope" in pol:                                    # exported policy: FWD, REV, UP, DOWN
        cells += [(ax, float(s)) for ax, s in zip(C.AXES, pol["envelope"])]
    for c in a.cells or []:
        ax, s = c.split(":")
        cells.append((ax, float(s)))
    cells = list(dict.fromkeys((ax, round(s, 4)) for ax, s in cells))
    rate = C.ref_rate(pol)
    kw = ref_kw(pol_ref_args(pol))
    common = dict(tg=a.tg, control=a.control, u_max=float(pol["u_max"]), zeta_max=float(pol["zeta_max"]),
                  H=pol["H"] if "H" in pol else None, iters=int(pol.get("alloc_iters", 0)), Ka=C.policy_ka(pol))
    dirs = C.command_dirs(e_f)
    jobs, meta = [], []
    for ax, s in cells:
        v_cmd = np.broadcast_to(s * dirs[C.CMDS.index(ax)], (1, a.beats, 3)).copy()
        x_ref, u_ff, v_ref = C.build_refs(v_cmd, pol["tab"], e_f, T, rate, float(pol["descend_after_stop"]),
                                          int(pol["descend_delay"]), **kw)
        g = None
        if "g_n0" in pol:
            n0 = int(pol["g_n0"])
            nbeat = C.g_beats(pol["tab"], e_f, v_ref, n0, int(pol.get("ff_lead", 0)))
            x_ref, g = C.retime_refs(x_ref, v_ref, T, nbeat, n0), C.g_of_steps(nbeat[0], n0)
        meta.append({"kind": "cell", "cmd": ax, "speed": s, "v_cmd": s * dirs[C.CMDS.index(ax)], "g": g})
        jobs.append({**common, "x_ref": x_ref[0], "u_ff": u_ff[0], "g": g, "kicks": None, "n_beats": a.beats})
    if a.run and a.eval_flights:
        tr = np.load(Path(a.run) / "traces.npz")
        tbeat = C.beat_times(T, tr["nbeat"], int(pol["g_n0"])) if "nbeat" in tr.files else None
        sel = np.flatnonzero((tr["arm"] == "vop") & (tr["push"] < 0))[:a.eval_flights]
        NB = min(a.eval_beats, tr["S"].shape[1] - 1)
        for i in sel:
            g = C.g_of_steps(tr["nbeat"][i], int(pol["g_n0"])) if "nbeat" in tr.files else None
            meta.append({"kind": "eval", "flight": int(i), "g": g, "NB": NB})
            jobs.append({**common, "x_ref": tr["x_ref"][i].astype(float), "u_ff": tr["u_ff"][i].astype(float), "g": g,
                         "kicks": None, "n_beats": NB})
    arms = [None] + ([a.limit] if a.limit else [])
    alljobs = [{**j, "limit": lim} for lim in arms for j in jobs]
    log(f"torque: {len(cells)} constant-command cells x {a.beats} beats + {len(jobs) - len(cells)} eval flights, limits {arms}, "
        f"{len(alljobs)} CPU flights on {a.procs} processes; policy {pol_path}")
    bird, rel, _ = HG.load(a.tg, a.control, 0.13)
    ids = list(rel.sim.crank_ids)
    mm = bird.model
    act = {"forcelimited": mm.actuator_forcelimited[ids].tolist(), "forcerange": mm.actuator_forcerange[ids].tolist(),
           "ctrlrange": mm.actuator_ctrlrange[ids].tolist(), "ctrllimited": mm.actuator_ctrllimited[ids].tolist(),
           "gainprm0": mm.actuator_gainprm[ids, 0].tolist(), "biasprm": mm.actuator_biasprm[ids, :3].tolist()}
    log(f"  compiled crank actuators: {act}")
    t = time.time()
    with Pool(a.procs) as p:
        res = p.map(_torque_job, alljobs, chunksize=1)
    log(f"  {time.time() - t:.0f} s")
    out = {"policy": str(pol_path), "beats": a.beats, "window_from": a.window_from, "skip": a.skip, "actuators": act,
           "flag_ratio": a.flag_ratio, "arms": {}}
    for li, lim in enumerate(arms):
        rows = []
        for (m_, r) in zip(meta, res[li * len(jobs):(li + 1) * len(jobs)]):
            nb = len(r["U"])
            row = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m_.items() if k not in ("g", "NB")}
            row["survived"] = bool(r["warn"] is None and np.all(np.linalg.norm(r["S"][:, 3:6], axis=1) < math.radians(60)))
            row["beats_flown"] = nb
            st = crank_stats(r, a.window_from if m_["kind"] == "cell" else 10, a.skip, m_["g"])
            row.update(st or {})
            if m_["kind"] == "cell" and st:
                S, (w0, nb) = r["S"], st["beats"]           # mean-motion velocity (strobed position over the window)
                Tw = T * (nb - w0) if m_["g"] is None else T * float(np.exp(-np.asarray(m_["g"])[w0:nb]).sum())
                row["v_err"] = float(np.linalg.norm((S[nb, :3] - S[w0, :3]) / Tw - m_["v_cmd"]))
            if m_["kind"] == "eval":                        # segment verdicts, as `cpu` grades them
                i, NB = m_["flight"], m_["NB"]
                cpu = r["S"][:NB + 1] if r["warn"] is None else r["S"][:r["warn"][0] + 1]
                n_c = len(cpu)
                Sc = np.zeros((1, NB + 1, 12)); Sc[0, :n_c] = cpu; Sc[0, n_c:] = cpu[-1]
                Ac = np.zeros((1, NB + 1), bool); Ac[0, :n_c] = True
                Ac[0] &= np.linalg.norm(Sc[0, :, 3:6], axis=1) < math.radians(60)
                Ucp = np.zeros((1, NB, 24)); Ucp[0, :min(NB, len(r["U"]))] = r["U"][:NB]
                Ti = T if tbeat is None else tbeat[i:i + 1]
                seg, _ = C.segment_metrics(Sc, Ucp, Ac, {"v_cmd": tr["v_cmd"][i:i + 1, :NB], "seg": tr["seg"][i:i + 1]},
                                           tr["x_ref"][i:i + 1].astype(float), Ti, e_f=e_f)
                row["segments"], row["segments_passed"] = len(seg), int(sum(sg["passed"] for sg in seg))
            rows.append(row)
        if li == 0:
            hov = next(r for r in rows if r.get("cmd") == "HOVER")         # ratios: to the model's own (unlimited) hover
        for r in rows:
            if "peak_all_mNm" in r:
                r["ratio_all"] = r["peak_all_mNm"] / hov["peak_all_mNm"]
                r["ratio_smooth"] = r["peak_smooth_mNm"] / hov["peak_smooth_mNm"]
                r["flag"] = bool(lim is None and max(r["ratio_all"], r["ratio_smooth"]) > a.flag_ratio)
        out["arms"]["none" if lim is None else f"{lim:g}"] = rows
        log(f"  crank limit {'none (the model)' if lim is None else f'+-{lim:g} N m'}: peak_all mNm (x hover) | smooth (x hover) | "
            f"spike med | RMS per crank | power W | |mean-motion v - v_cmd| | flag (ratios: to the unlimited HOVER)")
        for r in rows:
            name = f"{r['cmd']} {r['speed']:.2f}" if r["kind"] == "cell" else f"eval {r['flight']}"
            if "peak_all_mNm" not in r:
                log(f"    {name:12s} LOST at {r['beats_flown']}")
                continue
            log(f"    {name:12s} {'ok  ' if r['survived'] else 'LOST'} {r['peak_all_mNm']:7.1f} ({r['ratio_all']:.2f}x) | {r['peak_smooth_mNm']:6.1f} "
                f"({r['ratio_smooth']:.2f}x) | {r['boundary_spike_median_mNm']:6.1f} | {r['rms_mNm_per_crank']} | {r['power_W']:.2f} | "
                f"{r.get('v_err', float('nan')):.4f}" + (f" | segments {r['segments_passed']}/{r['segments']}" if "segments" in r else "")
                + f" | {'FLAG' if r['flag'] else ''}")
    dst = Path(a.out) if a.out else (Path(a.run) / "torque.json" if a.run else Path("torque.json"))
    jdump(dst, out)
    log(f"-> {dst}")


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
    env_post = dict(m["env_post"][str(int(pol["chosen_map"]))])
    # the stored envelope (LiveCommand / the viewer clip to it) = the CERTIFIED operating speed per axis (the
    # random-command eval flew exactly these); the post-ILC V_max (30-beat constant-command cells, run-to-run
    # noisy: it came out above and below V_op) is kept as envelope_post_ilc for information
    vo = [float(x) for x in pol["v_op"]]
    env = dict(zip(C.AXES, (vo[0], vo[3] if len(vo) > 3 else vo[0], vo[1], vo[2] if len(vo) > 2 else vo[1])))
    log("envelope = certified V_op: " + ", ".join(f"{k} {env[k]:g}" for k in C.AXES)
        + " (post-ILC V_max " + ", ".join(f"{k} {env_post[k]:g}" for k in C.AXES) + ")")
    T = float(pol["T"])
    Ka_P, rho_P = p_servo_gain(a, pol)
    P = Rd.design(Rd.load_model(a.sensor_lqr), a.p_obs, "lqg", beat_s=T, bias=a.p_bias or None, bias_walk=a.p_bias_walk,
                  noise=Sn.SensorNoise(scale=a.p_design_noise, pos_latency_ms=a.pos_latency_ms), process_scale=a.p_process_scale)
    z.update(sensor_Ka=Ka_P, sensor_q_int=a.p_q_int, sensor_climb_after_stop=a.p_climb_after_stop,
             sensor_pos_latency_ms=a.pos_latency_ms, envelope=np.array([env[k] for k in C.AXES]),
             envelope_post_ilc=np.array([env_post[k] for k in C.AXES]))
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


# `learn --preset fast` (round 7, docs/history/command_flight_faster_notes_2026-10.md): the round-6 config K (opt/runs_fast/x8_k7565_a) as defaults;
# any flag given on the command line still overrides. No --preset (= r18) keeps every old default, so r18 reproduces.
TRIMS_FAST = Path(__file__).resolve().parents[1] / "designs/hummer/free-hover/command_trims.npz"
PRESETS = {
    "r18": {},
    "fast": dict(lin_rigid_rotation=True, tilt_maps=[5.0, 15.0], climb_gain="reg", climb_speeds=[0.3, 0.6],
                 climb_chatter_max=1000.0, climb_refit=True, climb_window=[6, 20],
                 v_op_h=0.75, v_op_rev=0.65, v_op_up=0.5, v_op_down=0.45, rate_beats_z=6.0,
                 ilc_iters=1, ilc_carry=1, carry_ok_update=True, carry_dither=0.0, call_beats=20, window=[10, 20],
                 pass_ramp_beats=8.0, batch=384, ff_trims=TRIMS_FAST, ff_trims_keep=True),
}


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
    g.add_argument("--lin-rigid-rotation", action="store_true", help="the linearization's attitude perturbations also "
                   "rotate the free bodies' linear velocities about the COM velocity (hover.launch.rotate_bird rotate_vel; "
                   "default off = the r18 / hover.lqr model, whose att columns carry a spurious internal velocity change)")
    g.add_argument("--contact", dest="no_contact", action="store_false")
    pol = argparse.ArgumentParser(add_help=False)
    g = pol.add_argument_group("policy")
    g.add_argument("--v-op-h", type=float, default=C.V_OP_H, help="FWD / REV operating speed [m/s]")
    g.add_argument("--v-op-z", type=float, default=C.V_OP_Z, help="UP / DOWN operating speed [m/s]")
    g.add_argument("--v-op-rev", type=float, default=None, help="REV operating speed (default --v-op-h, which stays FWD's and "
                   "sets the horizontal reference rate)")
    g.add_argument("--v-op-down", type=float, default=None, help="DOWN operating speed (default --v-op-z)")
    g.add_argument("--v-op-up", type=float, default=None, help="UP operating speed (default --v-op-z; it replaces --v-op-z as "
                   "the UP speed and the vertical reference rate, and --v-op-down then defaults to --v-op-z)")
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
    g.add_argument("--climb-gain", default="off", choices=("off", "reg"), help="reg: the flight gain re-solved on a "
                   "model regressed from the ILC passes' dithered climb flights (fixes the sustained-climb roll departure "
                   "of the rigid-rotation model; default off = the hover-model gain)")
    g.add_argument("--climb-speeds", type=float, nargs=2, default=[0.24, 0.48], help="--climb-gain: UP cell speeds regressed")
    g.add_argument("--climb-reg", default="fixB", choices=("fixB", "full"), help="--climb-gain: regress A only with B "
                   "= the hover linearization's (fixB), or A and B (full)")
    g.add_argument("--climb-ridge", type=float, default=0.3, help="--climb-reg fixB: shrink the regressed A toward the "
                   "hover model's (relative Tikhonov weight; 0 = plain least squares)")
    g.add_argument("--climb-rho-max", type=float, default=0.95, help="--climb-gain: accept the climb gain only if its "
                   "closed loop on the hover model has |eig| <= this (else the ridge is raised x3, up to 3 times)")
    g.add_argument("--climb-window", type=int, nargs=2, default=[8, 30], help="--climb-gain: beats regressed")
    g.add_argument("--fwd-gain", default="off", choices=("off", "reg"), help="reg: a speed-scheduled FORWARD gain regressed "
                   "(A only, B from hover, ridge toward the hover A) from the learn's own FWD pass / carry flights, blended with "
                   "the flight gain by the reference forward speed (hover.command.gain_weight)")
    g.add_argument("--fwd-speeds", type=float, nargs=2, default=[0.5, 0.7], help="--fwd-gain: FWD cell speeds regressed")
    g.add_argument("--fwd-ridge", type=float, default=0.3, help="--fwd-gain: shrink the regressed A toward the hover A")
    g.add_argument("--fwd-window", type=int, nargs=2, default=None, help="--fwd-gain: beats regressed (default --climb-window)")
    g.add_argument("--fwd-blend", type=float, nargs=2, default=[0.4, 0.6], help="--fwd-gain: the weight of the FWD gain rises "
                   "from 0 to 1 as the reference forward speed goes from lo to hi [m/s]")
    g.add_argument("--du-penalty", type=float, default=0.0, help="augmented flight servo: the DARE state gets the previous "
                   "wingbeat's delivered feedback u_{k-1} and the input rate u_k - u_{k-1} is penalised with this x the input "
                   "weight R (hover.command.servo_gains_aug; the flight, climb and FWD gains; the ILC learning gain stays "
                   "plain). 0 = off")
    g.add_argument("--yaw-int", type=float, default=0.0, help="augmented flight servo: a yaw-angle integral state with this "
                   "LQR weight, gated / handed off like the position integral. 0 = off")
    g.add_argument("--ff-trims", type=Path, default=None, help="X8: a trims.npz of measured moving-cradle trims "
                   "(`opt/hover_cmd.py trims`) spliced into the largest-tilt map's FWD / REV base entries at speeds >= "
                   "--ff-trims-from before the ILC (seeds the ILC refines). Precomputed outside learn (untimed)")
    g.add_argument("--ff-trims-from", type=float, nargs=2, default=[0.70, 0.60], help="--ff-trims: FWD, REV grid speeds "
                   "from which the trims replace the tables")
    g.add_argument("--ff-trims-keep", action="store_true", help="--ff-trims: re-splice the trims after every ILC update "
                   "(the trim cells fly the trims themselves, unrefined)")
    g.add_argument("--g-down", type=float, default=0.0, help="crank-speed feedforward on DOWN (X8): g(s) = this x min(1, s / "
                   "--g-down-full) (e.g. -0.05 = 5%% slower cranks at DOWN >= --g-down-full), quantized to integer wingbeat "
                   "lengths; flown with variable-length wingbeats on the device. 0 = off (the fixed crank speed)")
    g.add_argument("--g-down-full", type=float, default=0.4, help="--g-down: DOWN speed [m/s] where the full g is reached")
    g.add_argument("--g-fd-steps", type=int, default=3, help="--g-down: steps of the CPU central difference for B_g")
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
    g.add_argument("--rate-beats-z", type=float, default=None, help="the vertical reference reaches V_op z in this many "
                   "wingbeats (default --rate-beats; the ILC passes keep --rate-beats); saved in the policy (hover.command.ref_rate)")
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
    learn_parsers = {}
    for name in ("learn", "all"):
        p = sub.add_parser(name, parents=[dev, pol])
        learn_parsers[name] = p
        p.add_argument("--preset", default="r18", choices=tuple(PRESETS), help="defaults of the whole flag set: r18 = the "
                       "shipped r18 config (every default below); fast = round 6's config K (V_op FWD 0.75 / REV 0.65 / UP 0.5 / "
                       "DOWN 0.45: --lin-rigid-rotation --tilt-maps 5 15 --climb-gain reg --climb-speeds 0.3 0.6 --climb-refit "
                       "--climb-window 6 20 --climb-chatter-max 1000 --rate-beats-z 6 --ilc-iters 1 --ilc-carry 1 --carry-ok-update "
                       "--carry-dither 0 --call-beats 20 --window 10 20 --pass-ramp-beats 8 --batch 384 --ff-trims "
                       "designs/hummer/free-hover/command_trims.npz --ff-trims-keep). Flags given explicitly override the preset")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--batch", type=int, default=512, help="B: flights per device call (the one program shape)")
        p.add_argument("--eval-beats", type=int, default=120)
        p.add_argument("--continue-to", type=int, default=240, help="untimed: extend the eval flights (0 = no)")
        p.add_argument("--ilc-iters", type=int, default=2, help="ILC / continuation passes before the eval")
        p.add_argument("--ilc-carry", type=int, default=0, help="after the --ilc-iters passes, this many carried-lane "
                       "continuation calls (run_carry_call: the lanes keep flying, good ones step up --carry-step speeds, lost "
                       "ones re-seed at the measured frontier); 0 = off")
        p.add_argument("--carry-step", type=int, default=1, help="--ilc-carry: grid speeds a good lane steps up per call")
        p.add_argument("--carry-ahead", type=int, default=3, help="--ilc-carry: lanes only on cells <= the frontier + this many")
        p.add_argument("--carry-dither", type=float, default=None, help="--ilc-carry: dither of the carry calls' copies (default "
                       "--dither; the dither costs ~0.15 s per wingbeat of device time: the solver iterates more at each beat's input step)")
        p.add_argument("--pass-ramp-beats", type=float, default=None, help="ILC / carry lanes reach their cell speed in at most this "
                       "many wingbeats (default: 2 --rate-beats; short --call-beats need the window at speed)")
        p.add_argument("--carry-ok-update", action="store_true", help="--ilc-carry: a cell is measured when ANY of its lanes is "
                       "usable, from the usable lanes only (default: when half of its lanes are, from all its alive lanes)")
        p.add_argument("--carry-gain", default="fly", choices=("fly", "learn"), help="--ilc-carry: the calls fly the flight gain "
                       "(after --climb-gain) or the learning gain")
        p.add_argument("--chatter-max", type=float, default=0.0, help="--climb-gain with --ilc-carry (fly gain): reject the climb gain "
                       "if the last carry call's clean UP lanes' median per-beat |du|inf exceeds this x the FWD / REV / DOWN lanes' (at <= V_op; "
                       "du_chatter; 0 = off)")
        p.add_argument("--eval-base-tilt", type=float, default=None, help="loss tilt [deg] of the eval's baseline-LQR arm "
                       "(default: the eval's 60 deg): a lower one parks its tumbling flights sooner, which shortens the eval calls "
                       "(the vmapped solver runs to its cap for the whole batch while one flight tumbles)")
        p.add_argument("--climb-chatter-max", type=float, default=0.0, help="--climb-gain: acceptance gate on the gain's own V_op "
                       "flights: if the eval's UP segments' median per-beat |du|inf exceeds this x the FWD/REV/DOWN median, the gain "
                       "is rejected and the eval re-flown with the hover flight gain (0 = off)")
        p.add_argument("--climb-refit", action="store_true", help="--climb-gain with --ilc-carry: re-solve the climb gain after "
                       "the carry calls on the pass + carry flights (the carried UP lanes climb 30-60 wingbeats in)")
        p.add_argument("--merge-axes", action="store_true", help="fly a merged table: every axis from the tilt map with the "
                       "best last-pass envelope on it (hover.command.merge_axis_tables)")
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
            p.add_argument("--p-gain", default="auto", choices=("auto", "hover", "climb"),
                           help="P servo gain: the --p-q-int DARE on the "
                           "policy's hover model or on its regressed climb model (auto: climb when the policy has one; p_servo_gain)")
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
    p.add_argument("--p-gain", default="auto", choices=("auto", "hover", "climb"),
                   help="P servo gain: the --p-q-int DARE on the "
                   "policy's hover model or on its regressed climb model (auto: climb when the policy has one; p_servo_gain)")
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
    p = sub.add_parser("trims", help="measured moving-cradle trims at the FWD / REV edge (the learn --ff-trims input; GPU, x64)")
    p.add_argument("--out", type=Path, required=True, help="directory: trims.npz + trims.json")
    p.add_argument("--policy", type=Path, default=FREE_HOVER_TG.parent / "command_policy.npz", help="seed tables (a run's "
                   "policy.npz or an exported *_policy.npz; the shipped command_trims.npz was seeded from round 5's x7_d45c)")
    p.add_argument("--map", type=int, default=None, help="seed table index (default: the policy's chosen map)")
    p.add_argument("--seed-trims", type=Path, default=None, help="replay the starting points (x0) stored in this trims.npz "
                   "for the cells it has (e.g. the shipped command_trims.npz: re-solves it on the current bird)")
    p.add_argument("--fwd", type=float, nargs="+", default=[0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
    p.add_argument("--rev", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--min-iters", type=int, default=2, help="steps that minimise |u|inf (then Newton steps)")
    p.add_argument("--yaw-max-deg", type=float, default=10.0)
    p.add_argument("--tilt-max-deg", type=float, default=25.0)
    p.add_argument("--n-cradle", type=int, default=24)
    p.add_argument("--n-avg", type=int, default=8)
    p.add_argument("--trust", type=float, default=0.02)
    p.add_argument("--ang-trust-deg", type=float, default=5.0)
    p.add_argument("--f32", action="store_true")
    p = sub.add_parser("torque", help="crank actuator cost on CPU: peak (every step / smooth), RMS, power vs hover")
    p.add_argument("--run", type=Path, default=None, help="a learn run (policy.npz; traces.npz for --eval-flights)")
    p.add_argument("--policy", type=Path, default=None, help="a policy npz instead of --run (e.g. the exported "
                   "designs/hummer/free-hover/command_policy.npz: its envelope cells are added)")
    p.add_argument("--cells", nargs="*", default=None, help="extra constant-command cells CMD:SPEED (e.g. FWD:0.7 REV:0.75)")
    p.add_argument("--beats", type=int, default=120)
    p.add_argument("--window-from", type=int, default=30, help="cells: statistics over beats >= this")
    p.add_argument("--skip", type=int, default=5, help="steps at each wingbeat start left out of the smooth peak")
    p.add_argument("--eval-flights", type=int, default=8, help="with --run: this many plain V_op eval flights (traces.npz)")
    p.add_argument("--eval-beats", type=int, default=240)
    p.add_argument("--flag-ratio", type=float, default=1.5)
    p.add_argument("--limit", type=float, default=None, help="also fly every flight with the cranks forcelimited to +- this [N m]")
    p.add_argument("--procs", type=int, default=16)
    p.add_argument("--out", type=Path, default=None)
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
    p.add_argument("--p-gain", default="auto", choices=("auto", "hover", "climb"),
                   help="P servo gain: the --p-q-int DARE on the "
                   "policy's hover model or on its regressed climb model (auto: climb when the policy has one; p_servo_gain)")
    p.add_argument("--p-climb-after-stop", type=float, default=0.05)
    p.add_argument("--pos-latency-ms", type=float, default=5.0)
    a = ap.parse_args()
    if PRESETS.get(getattr(a, "preset", None)):      # re-parse with the preset as the defaults (explicit flags still win)
        learn_parsers[a.cmd].set_defaults(**PRESETS[a.preset])
        a = ap.parse_args()
    if getattr(a, "v_op_up", None) is not None:      # --v-op-up: the UP speed (slot 1 of v_op, the vertical rate)
        if a.v_op_down is None:
            a.v_op_down = a.v_op_z
        a.v_op_z = a.v_op_up
    {"learn": cmd_learn, "dt": cmd_dt, "sensors": cmd_sensors, "cpu": cmd_cpu, "all": cmd_all, "export": cmd_export,
     "trims": cmd_trims, "torque": cmd_torque}[a.cmd](a)


if __name__ == "__main__":
    main()
