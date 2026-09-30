"""Command-following servo (hover.command, hover.mjx_command) against the hover code it builds on.

Host parts (LP tables, servo gains, the CPU policy) run in-process; the device parts run in a CPU-JAX x64
subprocess (2 wingbeats) and must match hover.mjx_flight.fly / hover.mjx_sensors to float64 round-off.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hover import command as C

ROOT = Path(__file__).resolve().parents[1]
LQR = ROOT / "designs" / "hummer" / "free-hover" / "lqr.npz"
P_CTL = ROOT / "opt" / "runs_imu" / "noise" / "ctl" / "P_lqgV1.npz"
Q_REF = np.array([0.982637124354, -0.0168423093447, 0.184771074478, -0.000518163825496])


def _model():
    z = np.load(LQR)
    return z["A"], z["B"], z["drift"], z["K"]


def test_tables_solve_the_steady_state_lp():
    A, B, drift, _ = _model()
    e_f = C.forward_axis(Q_REF)
    assert e_f[0] > 0.99 and e_f[2] == 0
    tab = C.steady_state_tables(A, B, drift, e_f, 5.0, log=None)
    assert tab["feasible"].all()
    assert np.nanmax(tab["residual"]) <= 1e-9
    assert np.abs(tab["u"][0, 0]).max() <= 0.02
    # every entry holds the att / vel / w rows of x_{k+1} = x_k at its velocity
    dirs = C.command_dirs(e_f)[1:]
    for a in range(4):
        for j in (3, 9):
            x = np.zeros(12); x[3:6] = tab["att"][a, j]; x[6:9] = tab["speeds"][a, j] * dirs[a]
            r = (A @ x + B @ tab["u"][a, j] + drift - x)[3:12]
            assert np.abs(r).max() <= 1e-9
    assert abs(np.degrees(tab["att"][:, :, :2]).max()) <= 5 + 1e-9
    # superposition lookup reproduces a table entry on its own axis
    att, vel, w, u = C.lookup(tab, e_f, np.array([tab["speeds"][0, 6] * e_f]))
    assert np.allclose(u[0], tab["u"][0, 6]) and np.allclose(att[0], tab["att"][0, 6]) and np.allclose(vel[0], tab["speeds"][0, 6] * e_f)


def test_servo_gains_are_stabilizing():
    A, B, _, _ = _model()
    Ka, rho = C.servo_gains(A, B)
    assert Ka.shape == (24, 15) and rho < 0.85
    Kv, rho_v = C.servo_gains_vel(A, B, 0.11707)
    assert rho_v < 1 and np.all(Kv[:, :3] == 0)


def test_linear_model_follows_random_commands():
    A, B, drift, _ = _model()
    e_f = C.forward_axis(Q_REF)
    tab = C.steady_state_tables(A, B, drift, e_f, 5.0, log=None)
    Ka, _ = C.servo_gains(A, B)
    sch = C.random_schedules(np.random.default_rng(0), 20, 120, e_f)
    xr, uf, _ = C.build_refs(sch["v_cmd"], tab, e_f, 0.11707, np.full(3, 0.3 / 4))
    S, U = C.linear_sim(A, B, drift, Ka, xr, uf)
    segs, fl = C.segment_metrics(S, U, np.ones(S.shape[:2], bool), sch, xr, 0.11707, e_f=e_f)
    sm = C.summarize(segs, fl)
    assert sm["follow_rate"] == 1.0 and sm["along_abs_p95"] < 0.01


def test_allocation_is_the_box_constrained_q_function_minimum():
    A, B, _, _ = _model()
    H = C.servo_hessian(A, B)
    assert np.allclose(H, H.T) and abs(np.linalg.eigvalsh(H).max() - 1) < 1e-12
    rng = np.random.default_rng(0)
    for _ in range(20):
        ud = rng.normal(0, 0.15, 24)
        u = C.allocate(ud, H, 0.15, 200)
        g = H @ (u - ud)                               # KKT: free components have zero gradient, bound ones push out
        free = np.abs(u) < 0.15 - 1e-9
        assert np.abs(g[free]).max(initial=0) < 1e-7
        assert np.all(g[~free] * np.sign(u[~free]) <= 1e-7)
        assert (u - ud) @ H @ (u - ud) <= (np.clip(ud, -0.15, 0.15) - ud) @ H @ (np.clip(ud, -0.15, 0.15) - ud) + 1e-12
    # the identity metric allocates exactly as clip (the baseline arm); inside the box nothing changes
    ud = rng.normal(0, 0.15, 24)
    assert np.array_equal(C.allocate(ud, np.eye(24), 0.15, 30), np.clip(ud, -0.15, 0.15))
    assert np.array_equal(C.allocate(0.1 * ud / np.abs(ud).max(), H, 0.15, 30), 0.1 * ud / np.abs(ud).max())


def test_command_policy_zero_command_is_the_plain_lqr():
    from design_paths import FREE_HOVER_TG
    from hover import launch as L
    from hover.policy import load_flight_policy
    ctl = FREE_HOVER_TG.parent / "control.tg"
    fp = load_flight_policy(ctl)
    bird = L.load_bird(FREE_HOVER_TG, control=ctl)
    rel = L.release(bird, fp.theta, fp.attitude, fp.K, k_phi=fp.phase_lock, n_cradle=3, n_avg=1)
    K = np.load(LQR)["K"]
    kick = np.r_[0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.5, 0]
    ref = L.fly(bird, rel, fp.theta, 2, policy=lambda s: np.concatenate([[0.0], np.clip(-K @ s, -0.15, 0.15)]), kicks={1: kick})
    pol = C.CommandPolicy(C.hover_ka(K), np.zeros((3, 12)), np.zeros((3, 24)))
    out = L.fly(bird, rel, fp.theta, 2, policy=pol, kicks={1: kick})
    assert np.abs(out - ref).max() <= 1e-12


SCRIPT = r"""
import json, sys
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pathlib import Path
from design_paths import FREE_HOVER_TG
from hover import launch as L, mjx_flight as F, mjx_command as MC, command as C
from hover.policy import load_flight_policy
CT = FREE_HOVER_TG.parent / "control.tg"
fp = load_flight_policy(CT)
bird = L.load_bird(FREE_HOVER_TG, control=CT)
rel = L.release(bird, fp.theta, fp.attitude, fp.K, k_phi=fp.phase_lock, n_cradle=3, n_avg=1)
K = np.load(FREE_HOVER_TG.parent / "lqr.npz")["K"]
kick = np.r_[0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.5, 0]
kicks = np.stack([np.zeros(12), kick])[None]
fl = F.Flight(bird, rel, fp.theta, fp.K, k_phi=fp.phase_lock)
f = F.batch_flyer(fl, 2, "linear")
S0, U0, A0 = f(fl.arrays, jnp.asarray(kicks), jnp.asarray(K[None]), jnp.zeros((1, 12)))
S0, U0 = np.asarray(S0[0]), np.asarray(U0[0])
Ka = C.hover_ka(K)[None]
z = lambda *s: np.zeros((1,) + s)
out = {"scale": float(np.abs(S0).max())}
st, c_rest = MC.release_carry(fl, 1)
ch = MC.Chain(fl, 1, 2)
S2, U2, A2, _, _ = ch.run(st, c_rest, kicks, z(2, 12), z(2, 24), z(2, 24), Ka, 1)
out["fly_cmd_vs_fly"] = float(np.abs(S2[0] - S0).max())
out["fly_cmd_u"] = float(np.abs(U2[0] - U0).max())
out["alive"] = bool(np.asarray(A2).all())
st, c_rest = MC.release_carry(fl, 1)
ch1 = MC.Chain(fl, 1, 1)
S11, U11, A11, _, _ = ch1.run(st, c_rest, kicks, z(2, 12), z(2, 24), z(2, 24), Ka, 2)
out["chain_1p1_vs_2"] = float(np.abs(S11 - S2).max())
# the allocating servo (saturating feedforward) on the device vs CommandPolicy on CPU (hover.launch.fly)
H = C.servo_hessian(np.load(FREE_HOVER_TG.parent / "lqr.npz")["A"], np.load(FREE_HOVER_TG.parent / "lqr.npz")["B"])
uff = np.zeros((1, 2, 24)); uff[0, :, 3] = 0.2; uff[0, :, 10] = -0.18
xr = np.zeros((1, 2, 12)); xr[0, :, 7] = 0.2
cha = MC.Chain(fl, 1, 2, H=H, alloc_iters=30)
st, c_rest = MC.release_carry(fl, 1)
Sa, Ua, Aa, _, _ = cha.run(st, c_rest, kicks, xr, uff, z(2, 24), Ka, 1)
pol = C.CommandPolicy(Ka[0], np.concatenate([xr[0], xr[0, -1:]]), np.concatenate([uff[0], uff[0, -1:]]), H=H, iters=30)
ref = L.fly(bird, rel, fp.theta, 2, policy=pol, kicks={1: kick})
out["alloc_cpu_vs_dev"] = float(np.abs(Sa[0] - ref).max())
out["alloc_u"] = float(np.abs(Ua[0] - np.array(pol.U)).max())
out["alloc_saturated"] = bool((np.abs(Ua) >= 0.15 - 1e-12).any())
# integral handoff: a 13-wide x_ref with keep = 1 flies exactly as the 12-wide one; the device law == servo_law
Ka_i, _ = C.servo_gains(np.load(FREE_HOVER_TG.parent / "lqr.npz")["A"], np.load(FREE_HOVER_TG.parent / "lqr.npz")["B"])
xr13 = np.concatenate([xr, np.ones((1, 2, 1))], -1)
outs = []
for x_ in (xr, xr13):
    st, c_rest = MC.release_carry(fl, 1)
    outs.append(MC.Chain(fl, 1, 2).run(st, c_rest, kicks, x_, uff, z(2, 24), Ka_i[None], 1)[0])
out["keep1_13_vs_12"] = float(np.abs(outs[0] - outs[1]).max())
rng = np.random.default_rng(1)
dl = 0.0
for _ in range(10):
    s_, xr_, uf_ = rng.normal(0, 0.05, 12), np.r_[rng.normal(0, 0.05, 12), rng.uniform(size=2)], rng.normal(0, 0.05, 24)
    zt = rng.normal(0, 0.3, 3)
    u_h, z_h = C.servo_law(s_, zt, xr_, uf_, 0.0, Ka_i)
    u_d, z_d = MC.law(jnp.asarray(s_), jnp.asarray(zt), jnp.asarray(xr_), jnp.asarray(uf_), 0.0, jnp.asarray(Ka_i))
    dl = max(dl, float(np.abs(np.asarray(u_d) - u_h).max()), float(np.abs(np.asarray(z_d) - z_h).max()))
    assert abs(xr_[12]) > 0
out["law_keep_dev_vs_host"] = dl
k = C.integral_keep(np.array([[[0.3, 0, 0], [0.3, 0, 0], [0.15, 0, 0], [0, 0, 0], [0, 0, 0.1]]]))
out["keep_decel"] = k[0].tolist()
vc = np.zeros((1, 12, 3)); vc[0, :2, 0] = 0.3
vr = np.zeros((1, 13, 3)); vr[0, 1:3, 0] = 0.3; vr[0, 3, 0] = 0.15
out["gate"] = C.integral_gate(vr, vc, 2)[0].tolist()
p_ctl = Path(sys.argv[1])
if p_ctl.exists():
    from hover import sensors as Sn, reduced as Rd, mjx_sensors as MS
    ctl = Rd.load_controller(p_ctl)
    sp = MS.SensorPipeline(fl, Sn.SensorNoise.zero(), att_mode="filter", ctl=ctl)
    g = MS.batch_sensor_flyer(fl, sp, 2)
    key = jax.random.split(jax.random.PRNGKey(0), 1)
    Sr, Yr, Ur, Ar = g(fl.arrays, jnp.asarray(kicks), key, jnp.zeros(1), jax.tree_util.tree_map(lambda x: x[None], sp.params().ctl))
    csp = MC.CmdSensorPipeline(fl, Sn.SensorNoise.zero(), 30, att_mode="filter")
    pc = MC.pad_observer(ctl, 30, 24)
    pc.update(Ka=C.hover_ka(ctl["K"]), x_ref=np.zeros((2, 12)), u_ff=np.zeros((2, 24)), cp=0.0, cv=0.0, T=0.0,
              cz_seq=np.tile(pc["cz"], (2, 1)))          # the per-beat offset path, constant = the plain observer
    h = MC.batch_cmd_sensor_flyer(fl, csp, 2)
    Sc, Yc, Xc, Uc, Ac = h(fl.arrays, jnp.asarray(kicks), key, jnp.zeros(1), {k: jnp.asarray(v)[None] for k, v in pc.items()})
    out["sensor_S"] = float(np.abs(np.asarray(Sc) - np.asarray(Sr)).max())
    out["sensor_U"] = float(np.abs(np.asarray(Uc) - np.asarray(Ur)).max())
print(json.dumps(out))
"""


def test_trim_consistent_observer_offset():
    """The reference is the observer model's own trajectory under u_ff; a constant offset = the plain observer."""
    if not P_CTL.exists():
        pytest.skip("no P controller")
    from hover import reduced as Rd
    ctl = Rd.load_controller(P_CTL)
    rng = np.random.default_rng(3)
    x_ref, u_ff = rng.normal(0, 0.05, (6, 14)), rng.normal(0, 0.05, (6, 24))
    cz = C.trim_consistent_cz(ctl, x_ref, u_ff)
    xs, Az, Bz = np.asarray(ctl["xs"], int), ctl["Az"], ctl["Bz"]
    n, nz = len(xs), Az.shape[0]
    assert cz.shape == (6, nz) and np.allclose(cz[0], ctl["cz"]) and np.all(cz[1:, n:] == 0)
    for k in range(1, 6):
        z = np.zeros(nz); z[:n] = x_ref[k - 1, xs]
        assert np.allclose((Az @ z + Bz @ u_ff[k - 1] + cz[k])[:n], x_ref[k, xs], atol=1e-12)
    Ka = C.hover_ka(ctl["K"])
    ys = rng.normal(0, 0.02, (5, 18))
    a = C.CmdLinearController(ctl, Ka, np.zeros((5, 12)), np.zeros((5, 24)), cp=0.0)
    b = C.CmdLinearController(ctl, Ka, np.zeros((5, 12)), np.zeros((5, 24)), cp=0.0, cz_seq=np.tile(ctl["cz"], (5, 1)))
    assert all(np.array_equal(a(y), b(y)) for y in ys)


def test_climb_after_stop_holds_the_climb_until_the_stop():
    A, B, drift, _ = _model()
    e_f = C.forward_axis(Q_REF)
    tab = C.steady_state_tables(A, B, drift, e_f, 5.0, log=None)
    vc = np.zeros((1, 40, 3)); vc[0, :20] = 0.3 * e_f; vc[0, 20:, 2] = 0.3
    rate = np.full(3, 0.3 / 8)
    ref = [C.build_refs(vc, tab, e_f, 0.117, rate, 0.05, 8, climb_after_stop=c) for c in (0.0, 0.05)]
    assert np.array_equal(ref[0][0], C.build_refs(vc, tab, e_f, 0.117, rate, 0.05, 8)[0])
    v0, v1 = ref[0][2][0], ref[1][2][0]
    assert v0[21, 2] > 0 and np.all(v1[:27, 2] == 0) and v1[29, 2] > 0 and np.isclose(v1[-1, 2], 0.3)


def test_device_servo_matches_mjx_flight_in_x64():
    """fly_cmd (zero command, Ka = [K, 0]) == mjx_flight.fly(linear K); a 1 + 1 chain == one 2-beat call; the
    padded sensor servo (zero noise, zero command) == mjx_sensors with the P controller. CPU-JAX x64, 2 beats."""
    pytest.importorskip("jax")
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(P_CTL)], capture_output=True, text=True, cwd=ROOT,
                       env={"PYTHONPATH": str(ROOT / "src"), "JAX_PLATFORMS": "cpu", "PATH": ""}, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    print(out)
    assert out["alive"] and out["scale"] > 1e-3
    assert out["fly_cmd_vs_fly"] < 1e-9 and out["fly_cmd_u"] < 1e-9, out
    assert out["chain_1p1_vs_2"] < 1e-9, out
    assert out["alloc_saturated"] and out["alloc_cpu_vs_dev"] < 1e-9 and out["alloc_u"] < 1e-9, out
    assert out["keep1_13_vs_12"] < 1e-12 and out["law_keep_dev_vs_host"] < 1e-12, out
    assert np.allclose(out["keep_decel"], [1.0, 0.5, 0.0, 1.0, 1.0]), out
    # commands FWD FWD then HOVER; v ramps 0 -> 0.3 (k 0), holds, decelerates at k 2, 3: gate 0 from k 2 to 3 + 2
    assert out["gate"] == [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], out
    if "sensor_S" in out:
        assert out["sensor_S"] < 1e-9 and out["sensor_U"] < 1e-9, out


COMMAND_TG = ROOT / "designs" / "hummer" / "free-hover" / "command.tg"


def _live_setup():
    from hover.policy import load_flight_policy
    fp = load_flight_policy(COMMAND_TG)
    pol = fp.command
    rng = np.random.default_rng(11)
    sch = C.random_schedules(rng, 6, 90, pol["e_f"], seg=(5, 12), speeds=tuple(float(x) for x in pol["v_op"]))
    g = int(pol.get("zeta_gate", -1))
    kw = dict(zeta_handoff=bool(pol.get("zeta_handoff", False)), zeta_gate=None if g < 0 else g,
              stop_rate=float(pol.get("stop_rate", 1.0)), ff_lead=int(pol.get("ff_lead", 0)))
    rate = np.array([pol["v_op"][0], pol["v_op"][0], pol["v_op"][1]]) / float(pol["rate_beats"])
    return fp, pol, sch, kw, rate


def test_live_reference_stream_equals_build_refs():
    """RefStream (a command known one wingbeat at a time) == build_refs on the whole sequence, with every option
    the exported policy flies (handoff, gate, ff_lead 1, descend / climb shaping)."""
    fp, pol, sch, kw, rate = _live_setup()
    assert kw["ff_lead"] <= 1 and kw["zeta_gate"] is not None
    for cas in (0.0, 0.05):
        x_ref, u_ff, _ = C.build_refs(sch["v_cmd"], pol["tab"], pol["e_f"], float(pol["T"]), rate,
                                      float(pol["descend_after_stop"]), int(pol["descend_delay"]), **kw, climb_after_stop=cas)
        for i in range(len(x_ref)):
            rs = C.RefStream(pol["tab"], pol["e_f"], float(pol["T"]), rate, descend_after_stop=float(pol["descend_after_stop"]),
                             descend_delay=int(pol["descend_delay"]), **kw, climb_after_stop=cas)
            for k in range(sch["v_cmd"].shape[1]):
                xr, uf = rs.next(sch["v_cmd"][i, k])
                assert np.allclose(xr, x_ref[i, k], atol=1e-12) and np.allclose(uf, u_ff[i, k], atol=1e-12), (cas, i, k)


def test_live_command_equals_the_evaluated_controllers():
    """LiveCommand with commands set per wingbeat flies exactly CommandPolicy (true state) and CmdLinearController
    + trim_consistent_cz (sensor observer) on the precomputed references of the same command sequence."""
    fp, pol, sch, kw, rate = _live_setup()
    names = np.array(C.CMDS)
    rng = np.random.default_rng(5)
    for sensor in (False, True):
        obs = fp.observer if sensor else None
        x_ref, u_ff, _ = C.build_refs(sch["v_cmd"][:2], pol["tab"], pol["e_f"], float(pol["T"]), rate,
                                      float(pol["descend_after_stop"]), int(pol["descend_delay"]), **kw,
                                      climb_after_stop=float(pol["sensor_climb_after_stop"]) if sensor else 0.0)
        for i in range(2):
            if sensor:
                ref = C.CmdLinearController(obs, pol["sensor_Ka"], x_ref[i], u_ff[i], u_max=float(pol["u_max"]),
                                            zeta_max=float(pol["zeta_max"]), cz_seq=C.trim_consistent_cz(obs, x_ref[i], u_ff[i]))
            else:
                ref = C.CommandPolicy(pol["Ka"], x_ref[i], u_ff[i], u_max=float(pol["u_max"]), zeta_max=float(pol["zeta_max"]))
            live = C.LiveCommand(pol, obs)
            for k in range(sch["v_cmd"].shape[1]):
                live.set_command(str(names[sch["cmd"][i, k]]))
                assert np.allclose(live.v_cmd, sch["v_cmd"][i, k])
                y = rng.normal(0, 0.02, 18 if sensor else 12)
                assert np.allclose(live(y), ref(y), atol=1e-12), (sensor, i, k)
