"""Flat-plate symmetry of the strip aero, CPU and MJX: a symmetric plate does not care
which edge leads, so mirroring the chordwise flow (vz -> -vz) mirrors the force, moves
the CoP to its mirror point about mid-chord, and moves the rotational lift's 3/4-chord
point to 1/4 chord from the leading edge. The strip geometry is the compiled wing mesh's."""

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
import pytest

from morphology.compile import ROOT, compile_file
from morphology.tg_parser import load_morphology
from morphology.wing_geometry import _section_values, load_wing_geometry
from runtime.aero import _SIDE_SIGNS, AERO_PARAMS, RHO_AIR, AeroBladeRuntime, aerodynamics
from runtime.mjx.aero import aero_wing
from runtime.mjx.aero_runtime import aero_wrench, build_aero_config, init_aero_state

CHORD, LENGTH, DT = 0.05, 0.2, 5e-4
X_ROT, ROOT_OFFSET = 0.2, 0.007  # hinge 0.2 chord behind the leading edge; wing starts 7 mm out
MIRROR = np.array([1.0, 1.0, -1.0])
# A pitch rate adds the rotational drag (all a strip feels on still air, and the same
# whichever edge leads) and the rotational lift. The strip velocity is v either way
# (omega_y x the spanwise lever = 0).


def cpu_strip(side, v, omega=(0.0, 0.0, 0.0)):
    zero = np.zeros(3)
    return aerodynamics(side, np.array(omega), zero, v, zero, 0.5, DT, LENGTH, 1,
                        RHO_AIR, CHORD, x_rot=X_ROT, root=ROOT_OFFSET)


def mjx_strip(side, v, omega=(0.0, 0.0, 0.0)):
    zero, v = jnp.zeros(3), jnp.asarray(v)
    f, t = aero_wing(jnp.array(omega), zero, v, zero, v, side=side,
                     chords=jnp.array([CHORD]), x_rot=jnp.array([X_ROT]),
                     length=LENGTH, root=ROOT_OFFSET, num_blade=1, dt=DT)[:2]
    return np.asarray(f[0], dtype=float), np.asarray(t[0], dtype=float)


@pytest.mark.parametrize("side", "LR")
@pytest.mark.parametrize("alpha_deg", [20.0, 45.0, 70.0])
@pytest.mark.parametrize("vx_sign", [1.0, -1.0])
def test_reversed_chordwise_flow_mirrors_the_plate(side, alpha_deg, vx_sign):
    a = np.radians(alpha_deg)
    v = 3.0 * np.array([vx_sign * np.sin(a), 0.0, np.cos(a)])  # vz > 0: the far edge leads on L
    pitch = (0.0, 5.0, 0.0)
    for strip in (cpu_strip, mjx_strip):
        (fa, ta), (fb, tb) = strip(side, v), strip(side, MIRROR * v)
        np.testing.assert_allclose(fb, MIRROR * fa, rtol=1e-5, atol=1e-9)
        # tau_y = z_cop * F_x; the plate spans z in [-X_ROT, 1 - X_ROT] chords on L, mirrored on R.
        mid = (1.0 - 2.0 * X_ROT) * CHORD
        assert ta[1] / fa[0] + tb[1] / fb[0] == pytest.approx(mid if side == "L" else -mid, rel=1e-4)
        drag = strip(side, np.zeros(3), pitch)[0][0]
        rot_a = strip(side, v, pitch)[0][0] - fa[0] - drag
        rot_b = strip(side, MIRROR * v, pitch)[0][0] - fb[0] - drag
        hinge, far = (rot_b, rot_a) if side == "L" else (rot_a, rot_b)
        assert hinge == pytest.approx((0.75 - X_ROT) / (0.25 - X_ROT) * far, rel=1e-4)
    # A general rotation puts the root offset in the strip velocity too.
    for omega in (pitch, (4.0, 5.0, -3.0)):
        for x, y in zip(mjx_strip(side, v, omega), cpu_strip(side, v, omega)):
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-9)


def test_aero_strips_sit_on_the_compiled_wing_mesh():
    """Blade sites, chords and hinge offsets match MuJoCo's own posed wing mesh."""
    tg = ROOT / "designs" / "hummer" / "hover" / "shoulder.tg"
    g = load_wing_geometry(load_morphology(tg))
    model = mujoco.MjModel.from_xml_string(compile_file(tg))
    n = len(g.chords)
    for side, chordwise in (("L", 1.0), ("R", -1.0)):  # hinge-side edge -> far edge is +y on L, -y on R
        body = model.body(f"{side}_wing").id
        gid = next(i for i in range(model.ngeom)
                   if model.geom_bodyid[i] == body and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH)
        mesh = model.geom_dataid[gid]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
        va, fa = model.mesh_vertadr[mesh], model.mesh_faceadr[mesh]
        verts = model.mesh_vert[va:va + model.mesh_vertnum[mesh]] @ rot.reshape(3, 3).T + model.geom_pos[gid]
        tris = verts[model.mesh_face[fa:fa + model.mesh_facenum[mesh]]]
        assert verts[:, 2].min() == pytest.approx(g.root_offset, abs=1e-6)
        assert verts[:, 2].max() == pytest.approx(g.root_offset + g.length, abs=1e-6)
        for j in range(n):
            z = model.site(f"{side}_wing_aero_blade_{j + 1:02d}").pos[2]
            assert z == pytest.approx(g.root_offset + (j + 0.5) * g.length / n, abs=1e-9)
            s = chordwise * _section_values(tris, 2, z, 1)
            assert s.max() - s.min() == pytest.approx(g.chords[j], abs=1e-6)
            assert s.min() == pytest.approx(-g.x_rot[j] * g.chords[j], abs=1e-6)


def test_uniform_flow_loads_every_strip_alike():
    """Uniform flow plus a pitch rate (omega_y adds no strip velocity): every strip of a
    uniform-chord wing sees the same flow, so gets the same force, rotational lift included."""
    v, omega, zero = np.array([1.0, 0.0, 2.0]), np.array([0.0, 5.0, 0.0]), np.zeros(3)
    f_in, f_out = (aerodynamics("L", omega, zero, v, zero, i, DT, LENGTH, 2, RHO_AIR, CHORD,
                                x_rot=X_ROT, root=ROOT_OFFSET)[0] for i in (0.25, 0.75))
    np.testing.assert_allclose(f_in, f_out, rtol=1e-12)
    f = aero_wing(jnp.asarray(omega), jnp.zeros(3), jnp.asarray(v), jnp.zeros(3), jnp.asarray(v), side="L",
                  chords=jnp.full(2, CHORD), x_rot=jnp.full(2, X_ROT),
                  length=LENGTH, root=ROOT_OFFSET, num_blade=2, dt=DT)[0]
    np.testing.assert_allclose(f[0], f[1], rtol=1e-6)
    np.testing.assert_allclose(f[0], f_in, rtol=1e-5)


@pytest.mark.parametrize("side", "LR")
def test_rotational_drag_is_the_chordwise_integral(side):
    """A pitch rate on still air loads a strip with the rotational drag alone: the integral
    over the chord of Crd*rho/2*w|w| * u|u| (force) and u^2|u| (moment), u the distance ahead
    of the hinge. (It was a span-gated right-endpoint sum over NumBlade elements.)"""
    w = 5.0
    u = CHORD * (X_ROT - (np.arange(10_000) + 0.5) / 10_000)  # midpoint rule, leading -> far edge
    k0 = 0.5 * RHO_AIR * w * abs(w) * LENGTH * AERO_PARAMS.Crd * CHORD / u.size
    for strip in (cpu_strip, mjx_strip):
        f, t = strip(side, np.zeros(3), (0.0, w, 0.0))
        assert f[0] == pytest.approx(_SIDE_SIGNS[side][0] * k0 * np.sum(u * np.abs(u)), rel=1e-5)
        assert t[1] == pytest.approx(-k0 * np.sum(u**2 * np.abs(u)), rel=1e-5)


def hover_moving():
    """The hover design's morphology, model and an MjData at rest with a brisk qvel."""
    tg = ROOT / "designs" / "hummer" / "hover" / "shoulder.tg"
    model = mujoco.MjModel.from_xml_string(compile_file(tg))
    data = mujoco.MjData(model)
    data.qvel[:] = 30.0 * np.sin(np.arange(model.nv) + 1.0)
    return load_morphology(tg), model, data


def test_aero_reads_the_pose_it_moved_to():
    """Straight out of mj_step the kinematics still hold the pre-step pose; the aero must
    see the stepped pose, i.e. act as if mj_forward had run (CPU and MJX)."""
    morph, model, stepped = hover_moving()
    mujoco.mj_step(model, stepped)
    forwarded = mujoco.MjData(model)
    forwarded.qpos[:], forwarded.qvel[:] = stepped.qpos, stepped.qvel
    mujoco.mj_forward(model, forwarded)

    mx, cfg = mjx.put_model(model), build_aero_config(model, morph)
    f = jax.jit(lambda d: aero_wrench(mx, d, cfg, init_aero_state(cfg))[0])
    a, b = (np.asarray(f(mjx.put_data(model, d))) for d in (stepped, forwarded))
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6 * np.abs(b).max())

    def cpu(data):  # after the MJX half: the CPU runtime refreshes ``data`` in place
        data.qfrc_applied[:] = 0.0
        AeroBladeRuntime(model, morph, substep=1).apply(model, data)
        return data.qfrc_applied.copy()

    np.testing.assert_array_equal(cpu(stepped), cpu(forwarded))


def test_aero_hold_keeps_the_generalized_force():
    """Between recomputes the CPU runtime adds the last recompute's generalized force
    unchanged, as the MJX rollout's hold does, however far the wing has moved."""
    morph, model, data = hover_moving()
    rt = AeroBladeRuntime(model, morph, substep=4)
    qfrc = []
    for _ in range(5):
        data.qfrc_applied[:] = 0.0
        rt.apply(model, data)
        qfrc.append(data.qfrc_applied.copy())
        mujoco.mj_step(model, data)
    for held in qfrc[1:4]:
        np.testing.assert_array_equal(held, qfrc[0])
    assert np.abs(qfrc[4] - qfrc[0]).max() > 1e-3 * np.abs(qfrc[0]).max()  # step 4 recomputes
