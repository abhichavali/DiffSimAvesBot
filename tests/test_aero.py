"""Flat-plate symmetry of the strip aero, CPU and MJX: a symmetric plate does not care
which edge leads, so mirroring the chordwise flow (vz -> -vz) mirrors the force, moves
the CoP to its mirror point about mid-chord, and moves the rotational lift's 3/4-chord
point from 0.75 to 0.25 chord behind the hinge."""

import jax.numpy as jnp
import numpy as np
import pytest

from runtime.aero import RHO_AIR, aerodynamics
from runtime.mjx.aero import aero_wing

CHORD, LENGTH, DT = 0.05, 0.2, 5e-4
MIRROR = np.array([1.0, 1.0, -1.0])
# A zero span cutoff switches the rotational drag off, so the pitch rate only adds the
# rotational lift. The strip velocity is v either way (omega_y x the spanwise lever = 0).


def cpu_strip(side, v, omega_y=0.0):
    zero = np.zeros(3)
    return aerodynamics(side, np.array([0.0, omega_y, 0.0]), zero, v, zero, 0.5, DT, LENGTH, 1,
                        RHO_AIR, CHORD, np.zeros(1))


def mjx_strip(side, v, omega_y=0.0):
    zero, v = jnp.zeros(3), jnp.asarray(v)
    f, t = aero_wing(jnp.array([0.0, omega_y, 0.0]), zero, v, zero, v, side=side,
                     chords=jnp.array([CHORD]), span_cutoffs=jnp.zeros(1), length=LENGTH,
                     num_blade=1, dt=DT)[:2]
    return np.asarray(f[0], dtype=float), np.asarray(t[0], dtype=float)


@pytest.mark.parametrize("side", "LR")
@pytest.mark.parametrize("alpha_deg", [20.0, 45.0, 70.0])
@pytest.mark.parametrize("vx_sign", [1.0, -1.0])
def test_reversed_chordwise_flow_mirrors_the_plate(side, alpha_deg, vx_sign):
    a = np.radians(alpha_deg)
    v = 3.0 * np.array([vx_sign * np.sin(a), 0.0, np.cos(a)])  # vz > 0: the far edge leads on L
    for strip in (cpu_strip, mjx_strip):
        (fa, ta), (fb, tb) = strip(side, v), strip(side, MIRROR * v)
        np.testing.assert_allclose(fb, MIRROR * fa, rtol=1e-5, atol=1e-9)
        # tau_y = z_cop * F_x; the plate spans z in [0, chord] on L and [-chord, 0] on R.
        assert ta[1] / fa[0] + tb[1] / fb[0] == pytest.approx(CHORD if side == "L" else -CHORD, rel=1e-4)
        rot_a = strip(side, v, 5.0)[0][0] - fa[0]
        rot_b = strip(side, MIRROR * v, 5.0)[0][0] - fb[0]
        hinge, far = (rot_b, rot_a) if side == "L" else (rot_a, rot_b)
        assert hinge == pytest.approx(3.0 * far, rel=1e-4)
    for x, y in zip(mjx_strip(side, v, 5.0), cpu_strip(side, v, 5.0)):
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-9)
