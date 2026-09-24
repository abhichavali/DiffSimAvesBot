"""Flat-plate symmetry of the strip aero, CPU and MJX: a symmetric plate does not care
which edge leads, so mirroring the chordwise flow (vz -> -vz) mirrors the force."""

import jax.numpy as jnp
import numpy as np
import pytest

from runtime.aero import RHO_AIR, aerodynamics
from runtime.mjx.aero import aero_wing

CHORD, LENGTH, DT = 0.05, 0.2, 5e-4
MIRROR = np.array([1.0, 1.0, -1.0])


def cpu_force(side, v):
    zero = np.zeros(3)
    return aerodynamics(side, zero, zero, v, zero, 0.5, DT, LENGTH, 1, RHO_AIR, CHORD)[0]


def mjx_force(side, v):
    zero, v = jnp.zeros(3), jnp.asarray(v)
    f = aero_wing(zero, zero, v, zero, v, side=side, chords=jnp.array([CHORD]),
                  span_cutoffs=jnp.array([LENGTH]), length=LENGTH, num_blade=1, dt=DT)[0]
    return np.asarray(f[0], dtype=float)


@pytest.mark.parametrize("side", "LR")
@pytest.mark.parametrize("alpha_deg", [20.0, 45.0, 70.0])
@pytest.mark.parametrize("vx_sign", [1.0, -1.0])
def test_reversed_chordwise_flow_mirrors_force(side, alpha_deg, vx_sign):
    a = np.radians(alpha_deg)
    v = 3.0 * np.array([vx_sign * np.sin(a), 0.0, np.cos(a)])
    for force in (cpu_force, mjx_force):
        np.testing.assert_allclose(force(side, MIRROR * v), MIRROR * force(side, v), rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(mjx_force(side, v), cpu_force(side, v), rtol=1e-5, atol=1e-9)
