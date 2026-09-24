"""The packed-theta layout: group order, sizes, and how a theta splits and seeds."""
import jax.numpy as jnp
import numpy as np

from codesign.genome import Genome, initial_theta

FULL = Genome(K=2, n_tend=3, control=True, frequency=True, tendons=True, pitch=True, motor=True,
              strokeplane=True)


def test_layout_order_and_sizes():
    s = FULL.slices()
    assert list(s) == ["control", "f_log", "stiff", "pre", "pitch", "motor", "sp"]
    assert [(v.start, v.stop) for v in s.values()] == [
        (0, 5), (5, 6), (6, 9), (9, 12), (12, 16), (16, 18), (18, 28)]
    assert FULL.size == 28
    assert Genome(K=2, n_tend=3, control=False, motor=True).slices() == {"motor": slice(0, 2)}


def test_split_and_frozen_control():
    theta = np.arange(FULL.size, dtype=np.float32)
    g = FULL.split(theta, control_fixed=None)
    assert g.control.tolist() == [0, 1, 2, 3, 4] and g.f_log == 5.0
    assert g.stiff.tolist() == [6, 7, 8] and g.pre.tolist() == [9, 10, 11]
    assert g.pitch.tolist() == [12, 13, 14, 15] and g.motor.tolist() == [16, 17]
    assert g.sp.tolist() == list(range(18, 28))
    frozen = Genome(K=2, n_tend=3, control=False, tendons=True)
    seed = np.full(5, 7.0)
    g = frozen.split(np.arange(6, dtype=np.float32), control_fixed=seed)
    assert g.control is seed and g.f_log == 0.0 and g.motor is None and g.stiff.tolist() == [0, 1, 2]


def test_initial_theta_is_seed_then_unit_factors():
    seed = jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=jnp.float32)
    theta = initial_theta(FULL, seed)
    assert theta.shape == (FULL.size,)
    assert np.asarray(theta[:5]).tolist() == np.asarray(seed).tolist()
    assert not np.asarray(theta[5:]).any()
