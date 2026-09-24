"""The hover controller's numpy (viewer) and jnp (trainer) paths agree and stay in bounds."""
import jax.numpy as jnp
import numpy as np

from runtime.hover_control import N_CTRL, N_FEAT, HoverCfg, pd_init_theta, policy_commands


def test_policy_commands_np_jnp_parity_and_bounds():
    cfg = HoverCfg(a0=65.0, f_max=0.15, tau_max=0.02, qadr=0, dofadr=0)
    K = np.arange(N_CTRL * N_FEAT).reshape(N_CTRL, N_FEAT) * 0.01
    b = np.arange(N_CTRL) * 0.1
    feats = np.linspace(-0.5, 0.5, N_FEAT)
    fn, tn, _ = policy_commands(K, b, feats, cfg, np)
    fj, tj, _ = policy_commands(jnp.asarray(K), jnp.asarray(b), jnp.asarray(feats), cfg, jnp)
    assert np.allclose(fn, np.array(fj), atol=1e-6), "force np/jnp mismatch"
    assert np.allclose(tn, np.array(tj), atol=1e-6), "tau np/jnp mismatch"
    assert np.all(np.abs(fn) <= cfg.f_max + 1e-9) and np.all(np.abs(tn) <= cfg.tau_max + 1e-9)
    assert pd_init_theta(cfg).shape == (N_CTRL * N_FEAT + N_CTRL,)
