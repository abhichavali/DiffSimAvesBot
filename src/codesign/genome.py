"""The packed parameter vector (theta, the "genome") the co-design optimizers move, and its control basis.

theta concatenates the active gene groups in this fixed order:

  control (2K+1)  a0_log, a_1..a_K, b_1..b_K. a0 is a log-factor of the design's a0; a_n/b_n
                  are modulation DEPTHS (fractions of a0): cmd = a0*(1 + sum a_n cos + b_n sin).
                  With wavelet=True the 2K coefficients are Ricker atom amplitudes instead.
  f_log (1)       log factor on the harmonic clock frequency (which a0_log also scales)
  stiff (n)       per structural tendon log stiffness factor (n = 24, or 12 mirrored pairs)
  pre (n)         per structural tendon log rest-length (pretension) factor
  pitch (4)       log factors on the up/down pitch deadband and up/down pitch stiffness
  motor (2)       log factors on the shared motor-tendon stiffness and rest length
  sp (10)         stroke-plane coefficients: tilt then pitch, each mean + 2 harmonics of
                  the L_t1 crank angle

Genome records which groups are present; split() cuts a jnp or numpy theta into Genes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np


class Genes(NamedTuple):
    """One theta cut into its groups; an absent group is None (control: the fixed seed,
    f_log: 0.0)."""
    control: Any
    f_log: Any
    stiff: Any
    pre: Any
    pitch: Any
    motor: Any
    sp: Any

    def clock_factor(self, exp=math.exp):
        """exp(f_log + a0_log): the harmonic clock over the design frequency. The clock is
        crank-locked: a0 (control[0]) is the crank speed, so the modulation rides it."""
        return exp(self.f_log + self.control[0])


class Genome(NamedTuple):
    K: int  # Fourier harmonics (or 2K wavelet atoms)
    n_tend: int  # tuned structural tendons
    control: bool = True
    frequency: bool = False
    tendons: bool = False
    pitch: bool = False
    motor: bool = False
    strokeplane: bool = False
    wavelet: bool = False

    def slices(self) -> dict[str, slice]:
        """Group name -> its slice of theta, for the groups present."""
        groups = (("control", 2 * self.K + 1, self.control), ("f_log", 1, self.frequency),
                  ("stiff", self.n_tend, self.tendons), ("pre", self.n_tend, self.tendons),
                  ("pitch", 4, self.pitch), ("motor", 2, self.motor), ("sp", 10, self.strokeplane))
        out, i = {}, 0
        for name, n, on in groups:
            if on:
                out[name] = slice(i, i + n)
                i += n
        return out

    @property
    def size(self) -> int:
        return sum(s.stop - s.start for s in self.slices().values())

    def split(self, theta, control_fixed) -> Genes:
        """Cut ``theta`` into Genes; a frozen control reads as ``control_fixed``."""
        s = self.slices()

        def cut(name):
            return theta[s[name]] if name in s else None

        return Genes(control=theta[s["control"]] if "control" in s else control_fixed,
                     f_log=theta[s["f_log"].start] if "f_log" in s else 0.0,
                     stiff=cut("stiff"), pre=cut("pre"), pitch=cut("pitch"), motor=cut("motor"),
                     sp=cut("sp"))


def load_control_seed(path: Path | None, K: int, drive, base_a0: float,
                      wavelet: bool = False) -> list[float]:
    """Control seed [a0_log, a_1..a_K, b_1..b_K]; a0 stored as log-factor vs base_a0.

    ``path`` is a theta_best.json (a/b absolute); None seeds from the design. In wavelet
    mode the 2K coeffs are atom amplitudes seeded to 0 (a pure sinusoid), so a Fourier
    warm start keeps its a0 and drops its harmonics.
    """
    if path is None:
        a, b, a0 = list(drive.a), list(drive.b), float(drive.a0)
    else:
        d = json.loads(Path(path).read_text())
        a, b, a0 = list(d["a"]), list(d["b"]), float(d["a0"])
    if wavelet:
        return [math.log(a0 / base_a0)] + [0.0] * (2 * K)
    a = (a + [0.0] * K)[:K]
    b = (b + [0.0] * K)[:K]
    a = [x / base_a0 for x in a]  # absolute -> modulation depth
    b = [x / base_a0 for x in b]
    return [math.log(a0 / base_a0)] + a + b  # same sign as base_a0 -> ratio > 0


def initial_theta(genome: Genome, control_seed, codesign: Path | None = None,
                  lock_wings: bool = False):
    """The starting theta: the control seed with every other factor at 1x (log 0).

    ``codesign`` (an evolve_codesign genome or a theta_full.json) seeds the groups it
    holds instead. A file with a "layout" (the Genome that wrote it) is split by group
    name: a group the seed lacks keeps its default, and a different K or tendon count
    raises. An older file without one is read BY POSITION as [control 2K+1][tendons
    2n][motor 2] with the stroke plane as the last 10 values.
    """
    K, n = genome.K, genome.n_tend
    seed = None
    if codesign is not None:
        d = json.loads(Path(codesign).read_text())
        seed = jnp.asarray(d.get("genome", d.get("theta")), dtype=jnp.float32)
        if "layout" in d:
            return _theta_from_layout(genome, control_seed, seed, Genome(**d["layout"]))
        want = 2 * K + 1 + 2 * n + 2
        assert seed.shape[0] >= want, f"init-codesign len {seed.shape[0]} < {want}"
        # Sliced BY POSITION, so a genome from the other wing mode silently decodes as
        # garbage (48 unlocked tendon params read as 24 mirrored ones). Only mismatched
        # widths can be caught, so catch them.
        assert not lock_wings or seed.shape[0] in (want, want + 10), (
            f"--lock-wings expects a {want}-param genome (or {want+10} with stroke-plane); "
            f"got {seed.shape[0]} -- that looks like an UNLOCKED (48-tendon) genome, "
            "which would be sliced into the wrong genes")
    parts = []
    if genome.control:
        if seed is not None:
            # Keep the warm-start a0; in wavelet mode zero the basis coeffs so a Fourier
            # design's harmonics aren't reinterpreted as atom amplitudes.
            c = seed[:2 * K + 1]
            parts.append(c.at[1:].set(0.0) if genome.wavelet else c)
        else:
            parts.append(control_seed)
    if genome.frequency:
        parts.append(jnp.zeros(1, dtype=jnp.float32))
    if genome.tendons:
        parts.append(seed[2 * K + 1:2 * K + 1 + 2 * n] if seed is not None
                     else jnp.zeros(2 * n, dtype=jnp.float32))
    if genome.pitch:
        parts.append(jnp.zeros(4, dtype=jnp.float32))
    if genome.motor:
        parts.append(seed[2 * K + 1 + 2 * n:2 * K + 1 + 2 * n + 2] if seed is not None
                     else jnp.zeros(2, dtype=jnp.float32))
    if genome.strokeplane:
        if seed is not None and seed.shape[0] >= 2 * K + 1 + 2 * n + 2 + 10:
            parts.append(seed[-10:])
        else:
            parts.append(jnp.zeros(10, dtype=jnp.float32))  # the compiled mount plane
    return jnp.concatenate(parts)


def _theta_from_layout(genome: Genome, control_seed, seed, src: Genome):
    """initial_theta for a seed written by ``src``: each group of ``genome`` taken by name."""
    if (src.K, src.n_tend) != (genome.K, genome.n_tend):
        raise ValueError(f"--init-codesign seed has K={src.K}, n_tend={src.n_tend}; this run "
                         f"has K={genome.K}, n_tend={genome.n_tend} (--harmonics, --lock-wings)")
    have = src.split(seed[:src.size], None)  # an EA genome may carry geometry genes after
    parts = []
    for name, sl in genome.slices().items():
        v = getattr(have, name)
        if name == "control":
            # Keep the warm-start a0; across a Fourier/wavelet switch zero the basis coeffs.
            v = control_seed if v is None else (v.at[1:].set(0.0) if src.wavelet != genome.wavelet else v)
        elif name == "f_log":
            v = jnp.full(1, v, dtype=jnp.float32)  # a scalar, 0.0 when the seed has none
        elif v is None:
            v = jnp.zeros(sl.stop - sl.start, dtype=jnp.float32)
        parts.append(v)
    return jnp.concatenate(parts)


def wavelet_dictionary(n_atoms):
    """A dyadic real-wavelet (periodic Ricker / Mexican-hat) frame of exactly
    ``n_atoms`` atoms on the crank cycle [0, 2*pi).

    Level L has ``2**L`` equal-spaced translations, each a Ricker atom of scale
    sigma = pi / 2**L (coarse-to-fine). Unlike Fourier harmonics (global sinusoids),
    each atom is LOCALIZED in crank phase.

    Returns (centers, sigmas, freqs) as float64 numpy arrays. ``freqs`` = 2**L is the
    atom's characteristic frequency, the wavelet analog of harmonic number n for the
    scale-weighted energy regularizer.

    ponytail: the last level is truncated to hit n_atoms exactly (e.g. 6 = 1+2+3
    of the level-2 quartet), leaving one coarse-only phase gap near 3*pi/2. Fine
    for an exploration basis; swap in a full 2^L frame if that gap ever matters.
    """
    centers, sigmas, freqs, level = [], [], [], 0
    while len(centers) < n_atoms:
        n = 2 ** level
        for k in range(n):
            centers.append(2.0 * math.pi * k / n)
            sigmas.append(math.pi / n)
            freqs.append(float(n))
            if len(centers) >= n_atoms:
                break
        level += 1
    return np.asarray(centers), np.asarray(sigmas), np.asarray(freqs)


def ricker_atoms(phase, centers, sigmas, xp):
    """Ricker (Mexican-hat) atom values at scalar ``phase`` for all dictionary atoms.
    ``xp`` is jnp (MJX rollout) or np (CPU validation), so both sides evaluate the SAME
    waveform.

    d = signed shortest angular distance to each center (periodic); the Gaussian
    envelope makes the antipodal wrap-kink negligible for fine atoms.
    """
    d = xp.mod(phase - centers + math.pi, 2.0 * math.pi) - math.pi
    u = d / sigmas
    return (1.0 - u * u) * xp.exp(-0.5 * u * u)
