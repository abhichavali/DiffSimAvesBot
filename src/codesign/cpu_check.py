"""CPU MuJoCo (numpy aero) replay of a co-design candidate: the transfer check for MJX results.

cpu_rollout() runs a design on the SAME compiled scene with the SAME tendon arrays the MJX
optimizer used (no .tg round trip), so any MJX<->CPU difference is the physics rollout
itself. to_cpu() decodes a packed theta into that CpuDesign; cpu_gate() rolls one design
over a list of (integrator, timestep) cells and reduces them to the robust numbers the
optimizer's CPU transfer gate selects on.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import mujoco
import numpy as np

from codesign.genome import Genome, ricker_atoms
from codesign.rig import INTEGRATORS
from runtime.disk_orientation import StrokePlaneSignal
from runtime.sim import PlateWingSim


class CpuDesign(NamedTuple):
    ctrl: tuple  # (a0, a_depths, b_depths, omega): cmd = a0*(1 + sum a_n cos + b_n sin)
    tendons: tuple | None = None  # (stiffness(ntendon,), lengthspring(ntendon, 2)) or the scene's
    pitch: tuple | None = None  # (up deadband, down deadband, up stiffness, down stiffness)
    strokeplane: tuple | None = None  # (10 coeffs, tilt0, pitch0, max_rad)


class CpuResult(NamedTuple):
    lift: float  # mean aero lift [N]
    peak: float  # max |motor torque| over the window [Nm]
    power: float  # mean supplied motor power [W]
    cycle_peaks: list  # per-cycle max |torque|; drift = last - first
    swing: float  # std of the |torque| trace (the MJX Metrics.swing)


def cpu_rollout(scene_path, design: CpuDesign, integrator, *, morph, periods=3.0, warmup=2.0,
                phase_lag=0.0, timestep_ms=None, aero_substep=None, wavelet=None) -> CpuResult:
    """Run ``design`` for ``warmup`` + ``periods`` drive periods on MuJoCo CPU.

    ``morph`` is the Morphology ``scene_path`` was compiled from.
    ``wavelet`` is genome.wavelet_dictionary(...) when the control coefficients are
    Ricker atom amplitudes (all in ctrl[1]; ctrl[2] unused).
    """
    a0, a, b, omega = design.ctrl
    a = np.asarray(a); b = np.asarray(b); harm = np.arange(1, len(a) + 1)
    if wavelet is not None:
        wl_centers, wl_sigmas = np.asarray(wavelet[0]), np.asarray(wavelet[1])
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.integrator = INTEGRATORS[integrator]
    if timestep_ms is not None:
        model.opt.timestep = timestep_ms * 1e-3
    if design.tendons is not None:
        stiff, lenspring = design.tendons
        model.tendon_stiffness[:] = np.asarray(stiff)
        model.tendon_lengthspring[:] = np.asarray(lenspring)
    sim = PlateWingSim(model, morph, phase_lag=phase_lag, aero_substep=aero_substep,
                       pitch_params=design.pitch)
    data, aids = sim.data, sim.crank_ids
    sp = StrokePlaneSignal(model, *design.strokeplane) if design.strokeplane is not None else None

    def step():
        if sp is not None:  # the signal the MJX rollout applied
            sp.apply(data)
        phase = omega * data.time
        if wavelet is not None:
            mod = float(np.sum(a * ricker_atoms(phase, wl_centers, wl_sigmas, np)))
        else:
            mod = float(np.sum(a * np.cos(harm * phase) + b * np.sin(harm * phase)))
        lift = sim.step(a0 * (1.0 + mod))
        pm = data.actuator_force[aids] * data.actuator_velocity[aids]
        return (lift, float(np.max(np.abs(data.actuator_force[aids]))),
                float(np.sum(np.maximum(0.0, pm))))

    period = math.tau / omega
    for _ in range(int(warmup * period / model.opt.timestep)):
        step()
    rec = [step() for _ in range(int(periods * period / model.opt.timestep))]
    L = np.array([r[0] for r in rec]); T = np.array([r[1] for r in rec]); P = np.array([r[2] for r in rec])
    spc = max(1, int(round(period / model.opt.timestep)))
    ncyc = max(1, len(T) // spc)
    cycle_peaks = [float(np.max(T[i * spc:min((i + 1) * spc, len(T))])) for i in range(ncyc)]
    return CpuResult(float(L.mean()), float(T.max()), float(P.mean()), cycle_peaks, float(np.std(T)))


def to_cpu(genome: Genome, theta, control_seed, rig, omega, strokeplane=None) -> CpuDesign:
    """Decode a packed theta into the CpuDesign the MJX rollout of it simulated.

    Harmonics ride omega*exp(f_log) here, not the MJX rollout's crank-locked
    omega*exp(f_log + a0_log).
    """
    g = genome.split(np.asarray(theta), np.asarray(control_seed))
    K, ctrl = genome.K, g.control
    a0 = float(rig.drive.a0) * math.exp(ctrl[0])
    a_dep, b_dep = (ctrl[1:], ctrl[:0]) if genome.wavelet else (ctrl[1 : 1 + K], ctrl[1 + K :])
    base_stiff, base_lenspring = np.asarray(rig.tendons[2]), np.asarray(rig.tendons[3])
    tendons = pitch = sp = None
    if g.stiff is not None or g.motor is not None:
        bs, bl = base_stiff.copy(), base_lenspring.copy()
        if g.stiff is not None:
            li, ri = np.asarray(rig.tendons[0]), np.asarray(rig.tendons[1])
            bs[li] = base_stiff[li] * np.exp(g.stiff); bs[ri] = bs[li]
            bl[li] = base_lenspring[li] * np.exp(g.pre)[:, None]; bl[ri] = bl[li]
        if g.motor is not None:
            mids = np.asarray(rig.motor_ids)
            bs[mids] = base_stiff[mids] * math.exp(g.motor[0])
            bl[mids] = base_lenspring[mids] * math.exp(g.motor[1])
        tendons = (bs, bl)
    if g.pitch is not None:
        p = rig.pitch
        pitch = (float(p.upstroke_pitch) * math.exp(g.pitch[0]),
                 float(p.downstroke_pitch) * math.exp(g.pitch[1]),
                 float(p.upstroke_stiffness) * math.exp(g.pitch[2]),
                 float(p.downstroke_stiffness) * math.exp(g.pitch[3]))
    if g.sp is not None:
        sp = (g.sp, strokeplane["tilt0"], strokeplane["pitch0"], strokeplane["max_rad"])
    return CpuDesign((a0, a_dep, b_dep, omega * math.exp(g.f_log)), tendons, pitch, sp)


def cpu_gate(scene_path, design: CpuDesign, cells, **kw):
    """Roll ``design`` in every (integrator, timestep_ms) cell.

    Returns (rows, robust peak = max peak, gap = (max - min peak)/max, min lift,
    max |per-cycle drift|); ``kw`` goes to cpu_rollout.
    """
    rows = []
    for integ, ts in cells:
        r = cpu_rollout(scene_path, design, integ, timestep_ms=ts, **kw)
        pc = r.cycle_peaks
        drift = (pc[-1] - pc[0]) if len(pc) >= 2 else 0.0
        rows.append({"integrator": integ, "timestep_ms": ts, "L": r.lift, "T": r.peak, "P": r.power,
                     "drift": drift})
    robust = max(x["T"] for x in rows)
    gap = (robust - min(x["T"] for x in rows)) / max(robust, 1e-9)
    lift = min(x["L"] for x in rows)
    driftmax = max(abs(x["drift"]) for x in rows)
    return rows, robust, gap, lift, driftmax
