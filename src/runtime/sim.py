"""The CPU plate-wing rollout step, in the one order every hummer loop uses.

``rest_state`` is the start of every CPU drive path (2000-step passive settle,
neutral wing pitch, crank sync); ``PlateWingSim`` adds the blade-element aero and
the pitch limiter and advances one physics step at a time:

    [caller: stroke-plane body_quat / mocap writes, command from data.time]
    qfrc_applied = 0, ctrl = 0, aero.apply, pitch.apply, [wrench],
    ctrl[cranks] = command, lift readout, mj_step

Float accumulation into ``qfrc_applied`` is order-sensitive, so aero -> pitch ->
wrench is fixed. ``body_quat`` and mocap writes are read only by ``mj_step`` and
the command depends only on ``data.time``, so callers make them before ``step``.
Loops that need to own ``mj_step`` (a viewer's pacing hook) call ``apply``.

The model is the caller's: set its integrator, timestep and tendon arrays before
constructing the sim. ``morph`` is the Morphology the model was compiled from.
"""

import mujoco

from runtime.aero import AeroBladeRuntime
from runtime.crank import CRANK_ACTUATORS, sync_crank_disks_at_rest
from runtime.wing_pitch import WingPitchLimiter, calibrate_wing_pitch_at_rest


def rest_state(model, morph, phase_lag: float | None = None) -> mujoco.MjData:
    """A fresh MjData at the rest pose: settled, pitch neutral, cranks synced.

    ``phase_lag=None`` applies the design's crank lag; ``data.time`` is 0.
    """
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data, morph, phase_lag)
    return data


class PlateWingSim:
    """A plate-wing model at rest plus its aero and pitch limiter; ``step`` advances it.

    ``pitch_params`` = (upstroke pitch, downstroke pitch, upstroke stiffness,
    downstroke stiffness) overrides the design's feathering.
    """

    def __init__(self, model, morph, *, phase_lag=None, aero_substep=None, pitch_params=None):
        self.model = model
        self.data = rest_state(model, morph, phase_lag)
        self.aero = AeroBladeRuntime(model, morph, substep=aero_substep)
        self.pitch = WingPitchLimiter(model, self.data, morph)
        if pitch_params is not None:
            (self.pitch.upstroke_pitch, self.pitch.downstroke_pitch,
             self.pitch.upstroke_stiffness, self.pitch.downstroke_stiffness) = (float(x) for x in pitch_params)
        ids = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in CRANK_ACTUATORS)
        self.crank_ids = [i for i in ids if i >= 0]

    def apply(self, ctrl, wrench=None) -> float:
        """Everything before ``mj_step``: forces into qfrc_applied, ``ctrl`` onto the cranks.

        ``ctrl`` is one velocity command for every crank, or one per entry of
        ``crank_ids``. ``wrench(model, data)`` adds external forces after the aero
        and pitch terms. Returns the aero lift (N) applied this step.
        """
        model, data = self.model, self.data
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        self.aero.apply(model, data)
        self.pitch.apply(model, data)
        if wrench is not None:
            wrench(model, data)
        data.ctrl[self.crank_ids] = ctrl
        return self.aero.lift()

    def step(self, ctrl, wrench=None) -> float:
        """``apply`` then ``mj_step``; returns the lift applied over the step."""
        lift = self.apply(ctrl, wrench)
        mujoco.mj_step(self.model, self.data)
        return lift
