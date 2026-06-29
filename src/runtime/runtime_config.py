"""Runtime parameters derived from the morphology design."""

import os
from pathlib import Path

from runtime.control_signal import load_control_for_morphology
from runtime.morphology_runtime import load_design_morphology


def load_runtime_config(config_path=None):
    """Return a dict-shaped view of runtime-relevant morphology fields.

    ``config_path`` is accepted for API compatibility but ignored; set
    ``MORPHOLOGY_PATH`` to choose a different `.tg` file.
    """
    del config_path
    morph = load_design_morphology()
    return {
        "aero": {
            "num_blades": morph.wing.aero.blades,
            # Recompute aero every N physics steps, hold the force between (the FD
            # window scales with substep, so it's consistent). At the scaled-down
            # timestep (0.13 ms) every-step aero is the runtime bottleneck; 4 puts
            # the aero cadence back at ~0.52 ms — the same rate that was fine at
            # the old 0.5 ms step — for ~4x fewer aero calls.
            "substep": 4,
            # Mirror the left wing's blade forces onto the right to halve aero
            # cost. OFF by default: the right wing's pitch DOF evolves
            # independently and desyncs from the perfect mirror, so it gets fed
            # the left wing's aero and over-feathers (~121 vs ~96 deg pitch
            # swing). Per-wing compute keeps both wings clean and symmetric.
            # Re-enable only if realtime speed matters more than per-wing pitch.
            "mirror_wings": False,
        },
        "wing_pitch_parameters": {
            "upstroke_pitch": morph.wing.pitch.upstroke_pitch_rad,
            "downstroke_pitch": morph.wing.pitch.downstroke_pitch_rad,
            "upstroke_stiffness": morph.wing.pitch.upstroke_stiffness,
            "downstroke_stiffness": morph.wing.pitch.downstroke_stiffness,
            "free_angle": morph.wing.pitch.upstroke_pitch_rad,
            "limit_stiffness": morph.wing.pitch.upstroke_stiffness,
            "limit_damping": morph.wing.pitch.damping,
            "initial_pitch_angle": morph.wing.pitch.initial_rad,
        },
        "assets": {
            "meshes": [{"file": morph.wing.mesh_file}],
        },
        "compiler": {
            "meshdir": morph.world.meshdir,
        },
        "motors": {
            "crank_phase_lag_rad": morph.motors.crank_phase_lag_rad,
            "tendon_rest_length": morph.motors.tendon_rest_length,
        },
    }


def get_control_parameters():
    morph = load_design_morphology()
    tg_path = Path(os.environ.get("MORPHOLOGY_PATH", "design/shoulder.tg"))
    control = load_control_for_morphology(tg_path)
    return {
        "control_frequency_rad_s": control.omega,
        "control_a0": morph.motors.motor_direction * control.a0,
        "control_a": tuple(morph.motors.motor_direction * value for value in control.a),
        "control_b": tuple(morph.motors.motor_direction * value for value in control.b),
        "ramp_duration": 2.0,
        "flap_actuator_names": ["L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4"],
    }
