"""TEMP: remove after integration. Dict views of the active design for the MJX configs.

Only runtime/mjx/wing_pitch.py, runtime/mjx/rollout.py and the golden harness read
these; the CPU runtime takes ``morph`` directly. Both resolve the design from
``MORPHOLOGY_PATH``.
"""

from pathlib import Path

from runtime.aero import DEFAULT_AERO_SUBSTEP
from runtime.control_signal import load_control_for_morphology
from runtime.crank import CRANK_ACTUATORS
from runtime.morphology_runtime import load_design_morphology, resolve_morphology_path


def load_runtime_config(config_path=None):
    """Runtime-relevant morphology fields, as the MJX wing-pitch config reads them."""
    del config_path
    morph = load_design_morphology()
    return {
        "aero": {
            "num_blades": morph.wing.aero.blades,
            "substep": DEFAULT_AERO_SUBSTEP,
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
    """The design's drive signal (motor_direction folded in), as mjx.rollout.build_drive_config reads it."""
    morph = load_design_morphology()
    tg_path = Path(resolve_morphology_path())
    control = load_control_for_morphology(tg_path)
    return {
        "control_frequency_rad_s": control.omega,
        "control_a0": morph.motors.motor_direction * control.a0,
        "control_a": tuple(morph.motors.motor_direction * value for value in control.a),
        "control_b": tuple(morph.motors.motor_direction * value for value in control.b),
        "ramp_duration": 2.0,
        "flap_actuator_names": list(CRANK_ACTUATORS),
    }
