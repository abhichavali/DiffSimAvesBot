"""Runtime parameters derived from the morphology design (shoulder/*.tg)."""

from morphology.motor_drive import motor_actuator_gains
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
            "drive_phase_offset_rad": morph.motors.drive_phase_offset_rad,
            "tendon_pretension": morph.motors.tendon_pretension,
        },
    }


def get_control_parameters():
    morph = load_design_morphology()
    return {
        "motor_speed": morph.motors.motor_direction * morph.motors.motor_speed,
        "ramp_duration": 2.0,
        "flap_actuator_names": ["L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4"],
        "flap_actuator_gains": motor_actuator_gains(morph),
    }
