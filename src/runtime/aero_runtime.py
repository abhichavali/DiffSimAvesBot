from dataclasses import dataclass

import mujoco
import numpy as np

from runtime.runtime_config import load_runtime_config
from utils.aero.aerodynamics import aerodynamics
from utils.aero.wing_geometry import load_default_wing_geometry


RHO_AIR = 1.1839

# The unsteady (added-mass) aero force is proportional to finite-differenced
# acceleration. In this explicit runtime it is noisy enough to inflate lift by
# 2x+ and does not transfer cleanly between CPU and MJX. Keep the default model
# quasi-steady; opt/optimize_control.set_unsteady_aero(True) opts into the old
# capped hack for diagnostics.
# ponytail: explicit added mass stays opt-in; use an implicit added-mass state if
# this needs to become production physics.
OMEGA_DOT_CAP = 0.0     # rad/s^2
LIN_ACC_CAP = 0.0       # m/s^2


@dataclass
class WingAeroState:
    side: str
    root_site_id: int
    blade_site_ids: list[int]
    wing_body_id: int
    previous_omega_local: np.ndarray | None = None
    previous_velocity_local: np.ndarray | None = None
    cached_forces: list[np.ndarray] | None = None
    cached_torques: list[np.ndarray] | None = None
    cached_points: list[np.ndarray] | None = None


class AeroBladeRuntime:
    def __init__(self, model, config_path=None, *, substep=None):
        config = load_runtime_config(config_path) or {}
        aero_cfg = config.get("aero", {})
        num_blades = int(aero_cfg.get("num_blades", 10))
        # Recompute aero forces every `substep` physics steps; reapply the
        # cached force between recomputes. 1 = compute every step.
        self.substep = max(1, int(aero_cfg.get("substep", 1) if substep is None else substep))
        # Compute only the left wing and reflect its forces to the right (the
        # wings are exact mirror images). Falls back to per-wing compute if off
        # or if there isn't exactly the L,R pair to mirror.
        self.mirror_wings = bool(aero_cfg.get("mirror_wings", False))
        self.geometry = load_default_wing_geometry(num_blades=num_blades, config_path=config_path)
        self.num_blades = num_blades
        self.dt = float(model.opt.timestep)
        self.wings = [
            self._resolve_wing(model, "L"),
            self._resolve_wing(model, "R"),
        ]
        self._step_counter = 0

    def apply(self, model, data) -> None:
        recompute = (self._step_counter % self.substep) == 0
        mirror = self.mirror_wings and len(self.wings) == 2
        for idx, wing in enumerate(self.wings):
            if recompute or wing.cached_forces is None:
                if mirror and idx == 1:
                    self._mirror_wing(model, data, self.wings[0], wing)
                else:
                    self._compute_wing(model, data, wing)
            self._reapply_wing(model, data, wing)
        self._step_counter += 1

    def _resolve_wing(self, model, side: str) -> WingAeroState:
        root_site_id = self._site_id(model, f"{side}_wing_Root_Site")
        blade_site_ids = [
            self._site_id(model, f"{side}_wing_aero_blade_{blade_index + 1:02d}")
            for blade_index in range(self.num_blades)
        ]
        wing_body_id = int(model.site_bodyid[root_site_id])

        for blade_site_id in blade_site_ids:
            blade_body_id = int(model.site_bodyid[blade_site_id])
            if blade_body_id != wing_body_id:
                raise ValueError(
                    f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, blade_site_id)} "
                    f"is attached to body {blade_body_id}, expected {wing_body_id}"
                )

        return WingAeroState(
            side=side,
            root_site_id=root_site_id,
            blade_site_ids=blade_site_ids,
            wing_body_id=wing_body_id,
        )

    def _site_id(self, model, name: str) -> int:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"Required aerodynamic site not found: {name}")
        return site_id

    def _compute_wing(self, model, data, wing: WingAeroState) -> None:
        # Read the root-site spatial velocity from the site Jacobian, NOT
        # mj_objectVelocity: the latter uses d.cvel, which after mj_step is stale
        # (it reflects the pre-integration qvel of the last step), so the aero
        # would run on a one-step-lagged velocity while the force is applied at the
        # current position. That ~20% velocity error is harmless for the quasi-
        # steady terms but the added-mass term finite-differences velocity (x
        # lamda/dt), so it blew up into a ~17x MJX<->CPU lift gap. jac @ qvel uses
        # the current qvel and matches the MJX runtime to ~1e-4 (the transfer fix).
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, wing.root_site_id)
        omega_global = jacr @ data.qvel
        velocity_global = jacp @ data.qvel

        root_rotation = np.asarray(data.site_xmat[wing.root_site_id]).reshape(3, 3)
        omega_local = root_rotation.T @ omega_global
        velocity_local = root_rotation.T @ velocity_global

        # Finite-difference window spans the substep gap between recomputes.
        fd_dt = self.dt * self.substep
        if wing.previous_omega_local is None:
            omega_local_dot = np.zeros(3)
            velocity_local_prev = velocity_local
        else:
            # Cap the finite-differenced accelerations so the explicit added-mass
            # force can't run away at small dt (see OMEGA_DOT_CAP / LIN_ACC_CAP).
            omega_local_dot = np.clip(
                (omega_local - wing.previous_omega_local) / fd_dt,
                -OMEGA_DOT_CAP, OMEGA_DOT_CAP,
            )
            dv_cap = LIN_ACC_CAP * fd_dt
            velocity_local_prev = velocity_local - np.clip(
                velocity_local - wing.previous_velocity_local, -dv_cap, dv_cap,
            )
        wing.previous_omega_local = omega_local.copy()
        wing.previous_velocity_local = velocity_local.copy()

        zero_body_omega = np.zeros(3)

        forces: list[np.ndarray] = []
        torques: list[np.ndarray] = []
        points: list[np.ndarray] = []

        for blade_index, blade_site_id in enumerate(wing.blade_site_ids):
            span_fraction = (blade_index + 0.5) / self.num_blades
            chord = float(self.geometry.chords[blade_index])
            # State is expressed in the root-site frame, so pass fd_dt (the
            # recompute window) as the finite-difference timestep.
            force_local, torque_local = aerodynamics(
                wing.side,
                omega_local,
                zero_body_omega,
                velocity_local,
                omega_local_dot,
                span_fraction,
                fd_dt,
                self.geometry.length,
                self.num_blades,
                RHO_AIR,
                chord,
                self.geometry.span_cutoffs,
                vel_h_prev=velocity_local_prev,
            )

            # force_local / torque_local are in the root-site frame; rotate both
            # with the root rotation and apply the force at the strip's site so
            # mj_applyFT supplies the spanwise moment (torque_local carries only
            # the chordwise pitching moment).
            forces.append(root_rotation @ force_local)
            torques.append(root_rotation @ torque_local)
            points.append(np.asarray(data.site_xpos[blade_site_id]).copy())

        wing.cached_forces = forces
        wing.cached_torques = torques
        wing.cached_points = points

    def _mirror_wing(self, model, data, src: WingAeroState, dst: WingAeroState) -> None:
        # The two wings are exact mirror images across the sagittal plane, and in
        # their respective root-site frames the mirror is a z-flip (verified: the
        # right wing's local velocity/force = reflz(left's) to ~9%). So map the
        # left blade forces into the right wing's frame instead of recomputing:
        #   F_R_global = R_root_R . reflz . R_root_L^T . F_L_global
        # Force is a true vector; torque (chordwise pitching moment) is a
        # pseudovector, so it picks up the extra det(reflz) = -1 sign.
        if src.cached_forces is None:
            dst.cached_forces = dst.cached_torques = dst.cached_points = None
            return
        reflz = np.diag([1.0, 1.0, -1.0])
        rot_src = np.asarray(data.site_xmat[src.root_site_id]).reshape(3, 3)
        rot_dst = np.asarray(data.site_xmat[dst.root_site_id]).reshape(3, 3)
        m_force = rot_dst @ reflz @ rot_src.T
        m_torque = rot_dst @ (-reflz) @ rot_src.T
        dst.cached_forces = [m_force @ f for f in src.cached_forces]
        dst.cached_torques = [m_torque @ t for t in src.cached_torques]
        # Apply at the right wing's own blade sites (already the mirror points).
        dst.cached_points = [np.asarray(data.site_xpos[sid]).copy() for sid in dst.blade_site_ids]

    def _reapply_wing(self, model, data, wing: WingAeroState) -> None:
        if wing.cached_forces is None:
            return
        for force_global, torque_global, point_global in zip(
            wing.cached_forces, wing.cached_torques, wing.cached_points
        ):
            mujoco.mj_applyFT(
                model,
                data,
                force_global,
                torque_global,
                point_global,
                wing.wing_body_id,
                data.qfrc_applied,
            )
