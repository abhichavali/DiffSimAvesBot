"""CPU quasi-steady blade-element aerodynamics (numpy): the reference the MJX port matches.

``aerodynamics`` is the per-strip model in the wing root-site frame; ``AeroBladeRuntime``
evaluates it for both wings each step and applies the forces at the blade sites.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from morphology.wing_geometry import load_default_wing_geometry
from runtime.runtime_config import load_runtime_config


def _ry(theta: float) -> np.ndarray:
    """Right-handed rotation about the Y axis (MATLAB's ``ry``)."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _cd(alpha, a, b, c, d):
    """Hovering drag coefficient (cosine fit) at angle of attack ``alpha`` [rad]."""
    alphadeg = alpha * 180 / np.pi
    CD = a - b * np.cos((c * alphadeg - d) * np.pi / 180)
    return CD


def _cl(alpha, e, f, g, h):
    """Hovering lift coefficient (sine fit) at angle of attack ``alpha`` [rad]."""
    alphadeg = alpha * 180 / np.pi
    CL = e + f * np.sin((g * alphadeg - h) * np.pi / 180)
    return CL


# Left and right wings differ ONLY by a fixed set of mirror sign flips. These
# match the validated MJX port (runtime/mjx/aero._side_signs):
#   Fik  rotational-drag chordwise force   lift  FLi sign in the force/torque sums
#   rot  F_rot sign in the sums            addo  rotational added-mass (lamda_xo) sign
#   add2 F_add2 sign                       cof   chordwise CoP lever-arm sign
_SIDE_SIGNS = {
    #     (Fik,  lift,  rot,  addo,  add2,  cof)
    "L": (1.0, -1.0, -1.0, 1.0, -1.0, -1.0),
    "R": (-1.0, 1.0, 1.0, -1.0, 1.0, 1.0),
}


def aerodynamics(side, omega_w, omega_b, vel_h, omega_w_d,
                 i, dt, Rw, NumBlade, rho, chord, span_cutoffs=None,
                 vel_h_prev=None, return_cop=False):
    """Blade-strip force + pitching moment in the root-site frame, for one wing.

    ``side`` is "L" or "R"; the two wings differ only by the mirror sign set above.
    """
    try:
        sFik, sLift, sRot, sAddo, sAdd2, sCof = _SIDE_SIGNS[side]
    except KeyError:
        raise ValueError(f"side must be 'L' or 'R', got {side!r}")

    Crd = 5
    Cadd2 = 0
    c4 = 1
    addr = 0.2500
    rotr = 0.2500
    ct1 = 0.82
    ct2 = 0.05

    # Quasi-steady polar coefficients (Sane & Dickinson 2002, revolving wing):
    #   CD = a - b*cos(c*alpha_deg - d);  CL = e + f*sin(g*alpha_deg - h)
    # Canonical fruit-fly fit; well-behaved at the endpoints but NOT fitted to
    # this robot's wing/Reynolds number -- re-fit before trusting magnitudes.
    a = 1.92
    b = 1.55
    c = 2.04
    d = 9.82
    e = 0.225
    f = 1.58
    g = 2.13
    h = 7.2

    x_rot = 0               # rotational axis, 0 corresponds to leading edge
    r_cof_rot = 0 * chord
    r_cof_add = 0 * chord

    if span_cutoffs is None:
        span_cutoffs = np.full(NumBlade, Rw)

    # Rotational drag, summed over strips (gated by span cutoff).
    Fik_lt = 0
    Tik_lt = 0
    for k in range(1, NumBlade + 1):
        m = k / NumBlade
        Rcurr = span_cutoffs[k - 1]

        rdi = 0 if (Rw * i) > Rcurr else 1

        wsum = omega_w[1] + omega_b[1]
        abswsum = abs(wsum)

        span = Rw / NumBlade
        local = chord * (x_rot - m)

        Fik_lt += sFik * rdi * 0.5 * rho * wsum * abswsum * span * Crd * local * abs(local) * chord / NumBlade
        Tik_lt -= rdi * 0.5 * rho * wsum * abswsum * span * Crd * (local ** 2) * abs(local) * chord / NumBlade

    # --------------------------------------------------------
    # Blade-element analysis
    # --------------------------------------------------------

    rle = -i * Rw

    vle = np.cross(omega_w + omega_b, np.array([0, rle, 0])) + vel_h
    vle[1] = 0

    # Previous-step element velocity for the added-mass finite difference. ``dt``
    # is the FD window (dt * substep); ``vel_h_prev`` lets the chordwise
    # acceleration include the hinge's translational acceleration, falling back to
    # ``vel_h`` for the old rotation-only behaviour.
    vh_prev = vel_h if vel_h_prev is None else vel_h_prev
    omega_wLI = omega_w - omega_w_d * dt
    vleLI = np.cross(omega_wLI + omega_b, np.array([0, rle, 0])) + vh_prev
    vleLI[1] = 0

    acc_xw = (vle[0] - vleLI[0]) / dt
    omega_yw_dot = omega_w_d[1]

    # Velocity direction & folded angle of attack in [0, pi/2] (arctan2 of
    # absolute components is robust to vz == 0; lift/drag *direction* is set
    # separately below, so quadrant folding is fine for a symmetric section).
    norm_v = np.linalg.norm(vle)
    if norm_v != 0:
        dir_vle = vle / norm_v
        alpha = np.arctan2(abs(dir_vle[0]), abs(dir_vle[2]))
    else:
        dir_vle = np.zeros(3)
        alpha = 0

    dir_drag = -dir_vle
    dir_drag[1] = 0

    # Non-zero sign so vx == 0 does not collapse the +-90 deg rotation to identity.
    lift_sign = 1.0 if vle[0] >= 0 else -1.0
    dir_lift = _ry(-lift_sign * np.pi / 2) @ dir_vle

    # --------------------------------------------------------
    # Translational lift & drag
    # --------------------------------------------------------

    q = 0.5 * rho * (norm_v ** 2)

    FDi = q * _cd(alpha, a, b, c, d) * chord * (Rw / NumBlade) * dir_drag
    FLi = q * _cl(alpha, e, f, g, h) * chord * (Rw / NumBlade) * dir_lift

    r_cof_trans = x_rot - (ct1 * abs(alpha) / np.pi + ct2)
    r_cof_trans = np.clip(r_cof_trans, x_rot - 1, x_rot)
    r_cof_trans *= chord

    # --------------------------------------------------------
    # Rotational aerodynamic force
    # --------------------------------------------------------

    Circulation_rot = np.pi * (0.75 - x_rot) * (chord ** 2) * omega_w[1]
    F_rot_i = c4 * abs(rle) * (1 / NumBlade) * np.array([rho * Circulation_rot * norm_v, 0, 0])

    # --------------------------------------------------------
    # Added-mass effects
    # --------------------------------------------------------

    cr = (-x_rot + 0.5) * chord
    lamda_x = np.pi * rho * (0.5 * chord) ** 2
    lamda_xo = -np.pi * rho * (0.5 * chord) ** 2 * cr

    F_addx_i = -lamda_x * acc_xw + sAddo * lamda_xo * omega_yw_dot
    F_add_i = (Rw / NumBlade) * np.array([F_addx_i, 0, 0])
    F_add2_i = sAdd2 * Cadd2 * np.pi * rho * (0.5 * chord) ** 2 * Rw * (Rw / NumBlade) * omega_w[2] * omega_w[1]

    # --------------------------------------------------------
    # Final force & torque summation. The spanwise lever arm ``rle`` is
    # intentionally excluded from Taui: the caller applies each strip's force at
    # the strip location, so the spanwise moment comes from the application point.
    # Including rle here too would double-count it.
    # --------------------------------------------------------

    Fi = (FDi + sLift * FLi + F_add_i + sRot * F_rot_i) \
         + np.array([Fik_lt, 0, 0]) \
         + np.array([F_add2_i, 0, 0])

    Taui = (
        np.cross([0, 0, sCof * r_cof_trans], FDi + sLift * FLi)
        + np.cross([0, 0, sCof * r_cof_add], F_add_i + np.array([F_add2_i, 0, 0]))
        + np.cross([0, 0, sCof * r_cof_rot], sRot * F_rot_i)
        + np.array([0, Tik_lt, 0])
    )

    if return_cop:
        return Fi, Taui, np.array([0.0, rle, sCof * r_cof_trans], dtype=float)

    return Fi, Taui


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
