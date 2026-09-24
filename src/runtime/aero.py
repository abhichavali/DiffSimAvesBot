"""CPU quasi-steady blade-element aerodynamics (numpy): the reference the MJX port matches.

``aerodynamics`` is the per-strip model in the wing root-site frame; ``AeroBladeRuntime``
evaluates it for both wings each step and applies the forces at the blade sites.
"""

from dataclasses import dataclass
from typing import NamedTuple

import mujoco
import numpy as np

from morphology.wing_geometry import load_wing_geometry


def _ry(theta: float) -> np.ndarray:
    """Right-handed rotation about the Y axis (MATLAB's ``ry``)."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


# Velocity -> lift direction for each sign of the chordwise velocity: ry(-sign * pi/2).
_LIFT_ROTATION = {sign: _ry(-sign * np.pi / 2) for sign in (1.0, -1.0)}


def _cross(a, b) -> np.ndarray:
    """``np.cross`` of two 3-vectors in numpy's exact operation order, without its ~30 us overhead."""
    a0, a1, a2 = a
    b0, b1, b2 = b
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


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


class AeroParams(NamedTuple):
    """Empirical aero coefficients, shared by this model and the MJX port.

    ``a..d`` are the drag polar, ``e..h`` the lift polar (Sane & Dickinson 2002,
    revolving wing): CD = a - b*cos(c*alpha_deg - d), CL = e + f*sin(g*alpha_deg - h).
    Canonical fruit-fly fit; well-behaved at the endpoints but NOT fitted to this
    robot's wing/Reynolds number -- re-fit before trusting magnitudes. The rest are
    the rotational-drag (Crd), added-mass (Cadd2), rotational-circulation (c4) and
    CoP (ct1, ct2) constants.
    """

    a: float = 1.92
    b: float = 1.55
    c: float = 2.04
    d: float = 9.82
    e: float = 0.225
    f: float = 1.58
    g: float = 2.13
    h: float = 7.2
    Crd: float = 5.0
    Cadd2: float = 0.0
    c4: float = 1.0
    ct1: float = 0.82
    ct2: float = 0.05


AERO_PARAMS = AeroParams()
RHO_AIR = 1.1839

# Left and right wings differ ONLY by a fixed set of mirror sign flips (the MJX
# port reads this same table):
#   Fik  rotational-drag chordwise force   lift  sign(vz) when the hinge edge leads
#   rot  F_rot sign in the sums            addo  rotational added-mass (lamda_xo) sign
#   add2 F_add2 sign                       cof   chordwise CoP lever-arm sign
_SIDE_SIGNS = {
    #     (Fik,  lift,  rot,  addo,  add2,  cof)
    "L": (1.0, -1.0, -1.0, 1.0, -1.0, -1.0),
    "R": (-1.0, 1.0, 1.0, -1.0, 1.0, 1.0),
}


def aerodynamics(side, omega_w, omega_b, vel_h, omega_w_d,
                 i, dt, Rw, NumBlade, rho, chord,
                 vel_h_prev=None, return_cop=False, *, x_rot, root):
    """Blade-strip force + pitching moment in the root-site frame, for one wing.

    ``side`` is "L" or "R"; the two wings differ only by the mirror sign set above.
    The strip is centred ``root + i * Rw`` out along the hinge (the root site's -y)
    and the hinge sits ``x_rot`` chords behind its leading edge: the wing mesh's
    ``WingGeometry.root_offset`` / ``.x_rot`` (morphology.wing_geometry).
    """
    try:
        sFik, sLift, sRot, sAddo, sAdd2, sCof = _SIDE_SIGNS[side]
    except KeyError:
        raise ValueError(f"side must be 'L' or 'R', got {side!r}")

    a, b, c, d, e, f, g, h, Crd, Cadd2, c4, ct1, ct2 = AERO_PARAMS
    r_cof_rot = 0 * chord
    r_cof_add = 0 * chord

    # Rotational drag of the chord pitching about the hinge: the exact integral over the
    # chord of Crd * rho/2 * w|w| * u|u| (force) and u^2|u| (moment), u = the element's
    # distance ahead of the hinge (x_rot at the leading edge, x_rot - 1 at the far edge).
    wsum = float(omega_w[1] + omega_b[1])  # a Python float: same IEEE ops, 4x cheaper than np.float64
    k0 = 0.5 * rho * wsum * abs(wsum) * (Rw / NumBlade) * Crd
    u0, u1 = x_rot, x_rot - 1
    Fik_lt = sFik * k0 * chord ** 3 * (u0 ** 2 * abs(u0) - u1 ** 2 * abs(u1)) / 3
    Tik_lt = -k0 * chord ** 4 * (u0 ** 3 * abs(u0) - u1 ** 3 * abs(u1)) / 4

    # --------------------------------------------------------
    # Blade-element analysis
    # --------------------------------------------------------

    rle = -(root + i * Rw)
    strip = (0.0, rle, 0.0)

    vle = _cross(omega_w + omega_b, strip) + vel_h
    vle[1] = 0

    # Previous-step element velocity for the added-mass finite difference. ``dt``
    # is the FD window (dt * substep); ``vel_h_prev`` lets the chordwise
    # acceleration include the hinge's translational acceleration, falling back to
    # ``vel_h`` for the old rotation-only behaviour.
    vh_prev = vel_h if vel_h_prev is None else vel_h_prev
    omega_wLI = omega_w - omega_w_d * dt
    vleLI = _cross(omega_wLI + omega_b, strip) + vh_prev
    vleLI[1] = 0

    acc_xw = (vle[0] - vleLI[0]) / dt
    omega_yw_dot = omega_w_d[1]

    # Velocity direction & folded angle of attack in [0, pi/2] (arctan2 of
    # absolute components is robust to vz == 0; lift/drag *direction* is set
    # separately below from the signs of vx and vz, so folding is fine for a
    # symmetric section).
    norm_v = np.sqrt(vle.dot(vle))  # np.linalg.norm's own formula
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
    # Flat-plate lift is the part of the normal force perpendicular to v whichever
    # edge leads, so it flips with the chordwise flow vz. sLift is vz's sign when
    # the hinge edge leads; it only breaks the vz == 0 tie.
    chord_sign = sLift if vle[2] == 0 else (1.0 if vle[2] > 0 else -1.0)
    dir_lift = chord_sign * (_LIFT_ROTATION[lift_sign] @ dir_vle)

    # --------------------------------------------------------
    # Translational lift & drag
    # --------------------------------------------------------

    q = 0.5 * rho * (norm_v ** 2)

    FDi = q * _cd(alpha, a, b, c, d) * chord * (Rw / NumBlade) * dir_drag
    FLi = q * _cl(alpha, e, f, g, h) * chord * (Rw / NumBlade) * dir_lift

    # The CoP and the rotational lift's 3/4-chord point are measured from the edge
    # that leads. When that is the far edge (vz against sLift), x -> 1 - x in chords
    # from the hinge edge.
    hinge_leads = chord_sign == sLift
    x_cop = ct1 * abs(alpha) / np.pi + ct2
    r_cof_trans = x_rot - (x_cop if hinge_leads else 1 - x_cop)
    r_cof_trans = min(max(r_cof_trans, x_rot - 1), x_rot)  # np.clip, minus its overhead
    r_cof_trans *= chord

    # --------------------------------------------------------
    # Rotational aerodynamic force
    # --------------------------------------------------------

    # Per strip: rho * Gamma * U * dr with dr = Rw / NumBlade, the width every term here uses.
    Circulation_rot = np.pi * ((0.75 if hinge_leads else 0.25) - x_rot) * (chord ** 2) * omega_w[1]
    F_rot_i = c4 * (Rw / NumBlade) * np.array([rho * Circulation_rot * norm_v, 0, 0])

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

    Fi = (FDi + FLi + F_add_i + sRot * F_rot_i) \
         + np.array([Fik_lt, 0, 0]) \
         + np.array([F_add2_i, 0, 0])

    Taui = (
        _cross((0.0, 0.0, sCof * r_cof_trans), FDi + FLi)
        + _cross((0.0, 0.0, sCof * r_cof_add), F_add_i + np.array([F_add2_i, 0, 0]))
        + _cross((0.0, 0.0, sCof * r_cof_rot), sRot * F_rot_i)
        + np.array([0, Tik_lt, 0])
    )

    if return_cop:
        return Fi, Taui, np.array([0.0, rle, sCof * r_cof_trans], dtype=float)

    return Fi, Taui


# Recompute the aero every DEFAULT_AERO_SUBSTEP physics steps and hold its
# generalized force in between (the finite-difference window scales with it, so it
# stays consistent). At the 0.13 ms step, 4 puts the aero cadence at ~0.52 ms --
# the rate that was fine at the old 0.5 ms step -- for ~4x fewer aero calls.
DEFAULT_AERO_SUBSTEP = 4

# The unsteady (added-mass) aero force is proportional to finite-differenced
# acceleration. In this explicit runtime it is noisy enough to inflate lift by
# 2x+ and does not transfer cleanly between CPU and MJX. Keep the default model
# quasi-steady; set_unsteady_aero(True) opts into the old capped hack for
# diagnostics. The MJX port reads these same caps.
# ponytail: explicit added mass stays opt-in; use an implicit added-mass state if
# this needs to become production physics.
OMEGA_DOT_CAP = 0.0     # rad/s^2
LIN_ACC_CAP = 0.0       # m/s^2


def set_unsteady_aero(enabled: bool) -> None:
    """Switch the added-mass (unsteady) aero term in both this and the MJX aero.

    Off (the default) clips the finite-difference accelerations to 0, which makes the
    rollout quasi-steady. MJX reads the caps at trace time, so this must run before
    a rollout is jitted.
    """
    global OMEGA_DOT_CAP, LIN_ACC_CAP
    OMEGA_DOT_CAP, LIN_ACC_CAP = (5.0e4, 1.0e3) if enabled else (0.0, 0.0)


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
    cached_lift: float = 0.0  # world-z sum of cached_forces


class AeroBladeRuntime:
    """Blade-element aero on both plate wings; call :meth:`apply` once per step, before mj_step.

    ``morph`` supplies the blade count and the wing mesh. Forces are recomputed every
    ``substep`` steps (default ``DEFAULT_AERO_SUBSTEP``); in between, the generalized
    force they made is added again unchanged, the hold the MJX rollout uses.
    """

    def __init__(self, model, morph, *, substep=None):
        self.num_blades = int(morph.wing.aero.blades)
        self.substep = max(1, int(DEFAULT_AERO_SUBSTEP if substep is None else substep))
        self.geometry = load_wing_geometry(morph)
        self.dt = float(model.opt.timestep)
        self.wings = [
            self._resolve_wing(model, "L"),
            self._resolve_wing(model, "R"),
        ]
        self._step_counter = 0
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._qfrc = np.zeros(model.nv)  # the held aero generalized force

    def apply(self, model, data) -> None:
        if self._step_counter % self.substep == 0:
            # After mj_step the kinematics (site_xpos/xmat, cdof, subtree_com) still sit
            # at the pre-step pose while qpos/qvel have moved on; bring them up to qpos so
            # the aero frames, Jacobians and points match the qvel it reads.
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            self._qfrc[:] = 0.0
            for wing in self.wings:
                self._compute_wing(model, data, wing)
        data.qfrc_applied += self._qfrc
        self._step_counter += 1

    def lift(self) -> float:
        """World-z aero force (N) of the forces the last :meth:`apply` applied."""
        return sum(wing.cached_lift for wing in self.wings)

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
        jacp, jacr = self._jacp, self._jacr
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
                vel_h_prev=velocity_local_prev,
                x_rot=float(self.geometry.x_rot[blade_index]),
                root=self.geometry.root_offset,
            )

            # force_local / torque_local are in the root-site frame; rotate both
            # with the root rotation and apply the force at the strip's site so
            # mj_applyFT supplies the spanwise moment (torque_local carries only
            # the chordwise pitching moment).
            forces.append(root_rotation @ force_local)
            torques.append(root_rotation @ torque_local)
            points.append(np.asarray(data.site_xpos[blade_site_id]).copy())
            mujoco.mj_applyFT(model, data, forces[-1], torques[-1], points[-1],
                              wing.wing_body_id, self._qfrc)

        wing.cached_forces = forces
        wing.cached_torques = torques
        wing.cached_points = points
        wing.cached_lift = float(np.sum([f[2] for f in forces]))

