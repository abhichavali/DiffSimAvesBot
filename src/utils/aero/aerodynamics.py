import numpy as np

# ------------------------------------------------------------
# Import helper functions from separate modules
# ------------------------------------------------------------
try:
    from .ry_rad import ry
    from .CD_hovering import CD_hu
    from .CL_hovering import CL_hu
except ImportError:  # pragma: no cover - supports direct execution from this folder
    from ry_rad import ry
    from CD_hovering import CD_hu
    from CL_hovering import CL_hu


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
    dir_lift = ry(-lift_sign * np.pi / 2) @ dir_vle

    # --------------------------------------------------------
    # Translational lift & drag
    # --------------------------------------------------------

    q = 0.5 * rho * (norm_v ** 2)

    FDi = q * CD_hu(alpha, a, b, c, d) * chord * (Rw / NumBlade) * dir_drag
    FLi = q * CL_hu(alpha, e, f, g, h) * chord * (Rw / NumBlade) * dir_lift

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
