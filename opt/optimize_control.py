"""Gradient co-design of the hummer rig: Fourier crank command + tendons + pitch + stroke plane.

Optimizes a packed genome (codesign.genome) against a robust MJX reward (codesign.loss)
with forward-mode gradients and hand-rolled Adam, under a torque-budget curriculum or one
--single-stage, then CPU-certifies the winners. This file is the CLI; the run lives in
src/codesign/ (train.run). See opt/README.md.

    uv run python opt/optimize_control.py --tg designs/hummer/hover-soft-it009p/shoulder.tg \
        --timestep 0.06 --tune-tendons --single-stage ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codesign import train  # noqa: E402
from codesign.loss import OBJ_CHOICES  # noqa: E402
from codesign.rig import INTEGRATORS  # noqa: E402
from design_paths import DEFAULT_HUMMER_TG  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tg", type=Path, default=DEFAULT_HUMMER_TG)
    p.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    p.add_argument("--harmonics", type=int, default=5, help="K Fourier harmonics")
    p.add_argument("--wavelet", action="store_true",
                   help="reinterpret the 2K control coeffs as a dyadic Ricker WAVELET frame "
                        "(localized crank-speed atoms) instead of K Fourier harmonics. Same "
                        "genome width (2K+1), same robustness/drift machinery; warm-starts "
                        "clean from a Fourier sinusoid (all atom amplitudes -> 0).")
    p.add_argument("--lift-floor", type=float, default=0.300,
                   help="soft floor: penalize mean lift below this [N] (smooth softplus hinge)")
    p.add_argument("--lift-cap", type=float, default=None,
                   help="optional upper cap [N]; OFF by default (overshooting lift is free). "
                        "Set it to re-enable a moderate above penalty at --lift-above-weight")
    p.add_argument("--lift-smooth", type=float, default=0.015,
                   help="softplus smoothing scale [N] for the lift floor (larger = softer, more gradient)")
    p.add_argument("--warmup-periods", type=float, default=8.0,
                   help="settling periods before sampling (the rig keeps building lift for ~8 "
                        "cycles; 5 samples an unsettled transient, well below the true steady state)")
    p.add_argument("--sample-periods", type=float, default=3.0)
    p.add_argument("--aero-substep", type=int, default=4,
                   help="recompute aero every N steps, hold between (CPU parity; 1 = noisy every-step)")
    p.add_argument("--integrator", choices=list(INTEGRATORS), default="euler",
                   help="physics integrator; euler is the MJX<->CPU-matching choice (default)")
    p.add_argument("--timestep", type=float, default=None,
                   help="override physics timestep [ms] (default: the tg's, 0.13 ms). Shorter "
                        "dt improves euler stability -- pairs with --integrator euler")
    p.add_argument("--tau-min", type=float, default=0.100,
                   help="tightest torque-budget stage [Nm]; curriculum 0.30->0.25->0.20->0.15->0.10 "
                        "clamped to this. Stage 5 (100 mNm) is the demo target")
    p.add_argument("--added-mass", action="store_true",
                   help="enable the unsteady added-mass aero term. Now transfers MJX<->CPU "
                        "(ratio ~1.01) after the mj_jacSite velocity fix; off by default")
    p.add_argument("--iters-per-stage", type=int, default=40)
    p.add_argument("--cpu-every", type=int, default=0,
                   help="every N iters, re-run the CURRENT design on MuJoCo CPU (numpy aero) under "
                        "BOTH euler and implicitfast. 0 = off. Tracks a CPU-selected best scored on "
                        "the ROBUST (worse-integrator) peak torque among designs that TRANSFER "
                        "(see --cpu-gap), saved to <out>/cpu_best/ -- this is the integrator gate")
    p.add_argument("--cpu-gap", type=float, default=0.15,
                   help="transfer gate for --cpu-every: reject a design as cpu_best if euler and "
                        "implicitfast peak torque disagree by more than this fraction (default 0.15). "
                        "A large gap means the gait is exploiting the euler integrator")
    p.add_argument("--tri-integrator", action="store_true",
                   help="INTEGRATOR-AGNOSTIC reward: run euler + implicitfast + rk4 MJX rollouts "
                        "(all differentiable) each step and optimize the ROBUST peak torque "
                        "max(tau_e,tau_i,tau_r) plus a spread penalty (--w-gap). Makes integrator-"
                        "hacking unrewarding in the gradient itself, not just at selection. ~3x cost")
    p.add_argument("--w-gap", type=float, default=15.0,
                   help="weight on the cross-variant torque spread penalty ((max-min)/tau_max)^2 "
                        "(--tri-integrator / --tri-timestep / --rotate-batch)")
    p.add_argument("--tri-timestep", action="store_true",
                   help="TIMESTEP-AGNOSTIC reward (Fix 3, EXPERIMENT_LOG.org): also roll out euler "
                        "at dt/2 and 2*dt (differentiable in MJX) and fold their peak torque into the "
                        "same robust max + spread penalty as --tri-integrator. Confirmed need: seed 1 "
                        "transferred across integrators at dt=0.06ms but degraded 10-30%% off that exact "
                        "dt. Stacks with --tri-integrator (5-way robust max: 3 integrators x baseline dt, "
                        "plus euler @ dt/2, euler @ 2dt). ~2x extra cost on top of --tri-integrator")
    p.add_argument("--tri-timestep-factors", type=float, nargs="*", default=None,
                   help="dt factors used by --tri-timestep in the differentiable loss. Default: 0.5 2.0")
    p.add_argument("--rotate-batch", type=int, default=0, metavar="K",
                   help="ROTATING robust mini-batches: each iteration draws one of --n-batches "
                        "precompiled K-variant batches (variants = 3 integrators x --dt-bank) and "
                        "optimizes the robust envelope over just that batch (same make_loss math as "
                        "--tri-integrator, which this replaces). Stochastic cover of the full "
                        "integrator x dt matrix at ~K rollouts/iter instead of the whole matrix")
    p.add_argument("--dt-bank", type=float, nargs="*", default=None, metavar="MS",
                   help="training timestep grid [ms] for --rotate-batch (default: the run dt only)")
    p.add_argument("--n-batches", type=int, default=6,
                   help="number of precompiled rotating mini-batches for --rotate-batch")
    p.add_argument("--batch-seed", type=int, default=0,
                   help="RNG seed for mini-batch composition and rotation order")
    p.add_argument("--dt-jitter", type=float, default=0.0,
                   help="rotate mode: pre-sample each training batch dt as center*(1+u), "
                        "u~Uniform(-jitter,+jitter). Full-matrix checks stay on centers")
    p.add_argument("--precision-perturb", type=float, default=0.0,
                   help="rotate mode: add one differentiable noisy-state rollout per batch "
                        "with qvel noise sigma; cheap MJX/CPU sensitivity proxy")
    p.add_argument("--softmax-beta-start", type=float, default=50.0,
                   help="initial beta [1/unit] for smooth robust max/min aggregation")
    p.add_argument("--softmax-beta-end", type=float, default=300.0,
                   help="final beta [1/unit] for smooth robust max/min aggregation")
    p.add_argument("--full-check-every", type=int, default=25,
                   help="rotate mode: every N iters evaluate the FULL integrator x dt-bank matrix "
                        "on MJX; track a transfer-gated robust best (saved to <out>/robust_best)")
    p.add_argument("--full-gap", type=float, default=0.15,
                   help="transfer gate for the full-matrix check: max fractional peak-torque "
                        "spread across the whole matrix for a design to be eligible")
    p.add_argument("--full-sample-periods", type=float, default=None,
                   help="sample window [periods] for full-matrix checks and the MJX holdout "
                        "(default: --sample-periods). Set >= one torque-beat period (~24) so "
                        "slow transients can't hide from the gate")
    p.add_argument("--cpu-sample-periods", type=float, default=None,
                   help="sample window [periods] for the CPU gates and CPU holdout "
                        "(default: --sample-periods)")
    p.add_argument("--score-swing", type=float, default=1.0,
                   help="weight of the swing term in candidate SELECTION scores (loss keeps "
                        "--w-swing); swing is converted to the objective's units as "
                        "swing / baseline tau_hat x baseline objective. Set 0 to rank promotions purely on certified worst-case "
                        "peak torque -- the right KPI for a current-limited motor")
    p.add_argument("--w-drift", type=float, default=0.0,
                   help="weight on the per-cycle peak-torque drift penalty in the LOSS "
                        "(linear in worst |drift|/tau_max across batch variants). Makes "
                        "beat-riding expensive in the gradient itself instead of only at "
                        "the selection gates; ~20 gives it real authority")
    p.add_argument("--drift-gate", type=float, default=0.03,
                   help="stationarity gate: reject a candidate when any gate variant's "
                        "per-cycle peak-torque drift (last minus first sampled cycle) exceeds "
                        "this fraction of the robust peak -- catches beat-transient parking, "
                        "where a 3-cycle window reports the minimum of a ~40-cycle beat")
    p.add_argument("--warmup-bank", type=float, nargs="*", default=None, metavar="P",
                   help="rotate mode: per-batch-member warmup periods sampled from this set "
                        "(e.g. 8 16 24 32). Randomizing where the sample window lands prices "
                        "slow torque beats into the robust envelope; default: --warmup-periods")
    p.add_argument("--holdout-dts", type=float, nargs="*", default=(), metavar="MS",
                   help="UNSEEN timesteps [ms] evaluated only at the END on the winner (3 "
                        "integrators on MJX + euler/implicitfast on CPU). Never trained on -- "
                        "passing these means the design learned physics, not a dt trick")
    p.add_argument("--control-after", type=int, default=0,
                   help="zero the control-harmonic gradient for the first N iters of each stage: "
                        "tendon stiffness/pretension + motor preload first (they generalize), "
                        "harmonics after (they are where timestep hacks live)")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weights", type=float, nargs=4, default=(10.0, 10.0, 8.0, 0.01),
                   metavar=("wL", "wtau", "wK", "ws"))
    p.add_argument("--w-power", type=float, default=5.0,
                   help="weight on inactive (reactive/braking) motor power -- pushes elastic "
                        "tendon storage to handle decel instead of the motor (resonance). A "
                        "primary objective now; dial after seeing Pin in the log")
    p.add_argument("--w-overlap", type=float, default=5.0,
                   help="weight on the two-motor torque-overlap penalty (t1 'blue' vs t4 "
                        "'orange'): cosine similarity of their |torque| traces. Drives each "
                        "motor to peak in the other's trough (anti-phase) to cut wasted work")
    p.add_argument("--bank-integrators", nargs="+", default=("euler", "implicitfast", "rk4"),
                   choices=("euler", "implicitfast", "rk4"),
                   help="integrators in the --rotate-batch bank (default all three). rk4 is "
                        "4x the physics per step and, inside a jacfwd over the whole genome, "
                        "took ~50 min just to XLA-compile one batch on a 5070 Ti -- drop it "
                        "(`--bank-integrators euler implicitfast`) when compile time is the "
                        "binding constraint. The CPU transfer gate still rolls rk4 (numpy, "
                        "no compile), so cross-integrator certification is unchanged; only "
                        "the GRADIENT stops seeing it")
    p.add_argument("--a0-free", action="store_true",
                   help="keep a0 (theta[0] = crank speed = flap frequency) live during the "
                        "--control-after freeze, so only the HARMONICS wait for the unlock. "
                        "Frequency is the dominant power knob (P~f^3, L~f^2): this gives it "
                        "the whole run while keeping the pre/post-unlock diff purely about "
                        "waveform shape")
    p.add_argument("--objective", choices=OBJ_CHOICES, default="torque",
                   help="primary minimization target. 'torque' = robust peak motor torque "
                        "(original behavior). 'power' = robust mean SUPPLIED motor power "
                        "(an exact window mean, not an Lp peak). 'energy' = robust "
                        "SPECIFIC power P/L [W/N]: the same watts, credited by the lift they "
                        "buy, with the denominator floored at --lift-floor so it cannot be "
                        "won by not flapping. Under power/energy peak torque is demoted to a "
                        "soft ceiling guard (--tau-guard); loss, selection scores and the "
                        "full/cpu transfer gates all follow the chosen metric. "
                        "'swing' = robust motor torque swing (std). Picked by "
                        "the Sep-10 objective screen: of every candidate, swing is the one the "
                        "SHARED FOURIER WAVEFORM both moves hard (3258%%/unit vs 203%% for a0) "
                        "and moves in a direction all 6 dt/integrator cells agree on "
                        "(consensus 0.996). Peak torque has the same authority with "
                        "consensus 0.39 -- same knob, but each discretization wants it "
                        "pushed a different way, which is what MJX-overfit looks like from "
                        "the inside. Power is nearly waveform-deaf (15%%/unit).")
    p.add_argument("--tau-guard", type=float, default=4.0,
                   help="--objective power/energy only: weight of the soft peak-torque ceiling "
                        "at the stage tau_max (protects a current-limited motor). "
                        "0 = torque completely out of the loss")
    p.add_argument("--lock-wings", action="store_true",
                   help="one design for BOTH wings: geometric-mean the L/R structural tendon "
                        "stiffness / rest length at load, and tune the 12 MIRRORED pairs (24 "
                        "params) instead of 24 independent tendons (48). Without it the two "
                        "wings drift into different machines sharing one command signal")
    p.add_argument("--lift-torque-tradeoff", type=float, default=None, metavar="N",
                   help="put lift back in the objective as a controlled trade: an N%% peak-torque "
                        "cut is worth a 1%% lift drop below baseline (N=5 -> 5:1). Linear marginal "
                        "rate; also gates the --single-stage winner. Off by default (lift ignored, "
                        "as with --weights wL=0)")
    p.add_argument("--amp-band-pct", type=float, default=0.02,
                   help="fractional +/- band around the baseline stroke amplitude A* (wK bites "
                        "on EITHER side, not just below). Default 2%% (was floor-only before)")
    p.add_argument("--amp-floor", type=float, default=None, metavar="M",
                   help="ABSOLUTE stroke-amplitude floor [m] replacing the +/- band: a hard "
                        "one-sided wall (weight wK) + feasibility gate below it; growth is free")
    p.add_argument("--w-swing", type=float, default=0.0,
                   help="weight on the torque-SWING penalty (temporal std of the two drive-group "
                        "torque envelopes, normalized by the budget). The v3 demo objective: a "
                        "flat low-torque profile, not just a low peak. Set >0 to enable")
    p.add_argument("--cpu-timestep-factors", type=float, nargs="*", default=(),
                   help="extra timestep factors for --cpu-every transfer checks, e.g. 0.5 2.0")
    p.add_argument("--single-stage", action="store_true",
                   help="v3: skip the torque curriculum -- one stage, normalize peak torque by the "
                        "baseline, and SELECT the design with the lowest peak+swing that keeps the "
                        "baseline flap amplitude (instead of lowest-tau clearing the lift floor)")
    p.add_argument("--w-lc", type=float, default=0.0,
                   help="weight on limit-cycle stability (cross-cycle lift variance); logged either "
                        "way. Default 0: gradient descent doesn't reduce it (chaos-corrupted variance "
                        "gradient), confirmed empirically")
    p.add_argument("--tune-frequency", action="store_true",
                   help="optimize the drive frequency (log-factor) -- the resonance lever for torque")
    p.add_argument("--tune-tendons", action="store_true",
                   help="optimize every structural tendon's stiffness + pretension (48 params; "
                        "24 mirrored with --lock-wings)")
    p.add_argument("--tendon-params", choices=("both", "stiffness", "pretension"), default="both",
                   help="restrict --tune-tendons to one half of the structural-tendon block; the "
                        "frozen half stays at its seed (--init-codesign, else 1x)")
    p.add_argument("--tune-pitch", action="store_true",
                   help="optimize the 4 wing-pitch params (up/down deadband + stiffness) -- feathering")
    p.add_argument("--tune-motor-tendons", action="store_true",
                   help="optimize the 2 motor-tendon params (shared stiffness + pretension)")
    p.add_argument("--tune-strokeplane", action="store_true",
                   help="learn the crank-disk mount orientation over the wingbeat: per DOF "
                        "(tilt, pitch) a mean + 2 Fourier harmonics of the L_t1 crank angle "
                        "(10 params), tanh-bounded by --sp-max. Kinematic stroke-plane control")
    p.add_argument("--sp-max", type=float, default=0.3,
                   help="stroke-plane bound [rad]: tilt/pitch deviation from the compiled "
                        "mount angles is squashed to +/- this via tanh")
    p.add_argument("--sp-lr-mult", type=float, default=1.0,
                   help="learning-rate multiplier for the 10 stroke-plane params. Adam steps "
                        "each param ~lr/iter regardless of gradient size, and lr is tuned for "
                        "log-factors; radians need a bigger stride or they ramp invisibly "
                        "(the run-7 a0/harmonics slow-ramp trap)")
    p.add_argument("--freeze-control", action="store_true",
                   help="hold the control signal fixed (use with --init-control)")
    p.add_argument("--init-control", type=Path, default=None,
                   help="warm-start control from a phase-1 theta_best.json")
    p.add_argument("--init-codesign", type=Path, default=None,
                   help="seed the genome from an evolve_codesign genome (best_genome.json, "
                        "codesign_seed.json) or a run's theta_full.json, group by group")
    p.add_argument("--out", type=Path, default=ROOT / "opt" / "runs")
    args = p.parse_args(argv)
    if args.dt_jitter < 0.0:
        p.error("--dt-jitter must be >= 0")
    if args.precision_perturb < 0.0:
        p.error("--precision-perturb must be >= 0")
    if args.softmax_beta_start <= 0.0 or args.softmax_beta_end <= 0.0:
        p.error("--softmax-beta-start/end must be > 0")
    if args.lift_torque_tradeoff is not None and args.lift_torque_tradeoff <= 0.0:
        p.error("--lift-torque-tradeoff must be > 0")
    if args.freeze_control and not (args.tune_tendons or args.tune_pitch
                                    or args.tune_motor_tendons or args.tune_strokeplane):
        p.error("nothing to optimize")
    if args.tri_timestep and any(f <= 0 for f in args.tri_timestep_factors or ()):
        p.error("--tri-timestep-factors must be positive")
    if args.rotate_batch and (args.tri_integrator or args.tri_timestep):
        p.error("--rotate-batch replaces --tri-integrator/--tri-timestep")
    if args.rotate_batch and args.rotate_batch < 2:
        p.error("--rotate-batch needs >=2 variants per batch")
    return args


if __name__ == "__main__":
    train.run(parse_args())
