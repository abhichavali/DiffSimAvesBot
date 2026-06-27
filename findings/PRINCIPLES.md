# Learned principles

Distilled, transferable lessons from optimizing this differentiable flapping-wing design.
Ordered by how much pain they caused / time they saved.

## On differentiable-sim optimization

1. **Optimize in the simulator you validate in — or make them bit-faithful first.**
   The entire effort stalled until MJX and CPU MuJoCo produced the *same* numbers. An
   optimizer that's faithful to a reference is worth more than a "better" sim that drifts.

2. **A 2× discrepancy is a bug, not chaos.** When two sims of the *same* model disagree by
   a clean factor, suspect a per-step force/cadence/implementation difference before
   blaming sensitivity or precision. Localize by comparing **trajectories step-by-step**
   (state first, then forces): here the dynamics matched perfectly and only the aero force
   diverged → the bug was in one aero term, not the physics.

3. **Unsteady/added-mass aero is treacherous.** It's a finite-difference acceleration term;
   tiny FD windows explode it, caps make it implementation-dependent, and two reasonable
   implementations diverged 4× *with opposite sign*. If two sims must agree, disable it or
   share one exact implementation. The quasi-steady (velocity²) lift is the robust quantity.

4. **Integrators are not interchangeable across frameworks.** `implicitfast` does an implicit
   solve that JAX and C implement differently → per-step divergence. **Euler matches** (plain
   explicit update) and was stable here. Match the integrator before trusting gradients.

5. **Aero substep/cadence matters.** Recompute-and-hold every N steps with an N·dt FD window
   is *not* the same as every-step; mismatching it across sims gave a 2× lift error.

## On parameterization (the optimizer's real bottleneck)

6. **Scale your parameters to your learning rate.** A single LR can't move a param of
   magnitude 60 and one of magnitude 0.01 at once. Use **log-factors** for positive,
   large-scale quantities (a0, frequency, stiffness, pretension) and **relative/modulation**
   encodings for perturbations. This single change turned a 7 %/100-iter optimizer into a
   34 %/16-iter one — the "weak" optimizer was a scaling artifact.

7. **The control harmonics were the dominant lever** once they could actually move. Don't
   assume a term is unhelpful before confirming its parameters are mobile.

8. **Forward-mode AD (jacfwd) is the right tool for few-parameter rollouts** (constant
   memory through thousands of steps; keeps the full-accuracy solver). Reverse-mode would
   need a 1-iteration solver and store the whole trajectory.

## On the reward function

9. **Only constrain what the sim measures faithfully.** Drop terms you can't measure cleanly
   (forearm/lateral amplitude) rather than fake a target.

10. **Curricula can over-constrain.** Forcing torque too low collapsed lift; making the
    tightest budget a *parameter* (and finding the resonant `τ_max=100` sweet spot) was the
    fix. Tighter isn't always the optimum.

11. **Some objectives don't gradient-optimize.** Cross-cycle variance (`C_lc`) barely moved
    under gradient descent (corrupted variance gradient); the *real* stabilizer was fixing
    the aero noise (added-mass), not weighting the term up.

12. **Penalizing inactive (reactive/braking) motor power drives resonance.** `mean(max(0,−τ·ω))`
    pushes elastic storage to absorb deceleration, yielding more lift-efficient gaits — lift
    rose 201→222 mN as `w_power` increased, with peak torque flat. This was the term that
    converted "good" designs into the best ones.

## On process

13. **Gate everything on the validation metric (CPU), not the optimizer's self-report.**
    The autonomous sweep only trusted designs that CPU-validated; that's what caught the
    overfit-to-MJX designs.

14. **Checkpoint per stage and run detached.** A non-detached remote run lost 233/240 iters
    to an SSH blip; `setsid` + per-stage checkpoints + a resumable driver fixed it.

15. **Don't thrash on a converged optimum.** Once a strong Pareto frontier is mapped,
    promote the best balanced point and watch lightly; flailing the config wastes GPU.
