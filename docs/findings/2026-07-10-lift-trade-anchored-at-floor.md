---
title: The lift-torque trade must be anchored at the lift floor, not the seed's lift, or every run hoards lift
date: 2026-07-10
status: current
tags: [codesign]
---

## Finding
With `--lift-torque-tradeoff N`, a 1% lift drop below the reference costs N% of peak torque.
When the reference was the seed's own lift, spending surplus lift to soften the tendons was
score-negative, so every run hoarded lift and never entered the soft corridor
(`soft-tendon-torque-corridor`). Anchored at `--lift-floor`, lift above the floor is free to
spend.

## Evidence
- `docs/history/codesign_campaign_2026-07.md` §12 "Reward bug"; fixed before run 4.
- Code: `src/codesign/train.py` `lift_ref = cfg.lift_floor if cfg.lift_torque_tradeoff else m0.Lbar`.

## Consequences
- The trade only makes sense with a meaningful floor. `--lift-floor` defaults to 300 mN, which
  is unreachable now that the seed lifts 187-210 mN (AGENTS.md Known bugs), so with the default
  the floor penalty binds from step one: always pass `--lift-floor`, chosen against the seed's
  measured lift (run-codesign skill).
- Any lift credit above the reference is its own hazard: `reward-lift-credit-fountain`.

## Status notes
Current (the anchoring is in the code today).
