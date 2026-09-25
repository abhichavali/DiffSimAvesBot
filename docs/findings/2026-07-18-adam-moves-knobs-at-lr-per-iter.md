---
title: Adam moves every gene about lr per iteration, so a knob that must travel far cannot get there within a run
date: 2026-07-18
status: current
tags: [codesign]
---

## Finding
The codesign optimizer is Adam (hand-rolled in `src/codesign/train.py`), which normalizes each
parameter's step by its own gradient scale: a stronger gradient cannot make a knob move faster.
Each gene moves ~lr per iteration.

## Evidence
- Stroke-plane run v3 (2026-07-18): after unlock the loss flatlined and the gradient looked dead;
  every stroke-plane coefficient had moved exactly ~lr per iteration (8e-4 rad/iter = 0.05 deg
  after 2 iterations, invisible).
- Run 7: a0 climbed at "full Adam speed", 65.07 -> 65.29 rad/s by it011; harmonics likewise.
- Crank lag: 6 -> 12 deg is out of reach within a run (`crank-lag-is-a-lift-knob`).

## Consequences
- Before calling a flat loss after an unlock a dead gradient, compute lr x iterations for the
  knob that should move.
- A knob that needs a large move: sweep it globally and bake it into the .tg, then codesign the
  rest (coordinate descent), or scale its Adam STEP (`--sp-lr-mult` for the stroke-plane tail).
  Scaling its gradient does nothing.

## Status notes
Current: a property of Adam and of the optimizer in the tree.
