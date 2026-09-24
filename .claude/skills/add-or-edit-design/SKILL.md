---
name: add-or-edit-design
description: Use when creating, copying or editing a .tg design (designs/<bird>/<name>/shoulder.tg + control.tg) - changing geometry, tendons, per-tendon overrides, motors, wing pitch or the control harmonics, adding a new design dir, fixing a .tg parse error, or changing which design a tool opens by default.
---

# Add or edit a design

A design is a dir `designs/<bird>/<name>/` with `shoulder.tg` (the WHOLE morphology; the
name is legacy) and `control.tg` (the Fourier crank-velocity command). The dir name is
the design name. `designs/README.md` lists every design, its lineage and certified
numbers. Never hand-edit a compiled `scene.xml`: edit the `.tg` and recompile.

## The .tg dialect (parser: `src/morphology/tg_parser.py`, docstring = the spec)

- Line-oriented; `#` comments; top-level `name =`, `bird =`, `strut ...` beside indented
  sections: `world:`, `archetype strut C:`, `arrange M relative_to F:`, `placement:`,
  `tendons:`, `motors:`, `wing on M.midpoint:` (with `pitch:` and `aero:` sub-blocks).
- **Units ride on the literal**: `mm`, `cm`, `g`, `ms` are scaled to SI; `m`, `s`, `kg`,
  `N/m`, `rad` are taken as written; an angle converts to radians only when its literal
  says `deg`. Exceptions by design: `stations = [...] deg` and `arrange.angle` stay in
  degrees. Write the unit every time (`crank_throw_t4 = 25 mm`, not `0.025`).
- **Strict**: unknown keys, unparseable lines, brace parts without `:` and overrides that
  name no tendon raise (`ValueError: Unrecognized motor line: crank_trow_t4 = 25 mm`).
  Only dead keys old files carry (`force_scale`, `body_torque`, `cop`, disk `outboard`,
  archetype `shape`/`symmetry`) are accepted and ignored: do not add them to new files.
- **Bird tag**: both files start with `bird = hummer` (plate wing) or `bird = eagle`
  (three-linkage bone wing) and must agree; tools refuse the wrong bird
  (`morphology.model.require_bird`).
- **Tendons**: `defaults { stiffness: 200 N/m, damping: 0.5, pretension: 0.827, width: 0.75 mm }`,
  `length = derived`, six `generators under sigma:` edges `M[1A] -- F[4B]` (station +
  face anchors) expand to 12 structural tendons; the R side mirrors L's geometry.
  Rest length = posed anchor distance x pretension: never write a length.
- **Overrides** are per tendon, by compiled name, L and R separately:
  `overrides:` then `L_t1_to_t1_2 { stiffness: 184.4553 N/m, pretension: 0.743129 }`.
  There are no per-sigma-family overrides. See `designs/hummer/hover-soft-it009p/shoulder.tg`.
- **Motors**: `tendon_rest_length` / `tendon_stiffness` (motor tendons), `crank_phase_lag`,
  `control = control.tg` (resolved beside the `.tg`); `meshdir = assets` resolves
  against the repo root.
- **Wing pitch**: `free_angle` (whole dead band) + `stiffness`, or the asymmetric
  `upstroke_pitch` / `downstroke_pitch` (half-bands) + `upstroke_stiffness` /
  `downstroke_stiffness`.
- **control.tg**: `bird`, `harmonics = K`, `fundamental_frequency = <Hz> Hz` (must equal
  a0 / 2pi: the crank is a velocity servo), `a0` (rad/s), `a_n` / `b_n` absolute.
  Harmonics are phased on a clock that starts at rollout start (after the rest settle).

## Workflow

```bash
cp -r designs/hummer/hover designs/hummer/<name>          # start from the nearest design
# edit shoulder.tg / control.tg (hover_policy/ and _recon_scene.xml need not come along)
uv run python -c "import sys; sys.path.insert(0,'src'); from morphology.tg_parser import load_morphology as L; m=L('designs/hummer/<name>/shoulder.tg'); print(m.bird, m.name)"
uv run python src/compile_tg.py --tg designs/hummer/<name>/shoulder.tg --out /tmp/<name>.xml
uv run python src/run_fixed_frequency.py --tg designs/hummer/<name>/shoulder.tg --periods 5 --scene /tmp/<name>_ff.xml
uv run pytest -q tests/test_tg_parser.py tests/test_tendon_overrides.py tests/test_bird_tags.py
```

`compile_tg.py` with no `--tg` compiles `legacy-baseline`, not the seed: always pass `--tg`.
Every tool takes `--tg`; nothing reads a design from the environment. Then certify it
(the `certify-design` skill) and add a row to `designs/README.md` (what it is, lineage,
certified peak and lift with the date, what defaults to it).

## Tool defaults

Tool defaults live only in `src/design_paths.py`: `DEFAULT_HUMMER_TG` (`hummer/hover`:
optimizer, viewers, GUI, renders, hover_policy, runners), `HOVER_BASE_TG` (EA, run_gui),
`LEGACY_BASELINE_TG` (compile_tg), `EAGLE_TG`. Repoint a default there, never with a
literal path in a tool, and update the "default for" column of `designs/README.md`.

## Golden consequences

The golden harness parses/compiles designs by logical name (`PATHS` in
`tests/golden/harness.py`): `shoulder`, `tuned`, `it009p` -> `hummer/hover`; `baseline`
-> `legacy-baseline`; `soft_it009p`; `hovering` -> `hover-base`; `tuned_hacked`; `eagle`.
Adding a design touches none of them. Editing one of those files changes its items: that
is a behavior change, recaptured by the rules of the `golden-harness` skill. Moving one
means editing only its `PATHS` entry.
