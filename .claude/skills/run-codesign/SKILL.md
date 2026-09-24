---
name: run-codesign
description: Use when running the gradient co-design optimizer (opt/optimize_control.py) - a local smoke test after touching src/codesign or the MJX rollout, a real campaign on a vast.ai GPU box (rsync, CUDA jax, launch, monitor, fetch), reading its logs and output dirs, or promoting a winning design dir into designs/.
---

# Run the co-design optimizer

CLI: `opt/optimize_control.py` (argparse -> `codesign.train.run`). Reference:
`opt/README.md` (flags, outputs, Known issues) and `docs/ROBUST_CODESIGN_METHODOLOGY.md`
sections 6 (full campaign commands), 8 (vast.ai), 10 (judging artifacts).

## Local smoke test (proves the pipeline runs, nothing more)

```bash
uv run python opt/optimize_control.py --tg designs/hummer/hover-soft-it009p/shoulder.tg \
    --warmup-periods 0.2 --sample-periods 0.2 --iters-per-stage 1 --harmonics 2 \
    --scene /tmp/oc_scene.xml --out /tmp/oc
```

~15 s warm, up to a minute cold. On 0.2-period windows the metrics are garbage by design:
`tau_hat` ~37000 mNm, `best tau=inf`, `swing=nan` are expected. It ends with two
`[cpu/<integ>/mjx_best]` lines and writes `stage_<N>mNm/`, `theta_*.json`, `log.json`,
`cpu_validation.json` under `--out`. `export JAX_COMPILATION_CACHE_DIR=~/.cache/jax`
halves repeat runs. Always pass a distinct `--scene` when anything else may be running.

## Flags that matter

- `--tg`: default `designs/hummer/hover` (the seed). Warm-start from a previous run with
  `--init-codesign <dir>/theta_full.json` (the full genome; partial seeds lost motor
  pretension before).
- **`--lift-floor` (N): always pass it.** The 0.300 default is unreachable (the seed makes
  187-210 mN on the current aero), so it binds from step one. Pick it against the seed's
  measured lift (see the `certify-design` skill).
- Genome: `--tune-tendons` (`--lock-wings` to tune 12 mirrored pairs), `--tune-motor-tendons`,
  `--tune-pitch`, `--tune-strokeplane`, `--harmonics K`, `--control-after N` (tendons
  first, harmonics after), `--freeze-control`.
- Physics: `--integrator euler --timestep 0.06` (euler is where MJX and CPU agree per
  step), `--aero-substep` (default 4 = the CPU runtime's cadence), `--warmup-periods`
  >= 8 for real runs.
- Robustness: `--rotate-batch K --dt-bank ... --n-batches ... --full-check-every N
  --full-sample-periods 24 --drift-gate ...` -> `robust_best/`; `--cpu-every N
  --cpu-gap 0.15` -> `cpu_best/` (the honest deliverable, slow); `--holdout-dts`.
- `--single-stage` skips the 300->100 mNm torque curriculum.
- `--wavelet` output cannot be replayed or loaded (Known issue) - avoid for deliverables.

## On a GPU box (vast.ai)

The repo is private: sync with rsync, never `git clone`. Rent/destroy: methodology
section 8. Set `H=root@HOST`, `PORT`, `R=/workspace/DiffSimAvesBot`.

```bash
ssh -p $PORT $H "apt-get install -y rsync; mkdir -p $R"      # fresh boxes lack both
rsync -az --delete -e "ssh -p $PORT" --exclude .git --exclude .venv --exclude __pycache__ \
  --exclude data --exclude 'opt/runs*' ./ $H:$R/
ssh -p $PORT $H "cd $R && (command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh) \
  && export PATH=\$HOME/.local/bin:\$PATH && uv sync -q && uv pip install -q 'jax[cuda12]==0.10.2'"
```

Launch with `.venv/bin/python` (or `uv run --no-sync`), **never plain `uv run`**: it
re-syncs and drops the CUDA jax. The NVIDIA wheels need their lib path:

```bash
ssh -p $PORT $H "cd $R && export LD_LIBRARY_PATH=\$(find \$PWD/.venv/lib/python3.*/site-packages/nvidia \
  -path '*/lib' -type d | paste -sd:) && nohup sh -c '.venv/bin/python opt/optimize_control.py \
  <flags> --out opt/runs_NAME > run_NAME.log 2>&1; echo \$? > run_NAME.done' >/dev/null 2>&1 &"
```

- Check jax sees the GPU first: `.venv/bin/python -c 'import jax; print(jax.devices())'`.
- One optimizer per GPU (two crash cuSolver). Compiling the graphs takes 30-50 min on a
  GPU before the first `it000` line; stdout is block-buffered, so a silent log is not a
  hang - check `ps`/`nvidia-smi`.
- Kill with `pkill -f '[o]ptimize_control'`; plain `pkill -f optimize_control` kills your
  own ssh session. Keep kill and relaunch in separate ssh calls.
- Monitor: `ssh ... "tail -5 $R/run_NAME.log; cat $R/run_NAME.done 2>/dev/null"`.
- Fetch before destroying the box: `rsync -az -e "ssh -p $PORT" $H:$R/opt/runs_NAME opt/`
  (`opt/runs*/` is gitignored). Boxes get evicted: fetch checkpoints as you go.

## Reading the output

- Log: a `[baseline]` line (seed `Lbar`, `tau_hat`, `tau_peak`), `[target]` (lift floor),
  then one `itNNN` line per iteration: `L` mean lift, `tau` smooth peak, `pk` exact peak,
  `dr` per-cycle peak drift, `swing`, `P` power, `A` flap amplitude. `[stage ...] best`
  and the final `best design:` line name the dir they wrote; `[cpu/<integ>/...]` lines
  are the end-of-run CPU replay (also in `cpu_validation.json`, with `"holdout"`).
- Dirs under `--out` (layout: `src/codesign/artifacts.py` docstring): the root (running
  best), `stage_<N>mNm/`, `checkpoints/itNNN/` (every 3 iterations), `cpu_best/`,
  `robust_best/`, `*preharmonics/`; `log.json` has every iteration's metrics.
- The run's final pick (and the holdout) takes `robust_best` > `cpu_best` > the MJX best,
  whichever exists; `robust_best` is MJX-gated only, `cpu_best` is CPU-gated. Nothing is
  a result until it passes the `certify-design` skill on CPU, next to the seed certified
  the same way on the same code.

## Promoting a result into designs/

There is no promote command: `codesign.artifacts` writes a run dir and reads it back
(`reconstruct_design`), but the viewers and the GUI read only `.tg`, so bake the JSON in.

1. `mkdir designs/hummer/<name>`; copy the run dir's `shoulder.tg` (its `control =` already
   points at `control.tg`) and `control.tg` (`bird =` tag included; a/b absolute).
2. `tendon_optimized.json` `pairs` -> the `tendons:` `overrides:` block, one line per
   tendon: `<name> { stiffness: <stiffness> N/m, pretension: <P_src x pretension_factor> }`,
   where `P_src` is that tendon's pretension in the source `.tg` (its override, else the
   `defaults` one). Under `--lock-wings` the run's base was the L/R pair's mean rest length
   (`codesign.rig.symmetrize_wings`), so use `P_src = (P_L + P_R) / 2` for both twins and
   write the `R_` line even when an older dir lists only `L_` names.
3. `motor_tendon_optimized.json` -> `motors:` `tendon_stiffness = <stiffness> N/m`,
   `tendon_rest_length = <source rest x pretension_factor> mm`.
4. `pitch_optimized.json` -> in `wing ... pitch:` replace `free_angle` and `stiffness` with
   `upstroke_pitch = <x> rad`, `downstroke_pitch = <x> rad` (half-deadbands, as the JSON
   holds them), `upstroke_stiffness`, `downstroke_stiffness`.
5. Copy `strokeplane_optimized.json` if present (the viewers and GUI apply it; certification does not).
6. Verify the bake: `opt/design_io.py` on the run dir and on the new dir must print the
   same numbers (checked 2026-09-24 with perturbed tendon, motor and pitch JSON:
   identical to 0.1). Then certify, add a row to `designs/README.md`, commit.
