---
name: gpu-box
description: Use when a job needs a vast.ai GPU box - renting or reconnecting to one, syncing the private repo onto it, installing CUDA jax, launching and monitoring a long run detached, fetching results, and destroying the box. Applies to any GPU job (co-design optimizer, MJX hover training, MJX sweeps).
---

# Work on a vast.ai GPU box

Boxes are rented per job and get evicted without warning: treat everything on them as
disposable and fetch results as you go. The user usually hands you the ssh command
(`ssh -p PORT root@HOST -L 8080:localhost:8080`); otherwise rent one below.

## Rent, reconnect, destroy

```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 reliability>0.97 cuda_vers>=12.4 \
  inet_down>200 rentable=true' -o dph+ --raw                   # cheapest sane offer first
vastai create instance <OFFER_ID> --image vastai/base-image:cuda-12.8.1-auto --disk 64 --ssh --direct
vastai show instances; vastai ssh-url <ID>                     # -> ssh://root@HOST:PORT
yes | vastai destroy instance <ID>                             # ask the user first: it deletes the disk
```

- Test the direct endpoint (`vastai ssh-url`) and the `sshN.vast.ai` proxy: usually only
  one works.
- The local `vastai` API key can be stale (404). The box has a working one in its
  environment: run `vastai` there with `--api-key $CONTAINER_API_KEY`.
- Iterate on a cheap box (~$0.10-0.30/h); rent the big one only for the confirmed run.

## Sync the repo (private: never `git clone` on the box)

In zsh, a `$SSH` holding `ssh -p PORT root@HOST` does not word-split: use a function,
`box() { ssh -p $PORT root@$HOST "$@"; }`. `R=/workspace/DiffSimAvesBot`.

```bash
box "mkdir -p $R"                                   # fresh boxes have no /workspace
# macOS ships openrsync, which fails against the box's rsync: stream a tarball instead
COPYFILE_DISABLE=1 tar --no-mac-metadata -czf - --exclude .git --exclude .venv \
  --exclude __pycache__ --exclude 'opt/runs*' --exclude '*.log' . | box "tar xzf - -C $R"
```

With a real rsync (`brew install rsync`): `rsync -az -e "ssh -p $PORT" --exclude .git
--exclude .venv --exclude __pycache__ --exclude 'opt/runs*' --exclude 'run_*.log'
--exclude 'run_*.done' ./ root@$HOST:$R/`. Do not add `--delete` without those excludes:
it wipes a live run's log.

## Python and CUDA jax

```bash
box "cd $R && (command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh) && \
  export PATH=\$HOME/.local/bin:\$PATH && uv sync -q && uv pip install -q 'jax[cuda12]==0.10.2'"
box "cd $R && export LD_LIBRARY_PATH=\$(find \$PWD/.venv/lib/python3.*/site-packages/nvidia \
  -path '*/lib' -type d | paste -sd:) && .venv/bin/python -c 'import jax; print(jax.devices())'"
```

- Run with `.venv/bin/python` (or `uv run --no-sync`), **never plain `uv run`**: it
  re-syncs and drops the CUDA jax.
- Every launch needs that `LD_LIBRARY_PATH` (the NVIDIA wheels' libs).
- If PyPI is unreachable from the box, `pip download` the wheels on the Mac
  (`--platform manylinux_2_28_x86_64 --only-binary=:all:`), copy them over, and
  `uv pip install --no-index --find-links <dir>`.

## Launch, monitor, fetch

```bash
box "cd $R && export LD_LIBRARY_PATH=... && setsid nohup sh -c '.venv/bin/python <script> <flags> \
  > run_NAME.log 2>&1; echo \$? > run_NAME.done' >/dev/null 2>&1 < /dev/null &"
box "tail -5 $R/run_NAME.log; cat $R/run_NAME.done 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu --format=csv"
box "tar czf - -C $R opt/runs_NAME" | tar xzf -                 # fetch (opt/runs*/ is gitignored)
```

- **Every job runs detached** with the recipe above, even a "quick" one: never run work in a
  foreground ssh call. Closing the local agent session, or a dropped connection, must not end
  it. Only short status/fetch commands go in foreground ssh calls.
- `.done` holds the exit code. stdout is block-buffered and XLA compiles take minutes
  (30-50 min for a rotating-batch optimizer): a silent log is not a hang. Check `ps` and
  `nvidia-smi`.
- One JAX job per GPU (two crash cuSolver).
- Kill with `pkill -f '[o]ptimize_control'`: a plain `pkill -f <name>` matches, and kills,
  your own ssh command. Keep the kill and the relaunch in separate ssh calls.
- `&` inside `box "cd X && ... &"` backgrounds only the last command; wrap the whole
  job in `nohup sh -c '...'` as above.
- Fetch checkpoints during the run, not only at the end.
