@AGENTS.md

## Claude Code notes

Everything about the project lives in `AGENTS.md` (imported above); this file only adds
what is specific to Claude Code.

Skills worth reaching for here:

| skill | when |
|---|---|
| `/code-review` | before committing a physics or codesign change |
| `/simplify` | after a change, to cut what it does not need |
| `/run` | to launch a runner, viewer or the GUI and see a change working |

Project skills (`.claude/skills/`, loaded by Claude Code; plain Markdown, so other agents
can read them too):

| skill | when |
|---|---|
| [`certify-design`](.claude/skills/certify-design/SKILL.md) | CPU doctrine numbers for a design or optimizer result; per-cycle lift (bistable pitch), stationarity |
| [`run-codesign`](.claude/skills/run-codesign/SKILL.md) | optimizer smoke test, GPU-box runs, reading outputs, promoting a result into `designs/` |
| [`golden-harness`](.claude/skills/golden-harness/SKILL.md) | check / capture rules, ADAPTER and `PATHS`, what a recapture commit says |
| [`add-or-edit-design`](.claude/skills/add-or-edit-design/SKILL.md) | `.tg` dialect, strict parser, bird tags, overrides, tool defaults |

- Shell cwd resets between Bash calls: use absolute paths, or `cd <repo> && ...` in one call.
- Viewers and Tk GUIs open real windows and block; run them with `run_in_background`, or
  use the headless paths (`--self-check`, `--gif`, `src/demos/render_*.py`).
- Long runs (golden full tier, certification, optimizer) belong in `run_in_background`;
  keep going and read the log when it finishes.
- A tokensave index (`.tokensave/`) may describe another checkout or worktree; when its
  results disagree with the files, trust the files (grep / Read).
