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

The project skills (`.claude/skills/`) are listed in AGENTS.md "Skills"; Claude Code
loads them by name.

- **Docs are part of the change** (AGENTS.md): before the final commit of a decently
  sized change, update the doc that owns each fact you changed.
- Shell cwd resets between Bash calls: use absolute paths, or `cd <repo> && ...` in one call.
- Viewers and Tk GUIs open real windows and block; run them with `run_in_background`, or
  use the headless paths (`--self-check`, `--gif`, `src/demos/render_*.py`).
- Long runs (golden full tier, certification, optimizer) belong in `run_in_background`;
  keep going and read the log when it finishes.
- A tokensave index (`.tokensave/`) may describe another checkout or worktree; when its
  results disagree with the files, trust the files (grep / Read).
