@AGENTS.md

## Claude Code notes

Everything about the project lives in `AGENTS.md` (imported above); this file only adds
what is specific to Claude Code.

## Keep your context clean: ask the librarian

Your context is the scarce resource; don't spend it reading docs. The `librarian` agent
(`.claude/agents/librarian.md`, read-only, cheap) reads `AGENTS.md`, `docs/findings/`, the
skills, the READMEs, docstrings and git history for you and returns a short answer with
`path:line` citations.

- **Ask it** why something is the way it is, whether X was tried and what happened, a
  current number or status, which doc / skill / flag / command covers something, and,
  before any non-trivial task, "I'm about to <task>: which findings, gotchas and known
  bugs apply?". Put all your questions in one call.
- **Don't ask it** about code structure (use tokensave) or about a file you are about to
  edit (read that yourself; you need the exact text anyway).
- **Don't read** `docs/findings/`, `docs/history/`, `opt/README.md` or other long docs
  yourself to answer a question. Read a cited line directly only when you need its exact
  text, or when a wrong answer would be costly to act on.
- Its answers are leads with citations, not ground truth: docs can be stale. If you find
  one wrong, fix the doc (AGENTS.md "Documentation is part of the change").

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
