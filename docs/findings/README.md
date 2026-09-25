# Findings

One file per durable piece of project knowledge: a result, a decision and its reason, a
negative result (something tried that did not work), or a gotcha that cost real time.
This is the knowledge the code cannot tell you. The `librarian` agent
(`.claude/agents/librarian.md`) searches it, so agents don't have to read it wholesale.

List them (no hand-kept index to go stale):

```bash
grep -H -m1 '^title:' docs/findings/2*.md            # every finding, in date order
grep -l '^status: current' docs/findings/2*.md       # only the ones still true today
grep -l 'free-flight' docs/findings/2*.md            # by tag
```

## Rules

- Name: `YYYY-MM-DD-<slug>.md`, dated when the finding was established. Other docs refer to
  a finding by its slug in backticks.
- Status:
  - `current`: still true on today's code;
  - `re-check`: the conclusion probably holds but its numbers predate a physics change
    (e.g. the 2026-09 aero and harmonic-clock fixes) and were not re-measured;
  - `superseded`: overturned. Keep the file (the history of what fooled us is the point),
    set `superseded_by:` to the slug or reason, and say what survives.
- When a change overturns or re-measures a finding, update its status in the same commit.
- Evidence is concrete: commits, commands, numbers with units, where the artifacts are.
- 15-40 lines. No environment details (hosts, keys, box ids).

## Template

```markdown
---
title: <the finding as one claim>
date: YYYY-MM-DD
status: current | re-check | superseded
superseded_by: <slug or short reason>   # only if superseded
tags: [<few of: aero, codesign, certification, harmonics, mjx, cpu, tendons, free-flight, gui, tooling, design>]
---

## Finding
## Evidence
## Consequences
## Status notes
```
