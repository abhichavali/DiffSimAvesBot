"""Flatten a .tg design file into named scalar knobs, and write them back.

Shared by opt/run_gui.py (to list what is tunable) and opt/evolve_codesign.py
(to materialize a geometry genome as a compilable .tg). Deliberately a text
layer, not a parse of src.morphology.tg_parser's Morphology: the EA has to hand
compile_tg a *file*, and round-tripping a typed tree back to source would have
to re-emit every construct the format has. Rewriting the number in place keeps
comments, units, and layout byte-identical.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_IDENT = re.compile(r"^\w+$")
_MEMBER = re.compile(r"(\w+)\s*:\s*([^,}]+)")
# a scalar knob: a number, optionally followed by a unit ("23 mm", "1e-5", "0.827")
_NUMBER = re.compile(r"^(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(\s*[^\s,}]*)$")


@dataclass(frozen=True)
class Param:
    """One scalar as it appears in a .tg file."""
    name: str        # dotted path, e.g. "motors.disk.crank_throw"
    value: str       # verbatim right-hand side, units included
    source: str      # file stem it came from
    line: int        # 0-based line index, for write-back
    number: float | None = None   # None when the value is not a bare scalar
    unit: str = ""

    @property
    def tunable(self) -> bool:
        """Only bare numbers can carry a continuous gene."""
        return self.number is not None


def _mk(name: str, raw: str, source: str, line: int) -> Param:
    m = _NUMBER.match(raw.strip())
    if not m:
        return Param(name, raw.strip(), source, line)
    return Param(name, raw.strip(), source, line, float(m.group(1)), m.group(2).strip())


def parse_tg(path: Path) -> list[Param]:
    """Flatten a .tg file into dotted scalar params.

    Handles the three shapes the format uses: ``key = value``, inline dicts
    (``key { a: 1, b: 2 }`` and ``key = { ... }``), and ``section:`` headers
    nested by indentation. Structural lines -- tendon generators, ``drive M[1]
    and M[4]``, ``strut F = C {...}`` -- are skipped: their left side is not a
    bare identifier, so they can never be a scalar knob.
    """
    stack: list[tuple[int, str]] = []  # (indent, section name)
    out: list[Param] = []
    open_dict: str | None = None       # dotted prefix of a brace dict still spanning lines
    for lineno, raw in enumerate(path.read_text().splitlines()):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if open_dict is not None:
            # continuation of a multi-line `key { ... }` (the motor disk spec)
            for m, v in _MEMBER.findall(line):
                out.append(_mk(f"{open_dict}.{m}", v, path.stem, lineno))
            if "}" in line:
                open_dict = None
            continue
        indent = len(line) - len(line.lstrip())
        while stack and stack[-1][0] >= indent:
            stack.pop()
        prefix = ".".join(s for _, s in stack)
        body = line.strip()

        if body.endswith(":"):
            # name the section by its first token -- except "archetype strut C",
            # where the second one ("strut") is what the design is actually about
            tokens = body[:-1].split()
            stack.append((indent, tokens[1] if tokens[0] == "archetype" and len(tokens) > 1
                          else tokens[0]))
            continue

        key, _, rhs = body.partition("=")
        if not rhs and "{" in body:  # `key { a: 1, b: 2 }` -- dict, no `=`
            key, _, rhs = body.partition("{")
            rhs = "{" + rhs
        key, rhs = key.strip(), rhs.strip()
        if not _IDENT.match(key) or not rhs:
            continue
        dotted = f"{prefix}.{key}" if prefix else key
        if rhs.startswith("{"):
            for m, v in _MEMBER.findall(rhs):
                out.append(_mk(f"{dotted}.{m}", v, path.stem, lineno))
            if "}" not in rhs:  # dict continues on following lines
                open_dict = dotted
        else:
            out.append(_mk(dotted, rhs, path.stem, lineno))
    return out


def substitute(text: str, params: list[Param], updates: dict[str, float]) -> str:
    """Return `text` with each named param's number replaced, units preserved.

    Several params can share a line (inline dicts), so edits are applied per
    line, left to right, against the ORIGINAL line -- rewriting the whole line
    per param would clobber a sibling's fresh value.
    """
    by_name = {p.name: p for p in params}
    missing = sorted(set(updates) - set(by_name))
    if missing:
        raise KeyError(f"not scalars in this .tg: {missing}")
    if bad := sorted(n for n in updates if not by_name[n].tunable):
        raise ValueError(f"not bare numbers, cannot substitute: {bad}")

    lines = text.splitlines(keepends=True)
    per_line: dict[int, list[Param]] = {}
    for name in updates:
        per_line.setdefault(by_name[name].line, []).append(by_name[name])

    for lineno, ps in per_line.items():
        line = lines[lineno]
        for p in ps:
            new = f"{updates[p.name]:.10g}{(' ' + p.unit) if p.unit else ''}"
            # anchor on `key` so a repeated number elsewhere on the line is safe
            pat = re.compile(rf"(\b{re.escape(p.name.rsplit('.', 1)[-1])}\s*[:=]\s*)"
                             rf"{re.escape(p.value)}")
            line, n = pat.subn(lambda m: m.group(1) + new, line, count=1)
            if n != 1:
                raise ValueError(f"could not rewrite {p.name} on line {lineno}: {line!r}")
        lines[lineno] = line
    return "".join(lines)


def write_variant(src: Path, dst: Path, updates: dict[str, float]) -> Path:
    """Write `src` to `dst` with `updates` applied.

    `dst` must sit in the same directory as `src`: the `control =` include is
    resolved relative to the .tg being compiled.
    """
    if dst.parent != src.parent:
        raise ValueError(f"variant must live beside its source ({src.parent}), got {dst.parent}")
    dst.write_text(substitute(src.read_text(), parse_tg(src), updates))
    return dst
