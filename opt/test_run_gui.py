"""Self-check for opt/run_gui.py: .tg flattening, gene wiring, argv building.

No Tk display needed -- the parsing and command layers are plain functions.
Run: uv run python opt/test_run_gui.py
"""
from pathlib import Path

from run_gui import (ROOT, RunConfig, block_of, build_commands, gene_blocks, grouped,
                     parse_tg)

TG = ROOT / "design" / "shoulder.tg"  # the promoted flagship (hovering-it009p); values below are its
CTRL = ROOT / "design" / "control.tg"  # must be named control.tg: the Fourier group keys on the file stem


def test_parse_tg():
    names = {p.name: p.value for p in parse_tg(TG)}
    # plain `key = value` inside a section
    assert names["world.timestep"] == "0.13 ms", names.get("world.timestep")
    assert names["placement.mount_height"] == "258 mm"
    assert names["motors.tendon_stiffness"] == "1000 N/m"
    # inline dict without `=`
    assert names["motors.disk.crank_throw"] == "22 mm"
    # ...that spans several lines: the continuation members are knobs too
    assert names["motors.disk.outboard"] == "30.625 mm"
    assert names["motors.disk.disk_dy"] == "-20 mm"
    assert "motors.disk_dy" not in names, "continuation lost its dict prefix"
    assert names["tendons.defaults.pretension"] == "0.827"
    # nested section keeps its own prefix
    assert names["wing.pitch.free_angle"] == "10 deg"
    assert names["wing.rod_length"] == "18.0 mm"
    # `archetype strut C:` is named for what it is, not its leading keyword
    assert "strut.radius" in names and "archetype.radius" not in names
    # structural lines are NOT knobs: `strut F = C {...}`, the tendon generators,
    # and `drive M[1] and M[4]` all have a non-identifier left side
    assert not any("--" in k for k in names), [k for k in names if "--" in k]
    for never in ("strut.role", "strut.plane", "motors.drive", "generators"):
        assert never not in names, never
    # comments stripped
    assert all("#" not in v for v in names.values())

    ctrl = {p.name: p.value for p in parse_tg(CTRL)}
    assert ctrl["a0"] == "65" and ctrl["b_5"] == "0" and ctrl["harmonics"] == "5"


def test_substitute_rewrites_only_the_named_scalars():
    from tg_scalars import substitute
    from evolve_codesign import geom_value

    text = TG.read_text()
    ps = parse_tg(TG)
    # two edits on ONE inline-dict line, plus a same-named key in another section
    out = substitute(text, ps, {"strut.radius": 52.0, "motors.disk.crank_throw": 25.0,
                                "motors.disk.pitch": 22.0})
    changed = [(a, b) for a, b in zip(text.splitlines(), out.splitlines()) if a != b]
    assert len(changed) == 2, changed          # strut radius line + the disk dict line
    assert "radius: 18.0 mm" in out            # disk.radius untouched by strut.radius
    assert "crank_throw: 25 mm" in out and "pitch: 22 deg" in out
    assert "crank_throw_t4 = 25 mm" in out     # prefix-sharing key not clobbered
    # re-parsing the result must round-trip the new values
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "v.tg"
        p.write_text(out)
        got = {q.name: q.number for q in parse_tg(p)}
    assert got["strut.radius"] == 52.0 and got["motors.disk.crank_throw"] == 25.0
    assert got["motors.disk.radius"] == 18.0

    # gene decoding: log-factor off the baseline, additive for a zero baseline
    assert abs(geom_value(258.0, 0.0, "mm") - 258.0) < 1e-9
    assert geom_value(258.0, 0.1, "mm") > 258.0 and geom_value(258.0, -0.1, "mm") < 258.0
    assert geom_value(-20.0, 0.1, "mm") < 0, "log-factor must not flip a sign"
    assert geom_value(0.0, 0.5, "rad") == 0.05, "zero baseline moves additively"


def test_grouping_is_trimmed_and_ordered():
    params = parse_tg(TG) + parse_tg(CTRL)
    groups = grouped(params)
    assert [t for t, _ in groups] == ["Control -- Fourier coefficients", "Struts",
                                      "Strut arrangement", "Placement", "Motors", "Tendons"]
    shown = {p.name for _, rows in groups for p in rows}
    # the sections the design is actually shaped by
    for want in ("strut.radius", "strut.arc_rod.mass", "arrange.gap", "arrange.inplane_offset",
                 "placement.mount_height", "motors.disk.crank_throw", "motors.crank_phase_lag"):
        assert want in shown, want
    # world/wing/aero are noise for this purpose and stay hidden
    assert not any(n.startswith(("world.", "wing.")) for n in shown), shown
    # non-scalars can't carry a gene, so they are never offered as a checkbox
    for never in ("strut.shape", "strut.stations", "motors.disk.ctrl", "motors.control"):
        assert never not in shown, never


def test_gene_blocks():
    params = parse_tg(TG) + parse_tg(CTRL)
    blocks = gene_blocks(params)
    assert set(blocks) == {"control", "tendon_stiffness", "tendon_pretension", "motor"}
    # control block = a0 + 2K coefficients, matching the EA genome's 2K+1 head
    assert len(blocks["control"]) == 11, blocks["control"]
    assert blocks["tendon_stiffness"] == ["tendons.defaults.stiffness"]
    assert sorted(blocks["motor"]) == ["motors.tendon_rest_length", "motors.tendon_stiffness"]
    assert block_of("a_3", blocks) == "control"
    assert block_of("motors.disk.crank_throw", blocks) is None  # geometry: unwired


def cfg(**kw) -> RunConfig:
    base = dict(tg=TG, scene=ROOT / "scene.xml", out=Path("/tmp/x"),
                genes=["control", "tendon_stiffness"])
    return RunConfig(**{**base, **kw})


def test_geom_genes_reach_the_ga():
    argv = dict(build_commands(cfg(
        geom_genes=["placement.mount_height", "motors.disk.crank_throw"])))["GA"]
    assert argv[argv.index("--geom-genes") + 1] == \
        "placement.mount_height,motors.disk.crank_throw"
    # no geometry selected -> the flag is absent, keeping the fast single-model path
    assert "--geom-genes" not in dict(build_commands(cfg()))["GA"]
    # geometry alone is enough to justify a GA run, with no model-array blocks
    assert dict(build_commands(cfg(genes=[], geom_genes=["arrange.gap"], run_adam=False)))


def test_build_commands():
    stages = dict((l, a) for l, a in build_commands(cfg()))
    assert set(stages) == {"GA", "diff-sim seed_1"}
    ga, ds = stages["GA"], stages["diff-sim seed_1"]
    assert ga[ga.index("--genes") + 1] == "control,tendon_stiffness"
    assert ga[ga.index("--timesteps") + 1:] == ["0.06", "0.03"]
    # chained run warm-starts Adam from the GA winner
    assert "--init-codesign" in ds
    assert "--tune-tendons" in ds and "--freeze-control" not in ds
    # both halves selected -> no restriction flag
    assert "--tendon-params" not in ds


def test_top_n_fans_out():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        stages = dict(build_commands(cfg(out=out, top_n=3)))
        assert list(stages) == ["GA", "diff-sim seed_1", "diff-sim seed_2", "diff-sim seed_3"]
        ga = stages["GA"]
        assert ga[ga.index("--n-seeds") + 1] == "3", "GA must materialize what diff-sim consumes"
        for rank in (1, 2, 3):
            o = stages[f"diff-sim seed_{rank}"]
            o = o[o.index("--out") + 1]
            assert o.endswith(f"adam/seed_{rank}"), o  # runs must not clobber each other
        # before the GA runs there is no seed_1/, so rank 1 falls back to the
        # every-generation checkpoint (what an interrupted run leaves behind)
        s1 = stages["diff-sim seed_1"]
        assert s1[s1.index("--init-codesign") + 1].endswith("ga/best_genome.json")

        # once the GA has materialized its seeds, every rank resolves to its own
        for rank in (1, 2, 3):
            d = out / "ga" / f"seed_{rank}"
            d.mkdir(parents=True)
            (d / "codesign_seed.json").write_text("{}")
        stages = dict(build_commands(cfg(out=out, top_n=3)))
        for rank in (1, 2, 3):
            argv = stages[f"diff-sim seed_{rank}"]
            seed = argv[argv.index("--init-codesign") + 1]
            assert seed.endswith(f"ga/seed_{rank}/codesign_seed.json"), seed

    # GA off: top-N is meaningless, exactly one cold-start run
    solo = build_commands(cfg(run_ga=False, top_n=4))
    assert [l for l, _ in solo] == ["diff-sim cold"]
    assert "--init-codesign" not in dict(solo)["diff-sim cold"]


def test_tendon_half_and_freeze():
    argv = dict(build_commands(cfg(run_ga=False, tune_pretension=False)))["diff-sim cold"]
    assert argv[argv.index("--tendon-params") + 1] == "stiffness"
    assert "--init-codesign" not in argv  # GA off -> cold start

    argv = dict(build_commands(cfg(run_ga=False, tune_stiffness=False)))["diff-sim cold"]
    assert argv[argv.index("--tendon-params") + 1] == "pretension"

    argv = dict(build_commands(cfg(run_ga=False, tune_fourier=False)))["diff-sim cold"]
    assert "--freeze-control" in argv

    # control-only: no tendon flags at all
    argv = dict(build_commands(cfg(run_ga=False, tune_stiffness=False,
                                   tune_pretension=False)))["diff-sim cold"]
    assert "--tune-tendons" not in argv and "--tendon-params" not in argv


def test_rejects_empty_configs():
    for bad, msg in ((dict(run_ga=False, run_adam=False), "no stage"),
                     (dict(genes=[], run_adam=False), "no tunable genes"),
                     (dict(top_n=0), "top-N must be >= 1"),
                     (dict(run_ga=False, tune_stiffness=False, tune_pretension=False,
                           tune_fourier=False), "nothing to optimize")):
        try:
            build_commands(cfg(**bad))
        except ValueError as e:
            assert msg in str(e), (bad, e)
        else:
            raise AssertionError(f"expected ValueError for {bad}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all good")
