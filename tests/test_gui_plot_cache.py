"""Self-checks for the GUI's plot cache and canvas resize guard."""
import os

import plot_fixed_frequency_gui as gui
from gui import sims


class FakeWidget:
    def __init__(self) -> None:
        self.bindings = {}

    def pack(self, **_options) -> None:
        pass

    def bind(self, event, callback) -> None:
        self.bindings[event] = callback

    def emit(self, event, width=800, height=600) -> None:
        callback = self.bindings.get(event)
        if callback:
            callback(type("Event", (), {"width": width, "height": height})())


class FakeCanvas:
    def __init__(self, _figure, master=None) -> None:
        self.widget = FakeWidget()
        self.resizes = 0
        self.draws = 0

    def get_tk_widget(self):
        return self.widget

    def resize(self, _event) -> None:
        self.resizes += 1

    def draw(self) -> None:
        self.draws += 1


def test_plot_cache_and_resize_guard(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sims, "PLOT_CACHE_DIR", tmp_path / "cache")
    tg = tmp_path / "a.tg"
    control = tmp_path / "c.tg"
    tg.write_text("x")
    control.write_text("y")

    calls = []

    def fake_sim(tg_path, scene_path, periods, *, control_path=None, settle_periods=3.0):
        calls.append(periods)
        return {"periods": periods}

    scene = tmp_path / "scene.xml"

    r1 = sims.cached_sim(fake_sim, tg, scene, 5.0, control_path=control, settle_periods=3.0)
    r2 = sims.cached_sim(fake_sim, tg, scene, 5.0, control_path=control, settle_periods=3.0)
    assert r1 == r2 == {"periods": 5.0}
    assert calls == [5.0], f"expected one sim run, got {calls}"

    sims.cached_sim(fake_sim, tg, scene, 2.0, control_path=control, settle_periods=3.0)
    assert calls == [5.0, 2.0], "different periods must miss"

    os.utime(tg, ns=(tg.stat().st_atime_ns, tg.stat().st_mtime_ns + 1))
    sims.cached_sim(fake_sim, tg, scene, 5.0, control_path=control, settle_periods=3.0)
    assert calls == [5.0, 2.0, 5.0], "touched design must miss"

    monkeypatch.setattr(gui, "FigureCanvasTkAgg", FakeCanvas)
    canvas = gui.embed_figure(object(), object(), expand=True)
    canvas.widget.emit("<Configure>")
    canvas.widget.emit("<Configure>")
    canvas.widget.emit("<Configure>", width=900)
    assert canvas.draws == 1
    assert canvas.resizes == 2, "tab remaps must skip duplicate sizes but real resizes must redraw"
