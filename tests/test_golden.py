"""Quick tier of the golden behavior harness (tests/golden). Full tier: python -m tests.golden check."""
import subprocess
import sys
from pathlib import Path


def test_golden_quick():
    # Own process: the harness chdirs, sets design env vars and patches mujoco while it runs.
    r = subprocess.run([sys.executable, "-m", "tests.golden", "check", "--quick"],
                       cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-3000:]
