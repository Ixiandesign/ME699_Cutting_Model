"""Direct Python launches must enter Streamlit instead of running in bare mode."""

from pathlib import Path
import subprocess
import sys


def test_direct_launch_enters_streamlit_cli():
    app = Path(__file__).resolve().parents[1] / "app.py"
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(app), "--help"],
        cwd=app.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "missing ScriptRunContext" not in output
    assert "--server.port" in output
