"""Regression test for issue #227.

The CLI must be importable and runnable on a clean install that does not
include pytest.  We verify this in-process by blocking the ``pytest`` name
before importing the modules.
"""

import subprocess
import sys


def test_cicd_and_cli_import_without_pytest() -> None:
    """Plugin and CLI entry point import cleanly when pytest is unavailable."""
    code = (
        "import sys\n"
        "sys.modules['pytest'] = None\n"
        "from openagent_eval.cicd import OAEvalPlugin\n"
        "from openagent_eval.cli.main import app\n"
        "print('import-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "import-ok" in result.stdout


def test_oaeval_help_runs_without_pytest() -> None:
    """``oaeval --help`` works when pytest is unavailable."""
    code = (
        "import sys\n"
        "sys.modules['pytest'] = None\n"
        "from typer.testing import CliRunner\n"
        "from openagent_eval.cli.main import app\n"
        "runner = CliRunner()\n"
        "result = runner.invoke(app, ['--help'])\n"
        "print(result.output)\n"
        "assert result.exit_code == 0, result.exception\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout
