"""Tests for the oaeval run command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from openagent_eval.cli.main import app
from openagent_eval.config.models import Config, DatasetConfig, LLMConfig, MetricsConfig

runner = CliRunner()


def _make_config() -> Config:
    """Return a minimal valid configuration for run-command tests."""
    return Config(
        dataset=DatasetConfig(path="data/questions.json"),
        llm=LLMConfig(provider="openai", model="gpt-4o"),
        metrics=MetricsConfig(
            retrieval=["context_precision"],
            generation=["faithfulness"],
        ),
    )


@patch("openagent_eval.cli.commands.run.load_config_from_path")
@patch("openagent_eval.cli.commands.run.load_dataset_for_run")
@patch("openagent_eval.cli.commands.run.execute_evaluation")
@patch("openagent_eval.cli.commands.run.ReportManager")
def test_run_loads_config_exactly_once(
    mock_manager_class: MagicMock,
    mock_execute: MagicMock,
    mock_load_dataset: MagicMock,
    mock_load_config: MagicMock,
    tmp_path: Path,
) -> None:
    """Regression test for #41: config must be parsed from disk exactly once.

    The pre-fix implementation loaded the configuration a second time inside the
    Progress context, so this test patches the function ``run.py`` imports and
    asserts it is called only once for a normal (non-dry-run) invocation.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset: data/questions.json\n", encoding="utf-8")

    config = _make_config()
    mock_load_config.return_value = config
    mock_load_dataset.return_value = [
        {
            "question": "What is Python?",
            "ground_truth": "A programming language.",
            "context": "Python is a language.",
        }
    ]
    mock_execute.return_value = MagicMock(summary={"score": 1.0})

    mock_manager = MagicMock()
    mock_manager.save_report.return_value = tmp_path / "report.json"
    mock_manager_class.return_value = mock_manager

    result = runner.invoke(app, ["--quiet", "run", str(config_path)])

    assert result.exit_code == 0, result.output
    mock_load_config.assert_called_once_with(Path(config_path))
