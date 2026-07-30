"""Unit tests for CLI test command."""

import json
import re

from typer.testing import CliRunner

from openagent_eval.cli.main import app


runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestTestCommand:
    """Tests for oaeval test command."""

    def test_test_command_help(self):
        """Test test command help output."""
        result = runner.invoke(app, ["test", "--help"])
        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "Run evaluation as a CI/CD test" in output

    def test_test_command_no_config(self):
        """Test test command without config shows error."""
        result = runner.invoke(app, ["test"])
        assert result.exit_code != 0

    def test_test_command_nonexistent_config(self):
        """Test test command with nonexistent config."""
        result = runner.invoke(app, ["test", "/nonexistent/config.yaml"])
        assert result.exit_code == 2

    def test_test_command_invalid_threshold_format(self):
        """Test test command with invalid threshold format."""
        result = runner.invoke(
            app,
            ["test", "config.yaml", "-t", "invalid_format"],
        )
        assert result.exit_code == 2

    def test_test_command_invalid_threshold_value(self):
        """Test test command with invalid threshold value."""
        result = runner.invoke(
            app,
            ["test", "config.yaml", "-t", "faithfulness:gte:not_a_number"],
        )
        assert result.exit_code == 2

    def test_test_command_invalid_operator(self):
        """Test test command with invalid operator."""
        result = runner.invoke(
            app,
            ["test", "config.yaml", "-t", "faithfulness:invalid:0.8"],
        )
        assert result.exit_code == 2

    def test_test_command_json_output(self):
        """Test test command with --json flag."""
        result = runner.invoke(app, ["test", "--help"])
        output = _strip_ansi(result.output)
        assert "json" in output.lower()

    def test_test_command_timeout_option(self):
        """Test test command with --timeout option."""
        result = runner.invoke(app, ["test", "--help"])
        output = _strip_ansi(result.output)
        assert "timeout" in output.lower()

    def test_test_command_threshold_option(self):
        """Test test command with --threshold option."""
        result = runner.invoke(app, ["test", "--help"])
        output = _strip_ansi(result.output)
        assert "threshold" in output.lower()
        assert "-t" in output

    def test_test_command_passing_threshold_is_not_reported_missing(self, tmp_path):
        """Regression test for issue #228.

        A threshold that should pass must not fail with "metric not found":
        the gate summary returned by ``OAEvalPlugin.run_evaluation`` must
        carry ``metrics_summary`` so the CLI threshold-evaluation path can
        look the metric up. Uses fully offline mock providers.
        """
        dataset_path = tmp_path / "data.json"
        dataset_path.write_text(
            json.dumps(
                [
                    {
                        "question": "What is RAG?",
                        "ground_truth": "RAG combines retrieval with generation.",
                        "context": "RAG combines retrieval with generation.",
                        "ground_truth_contexts": [
                            "RAG combines retrieval with generation."
                        ],
                    }
                ]
            )
        )
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"""
dataset:
  path: {dataset_path}
llm:
  provider: mock
  model: mock-model
retriever:
  provider: mock
  settings:
    collection_name: c
metrics:
  retrieval: []
  generation: ["exact_match"]
  performance: []
  cost: []
report:
  output: json
  output_dir: {tmp_path / "reports_out"}
parallel: false
"""
        )

        result = runner.invoke(
            app,
            ["test", str(config_path), "-t", "exact_match:gte:0.5"],
        )
        output = _strip_ansi(result.output)

        assert "not found in results" not in output
        assert result.exit_code == 0, output
