"""Tests for the public import surface of ``openagent_eval.config``.

Regression guard for #47: config model names must be importable directly from
``openagent_eval.config`` and must be listed in ``__all__``.
"""

from __future__ import annotations

import openagent_eval.config as config_pkg
from openagent_eval.config.models import CorpusConfig as ModelsCorpusConfig
from openagent_eval.config.models import OutputFormat as ModelsOutputFormat
from openagent_eval.config.models import ReportConfig as ModelsReportConfig


def test_all_lists_the_full_public_surface() -> None:
    """``__all__`` must enumerate exactly the intended public names."""
    assert set(config_pkg.__all__) == {
        "Config",
        "CorpusConfig",
        "DatasetConfig",
        "EmbedderConfig",
        "LLMConfig",
        "MetricsConfig",
        "OutputFormat",
        "ReportConfig",
        "RetrieverConfig",
        "load_config",
    }


def test_exported_names_match_defining_objects() -> None:
    """Re-exported names must be the same objects as the defining module."""
    assert config_pkg.CorpusConfig is ModelsCorpusConfig
    assert config_pkg.OutputFormat is ModelsOutputFormat
    assert config_pkg.ReportConfig is ModelsReportConfig


def test_exported_names_are_in_all() -> None:
    """Each newly exported name must be present in ``__all__``."""
    for name in ("CorpusConfig", "OutputFormat", "ReportConfig"):
        assert name in config_pkg.__all__, f"{name} missing from __all__"
