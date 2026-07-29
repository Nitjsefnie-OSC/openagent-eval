"""Regression test for issue #51.

Verifies that ``openagent_eval/core/pipeline.py`` reads the LLM model
identifier through the public ``LLMProvider.model_name`` property rather than
reaching into a private ``_model`` attribute.
"""

from __future__ import annotations

from typing import Any

import pytest

from openagent_eval.config.models import (
    Config,
    DatasetConfig,
    LLMConfig,
    MetricsConfig,
    ReportConfig,
    RetrieverConfig,
)
from openagent_eval.core.pipeline import Pipeline
from openagent_eval.metrics.base import BaseMetric, MetricResult
from openagent_eval.providers.base.llm import LLMProvider
from openagent_eval.providers.models import LLMResponse, TokenUsage


class _ModelNameCapturingMetric(BaseMetric):
    """Metric that records the ``model`` argument it receives."""

    name = "model_name_capture"
    description = "Captures the model name passed by the pipeline"

    def __init__(self) -> None:
        self.captured_model: str | None = None

    def evaluate(self, **kwargs: Any) -> MetricResult:
        self.captured_model = kwargs.get("model")
        return MetricResult(score=1.0, reason="captured")


class _FakeLLMProvider(LLMProvider):
    """Fake LLM provider that exposes ``model_name`` but not ``_model``.

    This proves the pipeline uses the public property: the provider's
    ``model_name`` differs from ``config.llm.model``, so a fallback to the
    config value is visibly distinguishable from a real read.
    """

    name = "fake"
    description = "Fake provider for regression testing"

    def __init__(self, model_name: str) -> None:
        # Intentionally no ``_model`` attribute.
        self._reported_model = model_name

    @property
    def model_name(self) -> str | None:
        return self._reported_model

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return "fake answer"

    async def get_token_count(self, text: str) -> int:
        return 0

    async def generate_with_usage(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            content="fake answer",
            model=self._reported_model,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            provider=self.name,
            latency_ms=1.0,
        )


@pytest.mark.asyncio
async def test_pipeline_uses_public_model_name_property() -> None:
    """A provider exposing only ``model_name`` has it forwarded to metrics."""
    config_model = "config-model"
    provider_model = "provider-specific-model"

    config = Config(
        dataset=DatasetConfig(path="data/questions.json"),
        llm=LLMConfig(provider="fake", model=config_model),
        retriever=RetrieverConfig(provider="mock"),
        metrics=MetricsConfig(),
        report=ReportConfig(),
        parallel=False,
    )

    llm = _FakeLLMProvider(model_name=provider_model)
    capture_metric = _ModelNameCapturingMetric()
    pipeline = Pipeline(config, llm=llm, metrics=[("model_name_capture", capture_metric)])

    await pipeline.execute(
        [{"question": "What is RAG?", "ground_truth": "RAG is retrieval augmented generation."}]
    )

    assert capture_metric.captured_model == provider_model
