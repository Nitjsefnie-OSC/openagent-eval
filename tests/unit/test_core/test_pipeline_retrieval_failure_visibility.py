"""Regression test for issue #256.

Verifies that a retriever failure inside ``Pipeline._retrieve`` is no longer
silently swallowed: the dataset-context fallback still applies (behaviour
unchanged), but the failure is now logged naming the exception and recorded
in the run's ``errors`` in the same shape as the other failure modes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
from openagent_eval.providers.base.retriever import Retriever

if TYPE_CHECKING:
    from openagent_eval.providers.models import Document


class _FailingRetriever(Retriever):
    """Retriever whose ``retrieve`` always raises, simulating a broken backend."""

    name = "failing"
    description = "Always raises from retrieve()"

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        ground_truth_contexts: list[str] | None = None,
    ) -> list[Document]:
        raise RuntimeError("vector store unreachable")


@pytest.mark.asyncio
async def test_retrieval_failure_is_visible_and_still_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising retriever degrades to dataset context but is surfaced."""
    from openagent_eval.providers.llm.mock import MockLLMProvider

    config = Config(
        dataset=DatasetConfig(path="data/questions.json"),
        llm=LLMConfig(provider="mock", model="mock-model"),
        retriever=RetrieverConfig(provider="failing"),
        metrics=MetricsConfig(),
        report=ReportConfig(),
        parallel=False,
    )

    pipeline = Pipeline(
        config, retriever=_FailingRetriever(), llm=MockLLMProvider()
    )

    with caplog.at_level(logging.WARNING, logger="openagent_eval.core.pipeline"):
        result = await pipeline.execute(
            [
                {
                    "question": "What is RAG?",
                    "ground_truth": "RAG is retrieval augmented generation.",
                    "context": "dataset-provided context",
                }
            ]
        )

    # (a) Fallback behaviour is unchanged: the dataset context is still used.
    assert result.results[0].contexts == ["dataset-provided context"]

    # (b) The failure is recorded in the run's errors, in the same shape the
    # other failure modes use (item / error / error_type).
    assert len(result.errors) == 1
    entry = result.errors[0]
    assert entry["error_type"] == "RuntimeError"
    assert "vector store unreachable" in entry["error"]
    assert entry["item"]["question"] == "What is RAG?"

    # (c) The failure is logged, naming the exception.
    assert "RuntimeError" in caplog.text
    assert "vector store unreachable" in caplog.text
