"""Regression test for issue #53.

Verifies that ``openagent_eval/core/pipeline.py`` passes
``ground_truth_contexts`` to every retriever through the public
``Retriever.retrieve`` signature instead of special-casing the mock retriever
by name.
"""

from __future__ import annotations

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
from openagent_eval.providers.models import Document


class _NonMockRetriever(Retriever):
    """Non-mock retriever that records the ground_truth_contexts it receives."""

    name = "not-mock"
    description = "Records ground_truth_contexts for regression testing"

    def __init__(self) -> None:
        self.captured_ground_truth_contexts: list[str] | None = None

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        ground_truth_contexts: list[str] | None = None,
    ) -> list[Document]:
        self.captured_ground_truth_contexts = ground_truth_contexts
        return [
            Document(
                content="retrieved context",
                score=1.0,
                id="doc-1",
            )
        ]


@pytest.mark.asyncio
async def test_pipeline_passes_ground_truth_contexts_to_non_mock_retriever() -> None:
    """A retriever whose name is not ``mock`` still receives ground_truth_contexts."""
    expected_contexts = ["ground truth context one", "ground truth context two"]

    # A minimal fake LLM so the pipeline can run generation. The provider is not
    # the subject of this test.
    from openagent_eval.providers.llm.mock import MockLLMProvider

    config = Config(
        dataset=DatasetConfig(path="data/questions.json"),
        llm=LLMConfig(provider="mock", model="mock-model"),
        retriever=RetrieverConfig(provider="not-mock"),
        metrics=MetricsConfig(),
        report=ReportConfig(),
        parallel=False,
    )

    retriever = _NonMockRetriever()
    pipeline = Pipeline(config, retriever=retriever, llm=MockLLMProvider())

    await pipeline.execute(
        [
            {
                "question": "What is RAG?",
                "ground_truth": "RAG is retrieval augmented generation.",
                "ground_truth_contexts": expected_contexts,
            }
        ]
    )

    assert retriever.captured_ground_truth_contexts == expected_contexts
