"""Regression tests for the Executor (#49, #50, #54)."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from openagent_eval.core.executor import Executor
from openagent_eval.exceptions import MetricExecutionError


class TestRunInThread:
    """Regression tests for issue #49 (event loop handling)."""

    @pytest.mark.asyncio
    async def test_run_in_thread_from_running_loop(self) -> None:
        """Test awaiting run_in_thread from a running event loop."""
        executor = Executor()
        try:
            result = await executor.run_in_thread(lambda x: x + 1, 41)
            assert result == 42
        finally:
            executor.shutdown()

    def test_run_in_thread_without_running_loop(self) -> None:
        """Test run_in_thread from a thread with no event loop set.

        The old implementation called asyncio.get_event_loop(), which
        raises RuntimeError in a thread without a current loop (always
        on Python 3.12+), and leaked the loops it created. The fixed
        coroutine must run cleanly under asyncio.run() and leave no
        event loop set in the calling thread.
        """
        executor = Executor()
        outcome: dict[str, Any] = {}

        def target() -> None:
            try:
                outcome["result"] = asyncio.run(
                    executor.run_in_thread(lambda x: x * 2, 21),
                )
            except Exception as e:
                outcome["error"] = e
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                outcome["leftover_loop"] = False
            else:
                outcome["leftover_loop"] = True

        thread = threading.Thread(target=target)
        thread.start()
        thread.join()
        executor.shutdown()

        assert "error" not in outcome
        assert outcome["result"] == 42
        assert outcome["leftover_loop"] is False


class TestGatherCancellation:
    """Regression tests for issue #50 (sibling cancellation on timeout)."""

    @pytest.mark.asyncio
    async def test_gather_cancels_siblings_on_timeout(self) -> None:
        """Test that a timed-out task cancels the remaining coroutines.

        With max_workers=1 the sibling is still queued on the semaphore
        when the first task times out, so its own per-task timeout has
        not started: if gather() leaves it running (old behaviour) it
        runs to completion once the semaphore is released. The sibling
        must instead be cancelled before it can complete.
        """
        executor = Executor(max_workers=1, timeout=0.05)
        sibling_completed = False

        async def slow() -> None:
            await asyncio.sleep(10)

        async def sibling() -> str:
            nonlocal sibling_completed
            await asyncio.sleep(0.01)
            sibling_completed = True
            return "done"

        with pytest.raises(MetricExecutionError) as exc_info:
            await executor.gather([slow(), sibling()])

        assert exc_info.value.message == "Task timed out after 0.05s"
        assert exc_info.value.details == {"timeout": 0.05}

        # Give any orphaned sibling task a chance to run to completion.
        await asyncio.sleep(0.2)
        assert not sibling_completed
        executor.shutdown()

    @pytest.mark.asyncio
    async def test_gather_returns_results(self) -> None:
        """Test that gather still returns results in order on success."""
        executor = Executor(max_workers=2, timeout=5.0)

        async def make(value: int) -> int:
            await asyncio.sleep(0)
            return value * 2

        results = await executor.gather([make(1), make(2), make(3)])
        assert results == [2, 4, 6]
        executor.shutdown()


class TestLazyThreadPool:
    """Regression tests for issue #54 (lazy thread pool allocation)."""

    def test_no_thread_pool_on_init(self) -> None:
        """Test that constructing an Executor allocates no thread pool."""
        executor = Executor()
        assert executor._thread_pool is None

    def test_shutdown_without_use_does_not_raise(self) -> None:
        """Test that shutdown() on a never-used executor does not raise."""
        executor = Executor()
        executor.shutdown()

    @pytest.mark.asyncio
    async def test_thread_pool_created_on_first_use(self) -> None:
        """Test that the thread pool is created lazily by run_in_thread."""
        executor = Executor()
        try:
            assert executor._thread_pool is None
            await executor.run_in_thread(lambda: 1)
            assert executor._thread_pool is not None
        finally:
            executor.shutdown()
        assert executor._thread_pool is None
