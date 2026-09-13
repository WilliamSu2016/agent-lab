"""Tests for src/reliability/timeout.py: mandatory timeout enforcement on
every external tool call.
"""

from __future__ import annotations

import time

import pytest

from src.reliability.timeout import TimeoutPolicy, ToolTimeoutError, run_with_timeout


class TestRunWithTimeout:
    def test_fast_function_returns_normally(self):
        assert run_with_timeout(lambda: "done", 1.0, tool_name="fast") == "done"

    def test_slow_function_raises_tool_timeout_error(self):
        def slow():
            time.sleep(2.0)
            return "too late"

        with pytest.raises(ToolTimeoutError) as exc_info:
            run_with_timeout(slow, 0.05, tool_name="slow_tool")
        assert exc_info.value.tool_name == "slow_tool"
        assert exc_info.value.timeout_seconds == 0.05

    def test_tool_timeout_error_is_a_timeout_error(self):
        """So existing ``except TimeoutError`` call sites keep working
        unmodified after consolidating onto this implementation."""
        assert issubclass(ToolTimeoutError, TimeoutError)

    def test_none_disables_timeout(self):
        assert run_with_timeout(lambda: "no deadline", None, tool_name="t") == "no deadline"

    def test_none_disables_timeout_even_for_a_slow_call(self):
        def slow():
            time.sleep(0.1)
            return "eventually done"

        assert run_with_timeout(slow, None, tool_name="t") == "eventually done"

    @pytest.mark.parametrize("bad_timeout", [0, -1, -0.5])
    def test_non_positive_timeout_raises_value_error(self, bad_timeout):
        with pytest.raises(ValueError):
            run_with_timeout(lambda: "unused", bad_timeout, tool_name="t")

    def test_underlying_exception_propagates_when_within_timeout(self):
        def raises():
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            run_with_timeout(raises, 1.0, tool_name="t")

    def test_default_tool_name_is_used_when_not_supplied(self):
        def slow():
            time.sleep(1.0)

        with pytest.raises(ToolTimeoutError) as exc_info:
            run_with_timeout(slow, 0.05)
        assert exc_info.value.tool_name == "tool"


class TestTimeoutPolicy:
    def test_rejects_non_positive_timeout_seconds(self):
        with pytest.raises(ValueError):
            TimeoutPolicy(tool_name="t", timeout_seconds=0)

    def test_run_delegates_to_run_with_timeout(self):
        policy = TimeoutPolicy(tool_name="my_tool", timeout_seconds=0.05)

        def slow():
            time.sleep(1.0)

        with pytest.raises(ToolTimeoutError) as exc_info:
            policy.run(slow)
        assert exc_info.value.tool_name == "my_tool"

    def test_run_returns_result_for_fast_call(self):
        policy = TimeoutPolicy(tool_name="my_tool", timeout_seconds=1.0)
        assert policy.run(lambda: 42) == 42
