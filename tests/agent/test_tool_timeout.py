# -*- coding: utf-8 -*-
"""Unit tests for agent tool timeout resolution (Issue #1890).

Covers:
- the pure ``_resolve_tool_timeout`` priority resolver
- registry / policy / definition timeout fields
- ``_resolve_per_tool_timeout`` wiring
- end-to-end ``_execute_tools`` per-tool timeout + fail-open behaviour

NOTE: ``ToolRegistry`` defines ``__len__`` but not ``__bool__``, so an *empty*
registry is falsy in Python and ``@tool(registry=empty)`` falls back to the
global default registry.  These tests therefore register tools directly via
``ToolDefinition`` (the same path ``factory.get_tool_registry`` uses) instead
of relying on the decorator's registry fallback.
"""
import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

# Mock heavy optional deps before importing agent modules (mirrors
# tests/agent/test_runtime_facts.py so the suite runs without litellm).
sys.modules.setdefault("litellm", MagicMock())

from src.agent.tools.registry import (
    ToolDefinition,
    ToolPolicy,
    ToolRegistry,
    _resolve_tool_timeout,
    tool,
)
from src.agent.runner import _execute_tools, _resolve_per_tool_timeout


# ---------------------------------------------------------------------------
# 1. Pure resolver
# ---------------------------------------------------------------------------
class TestResolveToolTimeout:
    def test_all_falsy_returns_none(self):
        assert _resolve_tool_timeout(None, 0, None) is None

    def test_returns_smallest_positive(self):
        assert _resolve_tool_timeout(30, 5, 10) == 5

    def test_ignores_none_and_zero(self):
        assert _resolve_tool_timeout(None, 0, 12.5) == 12.5

    def test_per_tool_smaller_than_category(self):
        assert _resolve_tool_timeout(2, 10, 60) == 2

    def test_category_smaller_than_global(self):
        assert _resolve_tool_timeout(None, 15, 60) == 15

    def test_global_only(self):
        assert _resolve_tool_timeout(None, None, 45) == 45

    def test_negative_ignored(self):
        assert _resolve_tool_timeout(-1, 8, -5) == 8

    def test_non_numeric_ignored(self):
        assert _resolve_tool_timeout("oops", 9, None) == 9


# ---------------------------------------------------------------------------
# 2. Registry / policy / definition fields
# ---------------------------------------------------------------------------
class TestRegistryTimeoutFields:
    def test_category_default_timeout_lookup(self):
        reg = ToolRegistry(category_timeout_map={"data": 30.0, "action": 5.0})
        assert reg.category_default_timeout("data") == 30.0
        assert reg.category_default_timeout("action") == 5.0
        assert reg.category_default_timeout("analysis") is None

    def test_policy_declared_carries_timeout(self):
        assert ToolPolicy.declared(read_only=True, timeout_seconds=7.0).timeout_seconds == 7.0
        assert ToolPolicy.declared(read_only=True).timeout_seconds is None

    def test_definition_carries_timeout(self):
        d = ToolDefinition(
            name="x", description="x", parameters=[], handler=lambda: None,
            timeout_seconds=3.0,
        )
        assert d.timeout_seconds == 3.0

    def test_decorator_exposes_timeout(self):
        reg = ToolRegistry()

        @tool(name="sample", category="data", description="sample", registry=reg, timeout_seconds=4.0)
        def sample():
            return 1

        assert getattr(sample, "_tool_definition", None) is not None
        assert sample._tool_definition.timeout_seconds == 4.0


# ---------------------------------------------------------------------------
# 3. Runner wiring
# ---------------------------------------------------------------------------
def _make_tool_call(name, args=None, tc_id="call_1"):
    return SimpleNamespace(name=name, arguments=args or {}, id=tc_id)


def _register(reg, name, fn, *, category="data", timeout_seconds=None):
    reg.register(ToolDefinition(
        name=name, description=name, parameters=[], handler=fn,
        category=category, timeout_seconds=timeout_seconds,
    ))


class TestResolvePerToolTimeout:
    def test_per_tool_overrides_category(self):
        reg = ToolRegistry(category_timeout_map={"data": 30.0})
        _register(reg, "t", lambda: None, category="data", timeout_seconds=2.0)
        assert _resolve_per_tool_timeout(_make_tool_call("t"), reg, 60.0) == 2.0

    def test_category_used_when_no_per_tool(self):
        reg = ToolRegistry(category_timeout_map={"data": 30.0})
        _register(reg, "t", lambda: None, category="data")
        assert _resolve_per_tool_timeout(_make_tool_call("t"), reg, 60.0) == 30.0

    def test_global_budget_caps_everything(self):
        reg = ToolRegistry(category_timeout_map={"data": 30.0})
        _register(reg, "t", lambda: None, category="data", timeout_seconds=2.0)
        # remaining budget of 1.0 caps the resolved timeout
        assert _resolve_per_tool_timeout(_make_tool_call("t"), reg, 1.0) == 1.0

    def test_no_limits_returns_none(self):
        reg = ToolRegistry()
        _register(reg, "t", lambda: None, category="data")
        assert _resolve_per_tool_timeout(_make_tool_call("t"), reg, None) is None


# ---------------------------------------------------------------------------
# 4. End-to-end _execute_tools
# ---------------------------------------------------------------------------
class TestExecuteToolsTimeout:
    def test_per_tool_timeout_fires_and_fail_open(self):
        reg = ToolRegistry(category_timeout_map={"data": 0.2})

        def slow():
            time.sleep(1.0)
            return {"ok": True}

        _register(reg, "slow", slow, category="data")
        log = []
        results = _execute_tools(
            [_make_tool_call("slow")], reg, step=1,
            progress_callback=None, tool_calls_log=log,
            tool_wait_timeout_seconds=None,
        )
        assert len(results) == 1
        parsed = json.loads(results[0]["result_str"])
        assert parsed.get("timeout") is True
        assert any(e.get("timeout") is True for e in log)

    def test_no_timeout_when_fast_enough(self):
        reg = ToolRegistry(category_timeout_map={"data": 2.0})

        def fast():
            return {"ok": True}

        _register(reg, "fast", fast, category="data")
        log = []
        results = _execute_tools(
            [_make_tool_call("fast")], reg, step=1,
            progress_callback=None, tool_calls_log=log,
            tool_wait_timeout_seconds=None,
        )
        assert json.loads(results[0]["result_str"]).get("ok") is True
        assert not any(e.get("timeout") for e in log)

    def test_backward_compat_no_limits_runs_inline(self):
        # No category map, no global timeout -> tool executes inline, no
        # spurious timeout.
        reg = ToolRegistry()

        def plain():
            return {"ok": True}

        _register(reg, "plain", plain, category="data")
        log = []
        results = _execute_tools(
            [_make_tool_call("plain")], reg, step=1,
            progress_callback=None, tool_calls_log=log,
            tool_wait_timeout_seconds=None,
        )
        assert json.loads(results[0]["result_str"]).get("ok") is True

    def test_parallel_per_tool_timeout(self):
        reg = ToolRegistry(category_timeout_map={"data": 0.2})

        def slowA():
            time.sleep(1.0)
            return {"a": True}

        def slowB():
            time.sleep(1.0)
            return {"b": True}

        _register(reg, "slowA", slowA, category="data")
        _register(reg, "slowB", slowB, category="data")
        log = []
        results = _execute_tools(
            [_make_tool_call("slowA", tc_id="c1"), _make_tool_call("slowB", tc_id="c2")],
            reg, step=1, progress_callback=None, tool_calls_log=log,
            tool_wait_timeout_seconds=None,
        )
        assert len(results) == 2
        assert all(json.loads(r["result_str"]).get("timeout") is True for r in results)


# ---------------------------------------------------------------------------
# 5. Config env contract (Issue #1890 test point 1: AGENT_DATA_TOOL_TIMEOUT_S=20)
# ---------------------------------------------------------------------------
class TestConfigEnvContract:
    def test_field_names_match_issue_contract(self):
        # Issue #1890 specifies env names like AGENT_DATA_TOOL_TIMEOUT_S.
        from src.config import Config

        names = {f.name for f in Config.__dataclass_fields__.values()}
        assert "agent_data_tool_timeout_s" in names
        assert "agent_search_tool_timeout_s" in names
        assert "agent_analysis_tool_timeout_s" in names
        assert "agent_action_tool_timeout_s" in names

    def test_field_default_is_zero_backward_compatible(self):
        # No env set -> 0.0 -> category default disabled -> behaves like today.
        from src.config import Config

        default = Config.__dataclass_fields__["agent_data_tool_timeout_s"].default
        assert default == 0.0

    def test_env_value_is_parsed(self, monkeypatch):
        import os

        import src.config as cfg

        monkeypatch.setenv("AGENT_DATA_TOOL_TIMEOUT_S", "20")
        val = cfg.parse_env_float(
            os.getenv("AGENT_DATA_TOOL_TIMEOUT_S"),
            0.0,
            field_name="AGENT_DATA_TOOL_TIMEOUT_S",
            minimum=0.0,
        )
        assert val == 20.0
