# -*- coding: utf-8 -*-
"""Unit tests for agent tool timeout resolution (Issue #1890).

Covers:
- the pure ``_resolve_tool_timeout`` minimum resolver
- registry / policy / definition timeout fields
- ``_resolve_per_tool_timeout`` wiring
- end-to-end ``_execute_tools`` per-tool timeout + fail-open behaviour

NOTE: ``ToolRegistry`` defines ``__len__`` but not ``__bool__``, so an *empty*
registry is falsy in Python and ``@tool(registry=empty)`` falls back to the
global default registry.  These tests therefore register tools directly via
``ToolDefinition`` (the same path ``factory.get_tool_registry`` uses) instead
of relying on the decorator's registry fallback.
"""
import gc
import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agent.factory import _build_category_timeout_map

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


# ---------------------------------------------------------------------------
# 6. Category coverage (Issue #1890 review: market tools must get a ceiling)
# ---------------------------------------------------------------------------
class TestCategoryTimeoutMap:
    def test_market_tools_share_data_timeout(self):
        from src.agent.factory import _build_category_timeout_map

        config = SimpleNamespace(
            agent_data_tool_timeout_s=15.0,
            agent_search_tool_timeout_s=0.0,
            agent_analysis_tool_timeout_s=0.0,
            agent_action_tool_timeout_s=0.0,
        )
        mapping = _build_category_timeout_map(config)
        assert mapping["data"] == 15.0
        assert mapping["market"] == 15.0
        assert "search" not in mapping


# ---------------------------------------------------------------------------
# 7. Tool registry cache invalidation (Issue #1890 follow-up review)
# ---------------------------------------------------------------------------
class TestToolRegistryCacheInvalidation:
    """The module-level ``_TOOL_REGISTRY`` cache must not shadow a reloaded
    ``Config`` instance.  These tests pin the contract:

    * ``reset_tool_registry`` empties the cache.
    * ``get_tool_registry(config)`` rebuilds when the resolved per-category
      timeouts differ from the ones the cached registry was built with, and
      does so even when the new ``Config`` instance happens to reuse the
      collected instance's ``id()``.
    * ``SystemConfigService._reload_runtime_singletons`` actually invokes
      ``reset_tool_registry`` so API/scheduler/bot entry-points observe the
      fresh ``AGENT_*_TOOL_TIMEOUT_S`` values.
    """

    @pytest.fixture(autouse=True)
    def _isolate_factory_module_state(self):
        """Snapshot/restore ``factory._TOOL_REGISTRY`` so this class cannot
        bleed cache state across other tests in the same pytest process.
        """
        from src.agent import factory

        saved_registry = factory._TOOL_REGISTRY
        saved_timeout_map = factory._CACHED_TIMEOUT_MAP
        factory.reset_tool_registry()
        try:
            yield
        finally:
            factory._TOOL_REGISTRY = saved_registry
            factory._CACHED_TIMEOUT_MAP = saved_timeout_map

    def _empty_tool_lists(self):
        """Patch the ALL_*_TOOLS module-level names on ``factory`` so the
        rebuild loop is a no-op and the test stays independent of the
        full tool-registration chain.
        """
        from src.agent import factory

        return (
            patch.object(factory, "ALL_DATA_TOOLS", [], create=True),
            patch.object(factory, "ALL_ANALYSIS_TOOLS", [], create=True),
            patch.object(factory, "ALL_SEARCH_TOOLS", [], create=True),
            patch.object(factory, "ALL_MARKET_TOOLS", [], create=True),
            patch.object(factory, "ALL_BACKTEST_TOOLS", [], create=True),
        )

    def test_reset_tool_registry_clears_module_cache(self):
        from src.agent import factory

        # Sanity: initial reset empties the module-level cache.
        factory.reset_tool_registry()
        assert factory._TOOL_REGISTRY is None
        assert factory._CACHED_TIMEOUT_MAP is None

        # Idempotent: calling reset on an empty cache is a no-op.
        factory.reset_tool_registry()
        assert factory._TOOL_REGISTRY is None
        assert factory._CACHED_TIMEOUT_MAP is None

    def test_get_tool_registry_rebuilds_when_config_changes(self):
        """Two callers passing two *distinct* ``Config`` instances must
        observe the matching per-category timeouts on each rebuild — never
        a stale view from the first build.
        """
        from src.agent import factory

        config_a = SimpleNamespace(
            agent_data_tool_timeout_s=10.0,
            agent_search_tool_timeout_s=0.0,
            agent_analysis_tool_timeout_s=0.0,
            agent_action_tool_timeout_s=0.0,
        )
        config_b = SimpleNamespace(
            agent_data_tool_timeout_s=25.0,
            agent_search_tool_timeout_s=5.0,
            agent_analysis_tool_timeout_s=0.0,
            agent_action_tool_timeout_s=0.0,
        )
        # Force the configs to be distinct objects (SimpleNamespace instances
        # already have distinct id() unless they're literal `()`).
        assert id(config_a) != id(config_b)

        empty_lists = self._empty_tool_lists()
        with patch.object(
            factory, "_build_category_timeout_map"
        ) as build_map, patch(
            "src.agent.tools.registry.ToolRegistry"
        ) as registry_cls, empty_lists[0], empty_lists[1], empty_lists[2], empty_lists[3], empty_lists[4]:
            registry_instance_a = MagicMock(name="registry_a")
            registry_instance_b = MagicMock(name="registry_b")
            registry_cls.side_effect = [registry_instance_a, registry_instance_b]
            build_map.side_effect = lambda cfg: _build_category_timeout_map(cfg)

            first = factory.get_tool_registry(config_a)
            second = factory.get_tool_registry(config_b)

            # Two distinct registry objects — never the cached first build.
            assert first is registry_instance_a
            assert second is registry_instance_b
            assert first is not second
            # And the cache now points at the *second* one.
            assert factory._TOOL_REGISTRY is registry_instance_b
            assert factory._CACHED_TIMEOUT_MAP == {
                "data": 25.0,
                "search": 5.0,
                "market": 25.0,
            }
            # ToolRegistry was constructed with the *matching* timeout map.
            assert registry_cls.call_args_list[0].kwargs["category_timeout_map"] == {
                "data": 10.0,
                "market": 10.0,
            }
            assert registry_cls.call_args_list[1].kwargs["category_timeout_map"] == {
                "data": 25.0,
                "search": 5.0,
                "market": 25.0,
            }

    def test_get_tool_registry_reuses_cache_for_equivalent_config(self):
        """Repeated calls carrying the same effective timeouts must reuse the
        cache so the tool-registration cost is paid at most once.

        The final call passes a *different object* with identical values: the
        registry only depends on the resolved timeout map, so rebuilding it
        would be pure waste on every request.
        """
        from src.agent import factory

        config = SimpleNamespace(
            agent_data_tool_timeout_s=10.0,
            agent_search_tool_timeout_s=0.0,
            agent_analysis_tool_timeout_s=0.0,
            agent_action_tool_timeout_s=0.0,
        )
        equivalent_config = SimpleNamespace(
            agent_data_tool_timeout_s=10.0,
            agent_search_tool_timeout_s=0.0,
            agent_analysis_tool_timeout_s=0.0,
            agent_action_tool_timeout_s=0.0,
        )
        empty_lists = self._empty_tool_lists()
        with patch.object(
            factory, "_build_category_timeout_map"
        ) as build_map, patch(
            "src.agent.tools.registry.ToolRegistry"
        ) as registry_cls, empty_lists[0], empty_lists[1], empty_lists[2], empty_lists[3], empty_lists[4]:
            registry_instance = MagicMock(name="registry")
            registry_cls.return_value = registry_instance
            build_map.side_effect = lambda cfg: _build_category_timeout_map(cfg)

            first = factory.get_tool_registry(config)
            second = factory.get_tool_registry(config)
            third = factory.get_tool_registry(equivalent_config)

            assert first is second is third is registry_instance
            # ToolRegistry was instantiated exactly once across three calls.
            assert registry_cls.call_count == 1

    def test_build_agent_executor_forwards_config_to_registry(self):
        """Regression for review blockers OR-COM-dd1e8fa7 / OR-COM-bff42110: the
        builder must hand its *caller-supplied* ``config`` to
        ``get_tool_registry`` so a distinct / updated config actually re-binds
        the category timeouts instead of silently reusing the first cached
        registry built from a frozen default config.
        """
        from src.agent import factory

        config = SimpleNamespace(
            agent_data_tool_timeout_s=12.0,
            agent_search_tool_timeout_s=0.0,
            agent_analysis_tool_timeout_s=0.0,
            agent_action_tool_timeout_s=0.0,
            agent_arch="single",
        )
        with patch.object(factory, "get_tool_registry") as gtr, patch.object(
            factory, "resolve_skill_prompt_state"
        ) as rsp, patch("src.agent.llm_adapter.LLMToolAdapter"), patch(
            "src.agent.executor.AgentExecutor"
        ) as ae_cls:
            gtr.return_value = MagicMock(name="registry")
            rsp.return_value = MagicMock(skill_manager=MagicMock())
            factory.build_agent_executor(config)
            # The registry is (re)built from THIS config, not a frozen default.
            assert gtr.call_args == ((config,),)
            assert ae_cls.call_args.kwargs["tool_registry"] is gtr.return_value

    def test_rebuild_survives_config_instance_id_reuse(self):
        """A reloaded ``Config`` must be honoured even when CPython hands the
        new instance the *same* ``id()`` as the collected one.

        ``Config.reset_instance()`` drops the only strong reference, so the
        allocator frequently reuses that address for the replacement instance.
        Keying the cache on ``id(config)`` would read that as "unchanged" and
        keep serving the stale registry — exactly the bug this class guards.
        """
        from src.agent import factory

        empty_lists = self._empty_tool_lists()
        with patch.object(
            factory, "_build_category_timeout_map"
        ) as build_map, patch(
            "src.agent.tools.registry.ToolRegistry"
        ) as registry_cls, empty_lists[0], empty_lists[1], empty_lists[2], empty_lists[3], empty_lists[4]:
            registry_old = MagicMock(name="registry_old")
            registry_new = MagicMock(name="registry_new")
            registry_cls.side_effect = [registry_old, registry_new]
            build_map.side_effect = lambda cfg: _build_category_timeout_map(cfg)

            config_old = SimpleNamespace(
                agent_data_tool_timeout_s=10.0,
                agent_search_tool_timeout_s=0.0,
                agent_analysis_tool_timeout_s=0.0,
                agent_action_tool_timeout_s=0.0,
            )
            factory.get_tool_registry(config_old)
            assert factory._TOOL_REGISTRY is registry_old

            # Simulate the worst case rather than relying on the allocator:
            # a new config whose id() collides with the collected instance.
            stale_id = id(config_old)
            del config_old
            gc.collect()

            config_new = SimpleNamespace(
                agent_data_tool_timeout_s=45.0,
                agent_search_tool_timeout_s=0.0,
                agent_analysis_tool_timeout_s=0.0,
                agent_action_tool_timeout_s=0.0,
            )
            # Force a deterministic id() collision so this doubles as a real
            # regression guard: a future implementation that re-keys the cache
            # on ``id(config)`` would read the new config as "unchanged" and
            # wrongly reuse ``registry_old``.  Patching ``builtins.id`` (not
            # ``factory.id``) is required because ``get_tool_registry`` uses the
            # builtin directly.
            with patch("builtins.id", return_value=stale_id):
                rebuilt = factory.get_tool_registry(config_new)

            # Value-keyed cache still notices the change.
            assert rebuilt is registry_new
            assert registry_cls.call_args_list[-1].kwargs["category_timeout_map"] == {
                "data": 45.0,
                "market": 45.0,
            }

    def test_get_tool_registry_tolerates_partial_config_objects(self):
        """Callers hand ``get_tool_registry`` whatever config they hold, which
        in tests is routinely a ``MagicMock`` or a stub missing the timeout
        attributes.  Neither may explode: ``MagicMock() > 0`` raises
        ``TypeError`` and a bare stub raises ``AttributeError``, so the
        resolver coerces both to "no category limit".
        """

        class _StubConfig:
            """No AGENT_*_TOOL_TIMEOUT_S attributes at all."""

        assert _build_category_timeout_map(MagicMock()) == {}
        assert _build_category_timeout_map(_StubConfig()) == {}

        # Garbage values degrade to "no limit" rather than propagating.
        noisy = SimpleNamespace(
            agent_data_tool_timeout_s="not-a-number",
            agent_search_tool_timeout_s=None,
            agent_analysis_tool_timeout_s=7.5,
            agent_action_tool_timeout_s=-3.0,
        )
        assert _build_category_timeout_map(noisy) == {"analysis": 7.5}

    def test_reload_runtime_singletons_drops_registry(self):
        """``SystemConfigService._reload_runtime_singletons`` must include
        ``reset_tool_registry``; otherwise long-running API/scheduler/bot
        processes will keep using the first build's per-category timeouts.
        """
        from src.agent import factory
        from src.services import system_config_service

        with patch.object(factory, "reset_tool_registry") as reset_mock:
            system_config_service.SystemConfigService._reload_runtime_singletons()

        reset_mock.assert_called_once()

    def test_reload_runtime_singletons_actually_invalidates_cached_registry(self):
        """End-to-end: with the cache populated against config_old, calling
        ``_reload_runtime_singletons`` must clear it so the next
        ``get_tool_registry(config_new)`` rebuilds against the new config.
        """
        from src.agent import factory

        config_old = SimpleNamespace(
            agent_data_tool_timeout_s=10.0,
            agent_search_tool_timeout_s=0.0,
            agent_analysis_tool_timeout_s=0.0,
            agent_action_tool_timeout_s=0.0,
        )
        config_new = SimpleNamespace(
            agent_data_tool_timeout_s=99.0,
            agent_search_tool_timeout_s=0.0,
            agent_analysis_tool_timeout_s=0.0,
            agent_action_tool_timeout_s=0.0,
        )

        empty_lists = self._empty_tool_lists()
        with patch.object(
            factory, "_build_category_timeout_map"
        ) as build_map, patch(
            "src.agent.tools.registry.ToolRegistry"
        ) as registry_cls, empty_lists[0], empty_lists[1], empty_lists[2], empty_lists[3], empty_lists[4]:
            registry_old = MagicMock(name="registry_old")
            registry_new = MagicMock(name="registry_new")
            registry_cls.side_effect = [registry_old, registry_new]
            build_map.side_effect = lambda cfg: _build_category_timeout_map(cfg)

            from src.services import system_config_service

            # 1) Prime the cache against the old config.
            factory.get_tool_registry(config_old)
            assert factory._TOOL_REGISTRY is registry_old

            # 2) Simulate a runtime reload — the service-side hook clears
            #    the cache (real call, not mocked, so we exercise the wiring).
            system_config_service.SystemConfigService._reload_runtime_singletons()
            assert factory._TOOL_REGISTRY is None
            assert factory._CACHED_TIMEOUT_MAP is None

            # 3) Next access, with the new config, must rebuild and observe
            #    the new timeout.
            rebuilt = factory.get_tool_registry(config_new)
            assert rebuilt is registry_new
            assert factory._TOOL_REGISTRY is registry_new
            assert registry_cls.call_args_list[-1].kwargs["category_timeout_map"] == {
                "data": 99.0,
                "market": 99.0,
            }
