# -*- coding: utf-8 -*-
"""
Shared runner — extracted LLM + tool execution loop.

Provides ``run_agent_loop``, the single authoritative implementation of the
ReAct execute-loop that was previously inlined inside ``AgentExecutor._run_loop``.
All current and future agents should delegate to this runner instead of
re-implementing the loop themselves.

Design goals:
- Keep the same observable behaviour as the original ``_run_loop``
- Accept pluggable callbacks for progress, message history, and result handling
- Remain stateless — all mutable state lives in the caller
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import threading
import contextvars
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.agent.llm_adapter import LLMToolAdapter
from src.agent.dashboard_payload import sanitize_agent_dashboard_payload
from src.agent.protocols import StageFailureReason
from src.agent.stream_events import stream_event
from src.agent.tools.registry import ToolRegistry
from src.agent.tools.execution import (
    TOOL_CANCEL_EVENT,
    _build_tool_cache_key,
    _guard_tool_stock_scope,
    _is_non_retriable_tool_result,
    _is_stock_scoped_tool,
    _normalize_guard_stock_code,
    _normalize_tool_stock_code,
    execute_runner_tool_call,
    serialize_tool_result,
)
from src.agent.stock_scope import StockScope
from src.llm.usage import should_persist_usage_telemetry
from src.utils.data_processing import normalize_report_signal_attribution
from src.storage import persist_llm_usage as _persist_usage

logger = logging.getLogger(__name__)

__all__ = [
    "RunLoopResult",
    "parse_dashboard_json",
    "run_agent_loop",
    "serialize_tool_result",
    "try_parse_json",
    "_build_tool_cache_key",
    "_guard_tool_stock_scope",
    "_is_non_retriable_tool_result",
    "_is_stock_scoped_tool",
    "_normalize_guard_stock_code",
    "_normalize_tool_stock_code",
]

# Tool name → friendly label for progress messages
_THINKING_TOOL_LABELS: Dict[str, str] = {
    "get_realtime_quote": "行情获取",
    "get_daily_history": "K线数据获取",
    "analyze_trend": "技术指标分析",
    "get_chip_distribution": "筹码分布分析",
    "search_stock_news": "新闻搜索",
    "search_comprehensive_intel": "综合情报搜索",
    "get_market_indices": "市场概览获取",
    "get_sector_rankings": "行业板块分析",
    "get_analysis_context": "历史分析上下文",
    "get_stock_info": "基本信息获取",
    "analyze_pattern": "K线形态识别",
    "get_volume_analysis": "量能分析",
    "calculate_ma": "均线计算",
    "get_skill_backtest_summary": "技能回测概览",
    "get_strategy_backtest_summary": "策略回测概览",
    "get_stock_backtest_summary": "个股回测数据",
}


# ============================================================
# RunLoopResult — the output of one run_agent_loop invocation
# ============================================================

@dataclass
class RunLoopResult:
    """Output produced by :func:`run_agent_loop`."""

    success: bool = False
    content: str = ""
    tool_calls_log: List[Dict[str, Any]] = field(default_factory=list)
    total_steps: int = 0
    total_tokens: int = 0
    provider: str = ""
    models_used: List[str] = field(default_factory=list)
    error: Optional[str] = None
    failure_reason: Optional[StageFailureReason] = None
    # Raw messages list at the end of the loop (callers may want to persist)
    messages: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def model(self) -> str:
        """Comma-separated de-duplicated model names used during the run."""
        return ", ".join(dict.fromkeys(m for m in self.models_used if m))


# ============================================================
# Helpers
# ============================================================

def parse_dashboard_json(content: str) -> Optional[Dict[str, Any]]:
    """Extract and parse a Decision Dashboard JSON from agent text.

    Tries multiple strategies:
    1. Markdown code blocks (```json ... ```)
    2. Raw JSON parse
    3. ``json_repair`` library
    4. Brace-delimited substring
    """
    if not content:
        return None

    from json_repair import repair_json

    # Strategy 1: markdown code blocks
    json_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if json_blocks:
        for block in json_blocks:
            parsed = _try_parse_json(block)
            if parsed is not None:
                return _finalize_dashboard_payload(parsed)
            parsed = _try_repair_json(block, repair_json)
            if parsed is not None:
                return _finalize_dashboard_payload(parsed)

    # Strategy 2: raw parse
    parsed = _try_parse_json(content)
    if parsed is not None:
        return _finalize_dashboard_payload(parsed)

    # Strategy 3: json_repair on full content
    parsed = _try_repair_json(content, repair_json)
    if parsed is not None:
        return _finalize_dashboard_payload(parsed)

    # Strategy 4: brace-delimited
    brace_start = content.find("{")
    brace_end = content.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        candidate = content[brace_start : brace_end + 1]
        parsed = _try_parse_json(candidate)
        if parsed is not None:
            return _finalize_dashboard_payload(parsed)
        parsed = _try_repair_json(candidate, repair_json)
        if parsed is not None:
            return _finalize_dashboard_payload(parsed)

    logger.warning("Failed to parse dashboard JSON from agent response")
    return None


def _finalize_dashboard_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize reserved fields before running normal dashboard normalization."""
    sanitized = sanitize_agent_dashboard_payload(payload)
    normalize_report_signal_attribution(sanitized)
    return sanitized


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON dict extraction from LLM text.

    Handles:
    1. Direct JSON parse
    2. Markdown code fences (```json ... ```)
    3. Brace-delimited substring
    4. ``json_repair`` fallback for slightly malformed JSON

    This is the shared utility that all agent ``post_process`` methods
    should use instead of duplicating the same logic.
    """
    if not text:
        return None

    candidates: List[str] = []
    cleaned = text.strip()
    if cleaned:
        candidates.append(cleaned)

    if cleaned.startswith("```"):
        unfenced = re.sub(r'^```(?:json)?\s*', '', cleaned)
        unfenced = re.sub(r'\s*```$', '', unfenced)
        if unfenced:
            candidates.append(unfenced.strip())

    fenced_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    for block in fenced_blocks:
        block = block.strip()
        if block:
            candidates.append(block)

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start:end + 1].strip()
        if snippet:
            candidates.append(snippet)

    seen: set[str] = set()
    unique_candidates: List[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue

    try:
        from json_repair import repair_json
    except Exception:
        repair_json = None

    if repair_json is not None:
        for candidate in unique_candidates:
            repaired = _try_repair_json(candidate, repair_json)
            if repaired is not None:
                return repaired

    return None


# Keep private alias used internally by parse_dashboard_json
_try_parse_json = try_parse_json


def _try_repair_json(text: str, repair_fn: Callable) -> Optional[Dict[str, Any]]:
    try:
        repaired = repair_fn(text)
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _remaining_timeout_seconds(
    start_time: float,
    max_wall_clock_seconds: Optional[float],
) -> Optional[float]:
    """Return remaining wall-clock budget in seconds, or None when disabled."""
    if max_wall_clock_seconds is None or max_wall_clock_seconds <= 0:
        return None
    return max(0.0, float(max_wall_clock_seconds) - (time.time() - start_time))


def _build_timeout_result(
    *,
    start_time: float,
    max_wall_clock_seconds: float,
    step: int,
    tool_calls_log: List[Dict[str, Any]],
    total_tokens: int,
    provider_used: str,
    models_used: List[str],
    messages: List[Dict[str, Any]],
) -> RunLoopResult:
    elapsed = time.time() - start_time
    return RunLoopResult(
        success=False,
        content="",
        tool_calls_log=tool_calls_log,
        total_steps=step,
        total_tokens=total_tokens,
        provider=provider_used,
        models_used=models_used,
        error=f"Agent timed out after {elapsed:.2f}s (limit: {max_wall_clock_seconds:.2f}s)",
        failure_reason=StageFailureReason.TIMEOUT,
        messages=messages,
    )


def _build_budget_guard_result(
    *,
    start_time: float,
    step: int,
    tool_calls_log: List[Dict[str, Any]],
    total_tokens: int,
    provider_used: str,
    models_used: List[str],
    messages: List[Dict[str, Any]],
    remaining_timeout_s: float,
    min_step_budget_s: float,
) -> RunLoopResult:
    elapsed = time.time() - start_time
    return RunLoopResult(
        success=False,
        content="",
        tool_calls_log=tool_calls_log,
        total_steps=step,
        total_tokens=total_tokens,
        provider=provider_used,
        models_used=models_used,
        error=(
            "Agent step skipped due to insufficient budget: "
            f"{remaining_timeout_s:.2f}s remaining, minimum {min_step_budget_s:.1f}s required"
        ),
        failure_reason=StageFailureReason.BUDGET_SKIP,
        messages=messages,
    )


# ============================================================
# Core loop
# ============================================================

def run_agent_loop(
    *,
    messages: List[Dict[str, Any]],
    tool_registry: ToolRegistry,
    llm_adapter: LLMToolAdapter,
    max_steps: int = 10,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    thinking_labels: Optional[Dict[str, str]] = None,
    max_wall_clock_seconds: Optional[float] = None,
    tool_call_timeout_seconds: Optional[float] = None,
    stock_scope: Optional[StockScope] = None,
    emit_stage_events: bool = True,
) -> RunLoopResult:
    """Execute the ReAct LLM ↔ tool loop.

    This is the *single shared implementation* of the agent execution loop.
    Both the legacy ``AgentExecutor`` and any future multi-agent runner
    should delegate here.

    Args:
        messages: The initial message list (system + user + optional history).
                  **Mutated in-place** — tool results are appended.
        tool_registry: Registry of callable tools.
        llm_adapter: LLM backend (handles multi-provider fallback).
        max_steps: Maximum number of LLM round-trips.
        progress_callback: Optional callback receiving progress dicts.
        thinking_labels: Override map of tool_name → friendly label.
        max_wall_clock_seconds: Optional overall timeout budget for the loop.
        tool_call_timeout_seconds: Optional timeout for one parallel tool batch.
        emit_stage_events: Whether to emit the synthetic ``agent_loop``
            stage lifecycle. Orchestrated business stages disable this so
            ``stage_start`` / ``stage_done`` only describe real stages.

    Returns:
        A :class:`RunLoopResult` with the final content, stats, and the
        (mutated) messages list.
    """
    labels = thinking_labels or _THINKING_TOOL_LABELS
    tool_decls = tool_registry.to_openai_tools()

    start_time = time.time()
    tool_calls_log: List[Dict[str, Any]] = []
    non_retriable_tool_results: Dict[str, str] = {}
    total_tokens = 0
    provider_used = ""
    models_used: List[str] = []

    # Minimum seconds needed for a meaningful LLM round-trip.  If the
    # remaining budget is positive but below this threshold, the step will
    # almost certainly timeout mid-call, wasting a billed request.  Only
    # enforced from step 2 onwards so the first step always gets a chance
    # even when the total budget is small.
    _MIN_STEP_BUDGET_S = 8.0

    def _finish(result: RunLoopResult) -> RunLoopResult:
        if progress_callback and emit_stage_events:
            progress_callback(
                stream_event(
                    "stage_done",
                    stage="agent_loop",
                    status="completed" if result.success else "failed",
                    duration=round(time.time() - start_time, 2),
                )
            )
        return result

    if progress_callback and emit_stage_events:
        progress_callback(
            stream_event(
                "stage_start",
                stage="agent_loop",
                message="Starting agent analysis...",
            )
        )

    for step in range(max_steps):
        remaining_timeout = _remaining_timeout_seconds(start_time, max_wall_clock_seconds)
        timeout_exhausted = remaining_timeout is not None and remaining_timeout <= 0
        budget_guard_triggered = (
            not timeout_exhausted
            and remaining_timeout is not None
            and step > 0
            and remaining_timeout <= _MIN_STEP_BUDGET_S
        )
        if timeout_exhausted or budget_guard_triggered:
            if budget_guard_triggered:
                logger.warning(
                    "Agent budget too low for step %d (%.1fs remaining, min %.1fs)",
                    step + 1,
                    remaining_timeout,
                    _MIN_STEP_BUDGET_S,
                )
                return _finish(_build_budget_guard_result(
                    start_time=start_time,
                    step=step,
                    tool_calls_log=tool_calls_log,
                    total_tokens=total_tokens,
                    provider_used=provider_used,
                    models_used=models_used,
                    messages=messages,
                    remaining_timeout_s=remaining_timeout,
                    min_step_budget_s=_MIN_STEP_BUDGET_S,
                ))

            if remaining_timeout <= 0:
                logger.warning("Agent timed out before step %d", step + 1)
            return _finish(_build_timeout_result(
                start_time=start_time,
                max_wall_clock_seconds=float(max_wall_clock_seconds),
                step=step,
                tool_calls_log=tool_calls_log,
                total_tokens=total_tokens,
                provider_used=provider_used,
                models_used=models_used,
                messages=messages,
            ))

        logger.info("Agent step %d/%d", step + 1, max_steps)

        # --- progress: thinking ---
        if progress_callback:
            if not tool_calls_log:
                thinking_msg = "正在制定分析路径..."
            else:
                last_tool = tool_calls_log[-1].get("tool", "")
                label = labels.get(last_tool, last_tool)
                thinking_msg = f"「{label}」已完成，继续深入分析..."
            progress_callback(stream_event("thinking", step=step + 1, message=thinking_msg))

        # --- LLM call ---
        response = llm_adapter.call_with_tools(
            messages,
            tool_decls,
            timeout=remaining_timeout,
        )
        provider_used = response.provider
        total_tokens += (response.usage or {}).get("total_tokens", 0)
        m = getattr(response, "model", "") or response.provider
        if m and m != "error":
            models_used.append(m)
        model_for_usage = m or response.provider
        if model_for_usage and model_for_usage != "error" and should_persist_usage_telemetry(response.usage):
            _persist_usage(response.usage, model_for_usage, call_type="agent")

        remaining_timeout = _remaining_timeout_seconds(start_time, max_wall_clock_seconds)
        if remaining_timeout is not None and remaining_timeout <= 0:
            logger.warning("Agent timed out after LLM call at step %d", step + 1)
            return _finish(_build_timeout_result(
                start_time=start_time,
                max_wall_clock_seconds=float(max_wall_clock_seconds),
                step=step + 1,
                tool_calls_log=tool_calls_log,
                total_tokens=total_tokens,
                provider_used=provider_used,
                models_used=models_used,
                messages=messages,
            ))

        if response.tool_calls:
            # ---- tool execution branch ----
            logger.info(
                "Agent requesting %d tool call(s): %s",
                len(response.tool_calls),
                [tc.name for tc in response.tool_calls],
            )

            # Append assistant message (with tool_calls) to history
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
                "_trace_provider": response.provider,
                "_trace_model": m,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        **({"provider_specific_fields": tc.provider_specific_fields} if tc.provider_specific_fields else {}),
                        **({"thought_signature": tc.thought_signature} if tc.thought_signature is not None else {}),
                    }
                    for tc in response.tool_calls
                ],
            }
            if response.reasoning_content is not None:
                assistant_msg["reasoning_content"] = response.reasoning_content
            if response.provider_blocks:
                assistant_msg["provider_blocks"] = response.provider_blocks
            messages.append(assistant_msg)

            # Execute tools (parallel when > 1)
            effective_tool_timeout = tool_call_timeout_seconds
            if remaining_timeout is not None:
                effective_tool_timeout = min(
                    remaining_timeout,
                    tool_call_timeout_seconds if tool_call_timeout_seconds and tool_call_timeout_seconds > 0 else remaining_timeout,
                )
            tool_results = _execute_tools(
                response.tool_calls,
                tool_registry,
                step + 1,
                progress_callback,
                tool_calls_log,
                non_retriable_tool_results,
                tool_wait_timeout_seconds=effective_tool_timeout,
                stock_scope=stock_scope,
            )

            # Append tool results preserving original call order
            tc_order = {tc.id: i for i, tc in enumerate(response.tool_calls)}
            tool_results.sort(key=lambda x: tc_order.get(x["tc"].id, 0))
            for tr in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "name": tr["tc"].name,
                        "tool_call_id": tr["tc"].id,
                        "content": tr["result_str"],
                    }
                )

            remaining_timeout = _remaining_timeout_seconds(start_time, max_wall_clock_seconds)
            if remaining_timeout is not None and remaining_timeout <= 0:
                logger.warning("Agent timed out after tool execution at step %d", step + 1)
                return _finish(_build_timeout_result(
                    start_time=start_time,
                    max_wall_clock_seconds=float(max_wall_clock_seconds),
                    step=step + 1,
                    tool_calls_log=tool_calls_log,
                    total_tokens=total_tokens,
                    provider_used=provider_used,
                    models_used=models_used,
                    messages=messages,
                ))

        else:
            # ---- final answer branch ----
            logger.info(
                "Agent completed in %d steps (%.1fs, %d tokens)",
                step + 1,
                time.time() - start_time,
                total_tokens,
            )
            if progress_callback:
                progress_callback(stream_event("generating", step=step + 1, message="正在生成最终分析..."))

            final_content = response.content or ""
            is_error = response.provider == "error"

            return _finish(RunLoopResult(
                success=not is_error and bool(final_content),
                content=final_content if not is_error else "",
                tool_calls_log=tool_calls_log,
                total_steps=step + 1,
                total_tokens=total_tokens,
                provider=provider_used,
                models_used=models_used,
                error=final_content if is_error else None,
                failure_reason=(StageFailureReason.STAGE_FAILURE if is_error else None),
                messages=messages,
            ))

    # Max steps exceeded
    logger.warning("Agent hit max steps (%d)", max_steps)
    return _finish(RunLoopResult(
        success=False,
        content="",
        tool_calls_log=tool_calls_log,
        total_steps=max_steps,
        total_tokens=total_tokens,
        provider=provider_used,
        models_used=models_used,
        error=f"Agent exceeded max steps ({max_steps}). Try increasing AGENT_MAX_STEPS if analysis tasks are complex.",
        failure_reason=StageFailureReason.STAGE_FAILURE,
        messages=messages,
    ))


# ============================================================
# Internal tool execution
# ============================================================

def _coerce_positive_timeout(value) -> Optional[float]:
    """Coerce a timeout candidate to a positive finite float, else ``None``.

    ``None`` / non-numeric / non-positive / ``inf`` / ``nan`` all map to
    ``None`` ("no limit at this level").  Rejecting non-finite values prevents
    an ``OverflowError`` from ``future.result(timeout=inf)`` and avoids the
    undefined ordering a ``nan`` would produce.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return v


def _build_timeout_result_payload(
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_s: float,
    non_retriable_tool_results: Optional[Dict[str, str]],
) -> str:
    """Build the timeout-shaped result string and record it as non-retriable.

    A timed-out call is marked ``"retriable": False`` and inserted into
    ``non_retriable_tool_results`` so the LLM's retry of the *same* call reuses
    the cached failure instead of spinning up a second (possibly side-effecting)
    execution — a best-effort guard against duplicate work, since Python cannot
    forcibly cancel an already-started tool thread.
    """
    label = f"{timeout_s:.2f}s"
    result_str = json.dumps({
        "error": f"Tool execution timed out after {label}",
        "timeout": True,
        "retriable": False,
    })
    if non_retriable_tool_results is not None:
        non_retriable_tool_results[_build_tool_cache_key(tool_name, arguments)] = result_str
    return result_str


def _resolve_per_tool_timeout(
    tool_call,
    tool_registry: Optional[ToolRegistry],
    global_timeout: Optional[float],
) -> Optional[float]:
    """Resolve the effective timeout for a single tool call (Issue #1890).

    Precedence is *first-wins within the registry fallback*, then the global
    ``tool_call_timeout_seconds`` / remaining wall-clock budget acts as an
    unbreakable outer cap:

        1. explicit per-tool timeout (``ToolDefinition.timeout_seconds``)
        2. category default (``AGENT_*_TOOL_TIMEOUT_S``)
        3. none (no per-tool limit)

    The explicit per-tool value therefore has the **highest** precedence and is
    never lowered by a smaller category default — the previous ``min()`` across
    all levels silently let a category default override an explicit per-tool
    declaration.  The global budget still caps the result, so a long per-tool
    timeout can never exceed the caller's overall budget.
    """
    global_budget = _coerce_positive_timeout(global_timeout)
    if tool_registry is None:
        return global_budget
    tool_def = tool_registry.get(tool_call.name)
    if tool_def is None:
        return global_budget

    per_tool = _coerce_positive_timeout(getattr(tool_def, "timeout_seconds", None))
    category = _coerce_positive_timeout(tool_registry.category_default_timeout(tool_def.category))

    # First-wins registry fallback: explicit per-tool > category default.
    base = per_tool if per_tool is not None else category

    if base is None:
        # No per-tool/category limit; the global budget governs.
        return global_budget
    if global_budget is not None:
        return min(base, global_budget)
    return base


def _execute_tools(
    tool_calls,
    tool_registry: ToolRegistry,
    step: int,
    progress_callback: Optional[Callable],
    tool_calls_log: List[Dict[str, Any]],
    non_retriable_tool_results: Optional[Dict[str, str]] = None,
    tool_wait_timeout_seconds: Optional[float] = None,
    stock_scope: Optional[StockScope] = None,
) -> List[Dict[str, Any]]:
    """Execute one or more tool calls, returning ordered result dicts.

    Single tools run inline; multiple tools run in parallel threads.
    """

    def _exec_single(tc_item):
        return execute_runner_tool_call(
            tool_call=tc_item,
            tool_registry=tool_registry,
            stock_scope=stock_scope,
            non_retriable_tool_results=non_retriable_tool_results,
        )

    def _exec_single_with_timeout(tc_item, per_tool_timeout, cancel_event=None):
        """Run a single tool call with an optional per-tool timeout.

        Returns ``(result_6tuple, timed_out)``.  When the per-tool timeout
        fires, a timeout-shaped 6-tuple is returned and ``timed_out`` is
        ``True`` so the outer batch loop treats it as a completed result
        rather than raising ``FuturesTimeoutError``.  A ``cancel_event``
        (created here when not supplied) is armed on timeout so a still-running
        handler can cooperate via ``is_tool_cancellation_requested()``.
        """
        if not per_tool_timeout or per_tool_timeout <= 0:
            return _exec_single(tc_item), False
        if cancel_event is None:
            cancel_event = threading.Event()
        pool = ThreadPoolExecutor(max_workers=1)
        ctx = contextvars.copy_context()
        ctx.run(TOOL_CANCEL_EVENT.set, cancel_event)
        try:
            future = pool.submit(ctx.run, _exec_single, tc_item)
            try:
                return future.result(timeout=per_tool_timeout), False
            except FuturesTimeoutError:
                future.cancel()
                cancel_event.set()
                result_str = _build_timeout_result_payload(
                    tc_item.name, tc_item.arguments, per_tool_timeout, non_retriable_tool_results,
                )
                return (tc_item, result_str, False, round(per_tool_timeout, 2), False, None), True
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        except Exception:  # pragma: no cover - defensive
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    results: List[Dict[str, Any]] = []

    if len(tool_calls) == 1:
        tc = tool_calls[0]
        if progress_callback:
            progress_callback(stream_event("tool_start", step=step, tool=tc.name))
        per_tool_timeout = _resolve_per_tool_timeout(tc, tool_registry, tool_wait_timeout_seconds)
        timeout_triggered = False
        if per_tool_timeout and per_tool_timeout > 0:
            pool = ThreadPoolExecutor(max_workers=1)
            ctx = contextvars.copy_context()
            cancel_event = threading.Event()
            ctx.run(TOOL_CANCEL_EVENT.set, cancel_event)
            try:
                future = pool.submit(ctx.run, _exec_single, tc)
                try:
                    _, result_str, success, dur, cached, guard_result = future.result(timeout=per_tool_timeout)
                except FuturesTimeoutError:
                    timeout_triggered = True
                    future.cancel()
                    cancel_event.set()
                    logger.warning("Tool '%s' timed out after %.2fs at step %d", tc.name, per_tool_timeout, step)
                    result_str = _build_timeout_result_payload(
                        tc.name, tc.arguments, per_tool_timeout, non_retriable_tool_results,
                    )
                    success = False
                    dur = round(per_tool_timeout, 2)
                    cached = False
                    guard_result = None
            finally:
                pool.shutdown(wait=not timeout_triggered, cancel_futures=timeout_triggered)
        else:
            _, result_str, success, dur, cached, guard_result = _exec_single(tc)
        if progress_callback:
            progress_callback(stream_event("tool_done", step=step, tool=tc.name, success=success, duration=dur))
        log_entry = {
            "step": step, "tool": tc.name, "arguments": tc.arguments,
            "success": success, "duration": dur, "result_length": len(result_str),
            "cached": cached,
        }
        if per_tool_timeout and per_tool_timeout > 0 and not success:
            try:
                if json.loads(result_str).get("timeout") is True:
                    log_entry["timeout"] = True
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        if guard_result is not None:
            log_entry.update({
                "guarded": True,
                "expected_stock_code": guard_result.get("expected_stock_code"),
                "requested_stock_code": guard_result.get("requested_stock_code"),
                "allowed_stock_codes": guard_result.get("allowed_stock_codes", []),
            })
        tool_calls_log.append(log_entry)
        results.append({"tc": tc, "result_str": result_str})
    else:
        for tc in tool_calls:
            if progress_callback:
                progress_callback(stream_event("tool_start", step=step, tool=tc.name))

        pool = ThreadPoolExecutor(max_workers=min(len(tool_calls), 5))
        timeout_triggered = False
        # Keyed by Future (not by tool name) so that two parallel calls to the
        # same tool each get their own cancel event — a name-keyed dict would let
        # one call's event shadow the other's on a batch timeout.
        future_cancel: Dict = {}
        try:
            futures = {}
            for tc in tool_calls:
                per_tool_timeout = _resolve_per_tool_timeout(tc, tool_registry, tool_wait_timeout_seconds)
                cancel_event = threading.Event()
                fut = pool.submit(
                    contextvars.copy_context().run,
                    _exec_single_with_timeout, tc, per_tool_timeout, cancel_event,
                )
                future_cancel[fut] = cancel_event
                futures[fut] = tc
            pending = set(futures)
            for future in as_completed(
                futures,
                timeout=tool_wait_timeout_seconds if tool_wait_timeout_seconds and tool_wait_timeout_seconds > 0 else None,
            ):
                pending.discard(future)
                (tc_item, result_str, success, dur, cached, guard_result), timed_out = future.result()
                if progress_callback:
                    progress_callback(stream_event("tool_done", step=step, tool=tc_item.name, success=success, duration=dur))
                log_entry = {
                    "step": step, "tool": tc_item.name, "arguments": tc_item.arguments,
                    "success": success, "duration": dur, "result_length": len(result_str),
                    "cached": cached,
                }
                if timed_out:
                    log_entry["timeout"] = True
                if guard_result is not None:
                    log_entry.update({
                        "guarded": True,
                        "expected_stock_code": guard_result.get("expected_stock_code"),
                        "requested_stock_code": guard_result.get("requested_stock_code"),
                        "allowed_stock_codes": guard_result.get("allowed_stock_codes", []),
                    })
                tool_calls_log.append(log_entry)
                results.append({"tc": tc_item, "result_str": result_str})
        except FuturesTimeoutError:
            timeout_triggered = True
            timeout_label = (
                f"{tool_wait_timeout_seconds:.2f}s"
                if tool_wait_timeout_seconds is not None
                else "the configured limit"
            )
            logger.warning("Tool batch timed out after %s at step %d", timeout_label, step)
            for future, tc_item in futures.items():
                if future in pending:
                    future.cancel()
                    future_cancel[future].set()
                    result_str = _build_timeout_result_payload(
                        tc_item.name, tc_item.arguments,
                        tool_wait_timeout_seconds, non_retriable_tool_results,
                    )
                    if progress_callback:
                        progress_callback(stream_event(
                            "tool_done",
                            step=step,
                            tool=tc_item.name,
                            success=False,
                            duration=round(tool_wait_timeout_seconds or 0.0, 2),
                        ))
                    tool_calls_log.append({
                        "step": step,
                        "tool": tc_item.name,
                        "arguments": tc_item.arguments,
                        "success": False,
                        "duration": round(tool_wait_timeout_seconds or 0.0, 2),
                        "result_length": len(result_str),
                        "cached": False,
                        "timeout": True,
                    })
                    results.append({"tc": tc_item, "result_str": result_str})
        finally:
            pool.shutdown(wait=not timeout_triggered, cancel_futures=timeout_triggered)

    return results
