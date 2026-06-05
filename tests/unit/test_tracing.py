"""Unit tests for :mod:`orchestrator.observability.tracing` (Stage 10d, BRD §14).

Stage 10d sends LangGraph/LangChain traces to LangSmith (SPEC §1 Q4),
tagged so a trace is findable by ``strategy_id`` + ``stage`` and shares
ONE id with the Stage 10c structlog lines. The contract pinned here:

1. ``trace_config`` mirrors the 10c nested ``run_id`` as the LangSmith
   ROOT run id (``config["run_id"] == uuid.UUID(run_id)``) — no second
   id is minted.
2. The same ``run_id`` (hex form, matching structlog) plus
   ``strategy_id`` / ``thread_id`` / ``stage`` land in ``metadata`` and
   ``tags`` so the trace is searchable.
3. ``trace_config`` is PURE — it returns a copy, never mutates the input,
   and preserves ``configurable`` (the checkpointer's thread_id).
4. ``stage`` is optional (omitted cleanly when None).
5. ``tracing_enabled`` / ``langsmith_project`` read the env (the actual
   on/off decision is langchain's, driven by ``LANGSMITH_TRACING``).
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from orchestrator.observability import tracing


def _run_id() -> str:
    return uuid.uuid4().hex


def test_trace_config_mirrors_run_id_as_the_langsmith_root_run_id() -> None:
    rid = _run_id()
    cfg = tracing.trace_config(
        {"configurable": {"thread_id": "strategy_abc"}},
        strategy_id="strategy_abc",
        thread_id="strategy_abc",
        run_id=rid,
        stage="research",
    )
    # The LangSmith root run id IS the 10c run_id (parsed back to its UUID) —
    # one shared id, no second id minted.
    assert cfg["run_id"] == uuid.UUID(rid)
    assert cfg["run_id"].hex == rid


def test_trace_config_metadata_carries_strategy_stage_run_id() -> None:
    rid = _run_id()
    cfg = tracing.trace_config(
        {"configurable": {"thread_id": "tid"}},
        strategy_id="sid",
        thread_id="tid",
        run_id=rid,
        stage="paper",
    )
    md = cfg["metadata"]
    assert md["strategy_id"] == "sid"
    assert md["thread_id"] == "tid"
    assert md["stage"] == "paper"
    # run_id in metadata is the HEX form — byte-identical to the structlog
    # line's run_id, so a LangSmith trace and its logs cross-reference exactly.
    assert md["run_id"] == rid


def test_trace_config_tags_make_trace_findable_by_strategy_and_stage() -> None:
    rid = _run_id()
    cfg = tracing.trace_config(
        {"configurable": {"thread_id": "tid"}},
        strategy_id="sid",
        thread_id="tid",
        run_id=rid,
        stage="live",
    )
    tags = cfg["tags"]
    assert "strategy_id:sid" in tags
    assert "stage:live" in tags
    assert f"run_id:{rid}" in tags


def test_trace_config_preserves_configurable_thread_id() -> None:
    cfg = tracing.trace_config(
        {"configurable": {"thread_id": "strategy_xyz"}},
        strategy_id="strategy_xyz",
        thread_id="strategy_xyz",
        run_id=_run_id(),
    )
    # The checkpointer keys off configurable.thread_id — it must survive intact.
    assert cfg["configurable"]["thread_id"] == "strategy_xyz"


def test_trace_config_does_not_mutate_the_input() -> None:
    original: dict[str, Any] = {"configurable": {"thread_id": "tid"}}
    tracing.trace_config(
        original, strategy_id="sid", thread_id="tid", run_id=_run_id(), stage="research"
    )
    # Pure: the caller's config is untouched (no run_id / metadata / tags added).
    assert original == {"configurable": {"thread_id": "tid"}}


def test_trace_config_stage_optional() -> None:
    cfg = tracing.trace_config(
        {"configurable": {"thread_id": "tid"}},
        strategy_id="sid",
        thread_id="tid",
        run_id=_run_id(),
    )
    assert "stage" not in cfg["metadata"]
    assert not any(t.startswith("stage:") for t in cfg["tags"])


def test_trace_config_merges_with_preexisting_metadata_and_tags() -> None:
    rid = _run_id()
    cfg = tracing.trace_config(
        {"configurable": {"thread_id": "tid"}, "metadata": {"k": "v"}, "tags": ["pre"]},
        strategy_id="sid",
        thread_id="tid",
        run_id=rid,
        stage="research",
    )
    assert cfg["metadata"]["k"] == "v"  # preexisting metadata preserved
    assert "pre" in cfg["tags"]  # preexisting tags preserved
    assert cfg["metadata"]["strategy_id"] == "sid"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("TRUE", True),
        ("false", False),
        ("0", False),
        ("", False),
        ("no", False),
    ],
)
def test_tracing_enabled_reads_env(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", value)
    assert tracing.tracing_enabled() is expected


def test_tracing_enabled_false_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert tracing.tracing_enabled() is False


def test_langsmith_project_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    assert tracing.langsmith_project() == "ai-trading-agent"  # SPEC §1 Q4 default
    monkeypatch.setenv("LANGSMITH_PROJECT", "custom-project")
    assert tracing.langsmith_project() == "custom-project"
