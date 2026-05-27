"""Per-strategy ReAct agents (BRD §5.3, §5.4, §5.5, §5.6).

One module per agent role. Each module exposes:

  - a real ``create_agent``-backed implementation, lazy-constructed so
    importing the module doesn't fail when ``ANTHROPIC_API_KEY`` is
    unset (e.g. CI without secrets, the operator's first ``python -c``
    inspection),
  - a ``<role>_node(state)`` function the subgraph builder wires in,
    which calls the agent and returns ``Command(goto, update)``,
  - a stub-friendly factory so unit/integration tests can inject a
    deterministic replacement and avoid burning LLM tokens on every
    pytest run.

Stage 4e ships :mod:`orchestrator.agents.risk_analyst` (Opus 4.7,
final automated check before HITL ``paper_gate``).

Stage 5 will add :mod:`researcher`, :mod:`critic`, and the generator
(latter is deterministic, not an agent).

Stage 7 will add :mod:`monitors` (paper / live).

Stage 8 will add :mod:`coordinator` (live-monitoring multi-vote merge).
"""
