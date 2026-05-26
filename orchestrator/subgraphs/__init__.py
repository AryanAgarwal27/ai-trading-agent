"""Per-strategy subgraphs (BRD §5.2).

Each module here owns one of the four lifecycle subgraphs:

- :mod:`orchestrator.subgraphs.validation` — Stage 4. Parallel walk-forward
  backtests + robustness fan-out + risk-analyst HITL hand-off to paper_gate.
- :mod:`orchestrator.subgraphs.research` — Stage 5. Researcher + generator
  + adversarial critic loop + lookahead-bias gate.
- :mod:`orchestrator.subgraphs.paper` — Stage 7. Dry-run spawn + 30-day
  monitor cycle + live_gate HITL hand-off.
- :mod:`orchestrator.subgraphs.live` — Stage 8. Live spawn + multi-agent
  review + out-of-band kill switch integration.

The per-strategy parent graph (``orchestrator.graph``) composes these in
sequence; the supervisor graph (``orchestrator.supervisor``, Stage 9)
spawns per-strategy threads on demand.
"""
