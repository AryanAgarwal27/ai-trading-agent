"""Observability helpers — Redis pubsub channels + Postgres audit writers.

Stage 6c lands the gate-event publishers and the ``gate_audits`` writer.
Stage 7+ will add the telemetry + kill-switch publishers on the channel
prefixes reserved in :mod:`orchestrator.observability.events`.

Kept distinct from ``orchestrator/tools/`` because these are write-side
plumbing for the dashboard / audit trail, not LLM tools.
"""
