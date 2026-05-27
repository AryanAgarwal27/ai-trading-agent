"""Hard-coded gate thresholds + HITL helpers for the per-strategy lifecycle.

Two responsibilities, kept separate so the threshold module has zero
non-stdlib imports and can be loaded by any other module without circular
risk:

- :mod:`orchestrator.gates.thresholds` — single source of truth for BRD §10
  threshold values. The validation, paper, and live subgraphs all import from
  here. SPEC §2 confirms no v1 overrides; re-tuning lands here (and only
  here) per the BRD's "do not put thresholds anywhere else" rule.
- :mod:`orchestrator.gates.hitl` (Stage 6) — ``interrupt()`` payload builder
  + autoresume test helper. No FastAPI / Streamlit imports here; the
  resume endpoint lands in Stage 6c on top of these primitives.
"""
