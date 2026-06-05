"""Operational orchestrator code (Stage 10f).

Runtime ops that the orchestrator process itself runs — currently the
on-startup disaster-recovery reconciliation (BRD §16). Lives UNDER the
``orchestrator`` package (not the top-level ``ops/`` of BRD §9, which holds
standalone shell scripts like ``backup.sh`` / ``restore.sh``) because this is
Python imported by the FastAPI lifespan and must be covered by the BLOCKING
mypy gate (``mypy orchestrator tests``).
"""
