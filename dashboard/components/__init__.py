"""Dashboard sub-renderers — extracted when a gate's layout diverges
enough from the others that inlining it in :mod:`dashboard.app` hurts
readability.

Stage 6g lands :mod:`dashboard.components.kill_switch_card` for the
kill-switch variant of ``live_pause_review`` (distinct red rendering
per SPEC §4.1). The other gate cards (``paper_gate``, ``live_gate``,
``live_pause_review`` coordinator path) live in ``dashboard/app.py``
because they share the same approve/reject template.
"""
