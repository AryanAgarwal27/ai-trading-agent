# DEFERRED — code follow-ups not done in the stage that surfaced them

Running ledger of deliberate deferrals: work consciously postponed (with the
reason and a proposed landing point), so nothing surfaced mid-stage is silently
lost. This file tracks **code-structure** follow-ups; stage-specific
contract/threshold deferrals also live in the `SPEC.md` change log (referenced
here, not duplicated).

| ID | Surfaced | Item | Reason deferred | Proposed landing |
|----|----------|------|-----------------|------------------|
| D-1 | Stage 8c | Backport the `registry_writer_fn` + `secrets_provider` injection seams from `live_spawn` to `paper_spawn`. `paper_spawn` writes the registry inline (`_connect_app_db` / `_upsert_registry_row`) and loads credentials inside `spawn_paper_container`; `live_spawn` (8c) exposes both as seams — cleaner and DB-/credential-stubbable in tests. | Refactoring the working paper path during Stage 8 is out of scope and carries regression risk; the asymmetry is harmless (both paths work). | A dedicated paper-refactor commit, or fold into Stage 11 hardening. |
| D-2 | Stage 8d | `ruff format .` drift across ~36 pre-existing files since Stage 4; CI's `ruff format --check .` would fail. 8d's own files are format-clean. Stage 8 commits are unpushed so CI has not seen 8i–8d. | Mixing 36 files of mechanical formatting into a decision-heavy commit pollutes the audit trail. | A dedicated `chore: ruff format .` housekeeping commit between 8h tag and the Stage 8 push, OR fold into Stage 10's observability/DR hardening pass. |

## Related deferrals tracked elsewhere

- **events.py mypy noise** (redis-untyped) → Stage 10 lockfile work item. See `SPEC.md` change log (2026-05-28).
- **BRD §5.7 / §5.8 `freqtrade_process_id` doc update** (int → str rename) → Stage 9 docs pass. See `SPEC.md` change log (2026-05-27).
