# Security checklist — audit & enforcement (BRD §15)

Stage 11c security audit. Every BRD §15 item below is marked **VERIFIED-OK**
(true in the code, with where it's enforced), **FIXED** (was wrong, what + the
fix), or **DEFERRED** (why + concrete landing). This file reflects what the
audit actually found — not aspiration.

Mechanically-checkable items are pinned by `tests/integration/test_security_smoke.py`
so a regression fails CI. Operational items (network, exchange-side config, key
rotation) are not code-enforceable and are marked DEFERRED-as-ops with their
landing.

Audit base: `HEAD` at Stage 11c. Two genuine code findings beyond the ops items —
the strict-msgpack silent-off hole (FIXED, item 1/10) and two partial-coverage
gaps tracked as **D-18** (secrets centralization) and **D-19** (per-DB users).

| # | §15 item | State | Where enforced / why |
|---|----------|-------|----------------------|
| 1 | `LANGGRAPH_STRICT_MSGPACK=true` in production | **FIXED** (Stage 11c) | Was silently OFF when set via `.env`: LangGraph reads the flag at **import time** (`langgraph/checkpoint/serde/_msgpack.py`), but `main.py` called `load_dotenv()` *after* importing langgraph (line 88, after the line-51 import) — so langgraph captured `false` before `.env` loaded. Fix: `load_dotenv()` moved to `orchestrator/__init__.py` (runs before any submodule imports langgraph), **plus** a fail-loud startup guard `main._assert_strict_msgpack_enabled()` in the lifespan that reads LangGraph's *resolved* `STRICT_MSGPACK_ENABLED` (not the env var) and refuses to start if it's off. Pinned: `test_strict_msgpack_is_in_effect`, `test_startup_refuses_when_strict_msgpack_off`. Verified the app round-trips real Postgres checkpoints under strict mode (101 integration tests green). |
| 2 | All ports bound to `127.0.0.1` | **VERIFIED-OK** | Every host-PUBLISHED port binds loopback: Postgres `docker-compose.yml:29`, Redis `:64`, Freqtrade paper `docker-compose.freqtrade.yml:45`, Freqtrade live `:107`, uvicorn `main.py:1050` (`ORCHESTRATOR_HOST` default `127.0.0.1`). The `0.0.0.0` strings (redis `--bind 0.0.0.0` `docker-compose.yml:55`; Freqtrade `listen_ip_address:0.0.0.0` in `paper-base.json`/`live-base.json`) are **container-internal** listen addresses — required so the host reaches them via the loopback-published port and siblings reach them over the docker bridge; they are NOT host-exposed. Pinned: `test_no_compose_port_publishes_on_non_loopback`. |
| 3 | Remote access via WireGuard / SSH tunnel only | **DEFERRED — ops** | Network/deployment control, not enforceable in the repo. Landing: deploy runbook + host firewall; the loopback-only binds (item 2) are the code-side half. |
| 4 | Exchange keys: live subaccount, withdrawals disabled, IP allowlist, **separate paper/live keys** | **VERIFIED-OK** (separation) / **DEFERRED — ops** (exchange-side) | Separate paper/live keys: `secrets.load_live_credentials` raises `SecretCollisionError` if a live key/secret equals the paper one (BRD §1.1 rule 5); pinned `test_live_credentials_reject_paper_key_collision` / `_accept_distinct_keys`. Subaccount / withdrawals-disabled / IP-allowlist are set **at the exchange**, not in code — DEFERRED-as-ops (landing: exchange-account runbook). |
| 5 | Secrets via `secrets.py`; never in committed `config.json` | **VERIFIED-OK** (no committed secrets) / **DEFERRED — D-18** (centralization) | No hardcoded/committed secrets: tracked configs (`paper-base.json`, `live-base.json`) use `${VAR}` placeholders, resolved at write time; `git grep` for secret literals in tracked files is clean. Live exchange creds route through `secrets.py`. **Gap (D-18):** `OPERATOR_TOKEN`, `DATABASE_URL`, the Freqtrade REST passwords, and the paper exchange keys read raw `os.environ` rather than via `secrets.py` — not exploitable (all from env; v1 `EnvSecretProvider` == `os.environ.get`), but the "via secrets.py" half is partial. Landing in D-18 (fold into D-1, or when a real sops/1Password backend is wired). |
| 6 | Postgres SCRAM-SHA-256; separate DB user per logical DB | **VERIFIED-OK** (SCRAM) / **DEFERRED — D-19** (per-DB users) | SCRAM enforced: `docker-compose.yml:24-25` (`POSTGRES_HOST_AUTH_METHOD: scram-sha-256` + `--auth-host/--auth-local=scram-sha-256`); pinned `test_postgres_uses_scram_auth`. **Gap (D-19):** the three logical DBs (`app`/`langgraph_checkpoints`/`langgraph_store`, `db/init/01_create_databases.sql`) share the single `trading_agent` role — no per-DB least-privilege user. Low risk (single host, loopback, one trust domain); landing in D-19 (dedicated infra commit). |
| 7 | AST validator rejects disallowed imports in every generated strategy | **VERIFIED-OK** | `orchestrator/security/ast_validator.py` — allowlist (`ALLOWED_TOP_LEVEL_IMPORTS`), bans `importlib` (the `__import__` bypass) + `eval`/`exec`/`compile`/`__import__`/`open`/`input`. Run on **every** generated strategy: `generator.py:362` calls `validate_strategy_source` on the rendered source and archives on violation — no bypass path. Pinned: `test_ast_validator_rejects_disallowed_import`, `_rejects_forbidden_calls`, `_allows_an_allowlisted_import`. |
| 8 | Generator uses `with_structured_output(Schema)`; no free-form code path | **VERIFIED-OK** | `generator.py:197` — `model.with_structured_output(schema_cls)`; the model returns a schema-bounded params object, and the strategy file is produced by deterministic SLOT substitution (`render_template`) + AST validation, never free-form code emission. Pinned: `test_generator_uses_structured_output_only`. |
| 9 | `.env`, `secrets/`, `freqtrade/user_data/data/` in `.gitignore` | **VERIFIED-OK** | `.gitignore:2-16` — `.env`, `.env.*` (with `!.env.example`), `secrets/`, `freqtrade/user_data/data/`, `_workers/`, `_live_workers/` (the last holds resolved live configs with secrets). Pinned: `test_gitignore_excludes_secrets_and_local_data`. |
| 10 | LangGraph checkpoint serializer in strict mode | **FIXED** (Stage 11c) | Same control as item 1 — strict msgpack IS the checkpoint serializer's safe-deserialization mode. See item 1 for the silent-off hole + fix + guard. |
| 11 | Quarterly key-rotation reminder | **DEFERRED — ops** | Calendar/ops reminder, not code. Landing: ops runbook + a recurring reminder; rotation itself is exchange + `.env` operator action. |

## Audit notes

- **Only one silently-off control.** Item 1/10 (`LANGGRAPH_STRICT_MSGPACK`) was the
  *only* §15 control with an import-time silent-off failure mode, because it is the
  only one read at module-import time. The others are structural (ports, SCRAM, AST
  validator, structured output, gitignore — cannot be toggled by env timing) or read
  at call time inside functions (`secrets.py`, `LANGSMITH_TRACING`).
- **No exploitable holes beyond F1.** No public/`0.0.0.0` host bind; no hardcoded or
  committed secret; no AST-validator bypass; paper≠live key separation holds.
- **Deferrals are tracked, not buried.** D-18 (secrets centralization) and D-19
  (per-DB users) are real §15 partial-coverage gaps with low v1 risk and concrete
  landings in `DEFERRED.md` — they are marked DEFERRED here, *not* VERIFIED-OK, so
  this checklist tells the truth about what is actually enforced today.
