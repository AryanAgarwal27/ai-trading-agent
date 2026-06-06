"""Stage 11c security smoke tests (BRD §15) — durable regression guards.

Pins the mechanically-checkable items from the §15 hardening checklist so a
regression fails CI rather than silently reopening a hole. See ``ops/SECURITY.md``
for the full audit (VERIFIED-OK / FIXED / DEFERRED per item, with evidence).

Marked ``integration`` so it runs in the CI integration job (these are pure —
no services needed — but the suite-level conftest setup applies, including the
Stage 11c F1 strict-msgpack forcing the msgpack guard relies on).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]  # PyYAML ships no py.typed / stubs

import orchestrator.agents.generator as generator_mod
from orchestrator.security.ast_validator import (
    ASTValidationError,
    validate_strategy_source,
)
from orchestrator.security.secrets import (
    LIVE_API_KEY_VAR,
    LIVE_API_PASSWORD_VAR,
    LIVE_API_SECRET_VAR,
    PAPER_API_KEY_VAR,
    PAPER_API_SECRET_VAR,
    SecretCollisionError,
    load_live_credentials,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE_FILES = ("docker-compose.yml", "docker-compose.freqtrade.yml")


def _published_port_bindings(compose_path: Path) -> list[str]:
    """Every host-published port mapping across all services in a compose file
    (the ``ports:`` list entries — the real host-exposure boundary)."""
    data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    bindings: list[str] = []
    for svc in (data.get("services") or {}).values():
        for entry in svc.get("ports") or []:
            bindings.append(str(entry))
    return bindings


# ── §15: all ports bound to 127.0.0.1 ──────────────────────────────────


def test_no_compose_port_publishes_on_non_loopback() -> None:
    """Every PUBLISHED port in every compose file binds 127.0.0.1 — never a
    public / all-interfaces address (BRD §15).

    Note: container-internal listen addresses (redis ``--bind 0.0.0.0``,
    Freqtrade ``listen_ip_address: 0.0.0.0``) are NOT host-published and are out
    of scope — the host-exposure boundary is the ``ports:`` mapping, which is what
    this asserts. See ops/SECURITY.md item 2 for why those are correct.
    """
    found = 0
    for name in _COMPOSE_FILES:
        for binding in _published_port_bindings(_REPO_ROOT / name):
            found += 1
            assert binding.startswith(
                "127.0.0.1:"
            ), f"{name}: published port {binding!r} is not bound to 127.0.0.1 (BRD §15)"
    assert found >= 4, f"expected >=4 published ports across compose files, found {found}"


def test_postgres_uses_scram_auth() -> None:
    """Postgres enforces SCRAM-SHA-256 (BRD §15 item 6)."""
    data = yaml.safe_load((_REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    pg_env = data["services"]["postgres"]["environment"]
    assert pg_env["POSTGRES_HOST_AUTH_METHOD"] == "scram-sha-256"


# ── §15: AST validator rejects disallowed imports ──────────────────────


@pytest.mark.parametrize("snippet", ["import os\n", "import subprocess\n", "import importlib\n"])
def test_ast_validator_rejects_disallowed_import(snippet: str) -> None:
    """The AST validator rejects forbidden imports — including ``importlib``, the
    obvious ``__import__`` bypass (BRD §15 item 7)."""
    with pytest.raises(ASTValidationError):
        validate_strategy_source(snippet)


@pytest.mark.parametrize("snippet", ["eval('1')\n", "exec('x=1')\n", "compile('1','<s>','eval')\n"])
def test_ast_validator_rejects_forbidden_calls(snippet: str) -> None:
    with pytest.raises(ASTValidationError):
        validate_strategy_source(snippet)


def test_ast_validator_allows_an_allowlisted_import() -> None:
    """Counterexample (no false-positive): an allowlisted import passes."""
    validate_strategy_source("import pandas\nimport numpy\n")


# ── §15: generator uses structured output, no free-form code path ───────


def test_generator_uses_structured_output_only() -> None:
    """The generator's LLM call extracts params via
    ``with_structured_output(Schema)`` (BRD §15 item 8) — the model returns a
    schema-bounded object, never free-form code. The two structural guarantees:
    (1) the extractor uses ``with_structured_output``; (2) the generator node
    AST-validates its rendered output via ``validate_strategy_source`` before the
    strategy can run. Pinned by source so a refactor that drops either trips
    here."""
    extractor_src = inspect.getsource(generator_mod._default_params_extractor)
    assert (
        "with_structured_output" in extractor_src
    ), "generator's param extractor must use with_structured_output (no free-form output)"

    module_src = inspect.getsource(generator_mod)
    assert (
        "validate_strategy_source(" in module_src
    ), "generator must AST-validate rendered output (the disallowed-import gate)"


# ── §15: paper vs live exchange-key separation (BRD §1.1 rule 5) ────────


class _DictProvider:
    """A SecretProvider stub backed by an in-test dict (no os.environ touch)."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    def get(self, name: str) -> str | None:
        return self._values.get(name)


def test_live_credentials_reject_paper_key_collision() -> None:
    """``load_live_credentials`` refuses live keys equal to paper keys — the
    copy-paste catastrophe that points a live config at paper creds (BRD §1.1
    rule 5)."""
    shared = "AAAA-identical-key"
    provider = _DictProvider(
        {
            LIVE_API_KEY_VAR: shared,
            LIVE_API_SECRET_VAR: "live-secret",
            LIVE_API_PASSWORD_VAR: "rest-pw",
            PAPER_API_KEY_VAR: shared,  # collision
            PAPER_API_SECRET_VAR: "paper-secret",
        }
    )
    with pytest.raises(SecretCollisionError):
        load_live_credentials(provider)


def test_live_credentials_accept_distinct_keys() -> None:
    provider = _DictProvider(
        {
            LIVE_API_KEY_VAR: "live-key",
            LIVE_API_SECRET_VAR: "live-secret",
            LIVE_API_PASSWORD_VAR: "rest-pw",
            PAPER_API_KEY_VAR: "paper-key",
            PAPER_API_SECRET_VAR: "paper-secret",
        }
    )
    creds = load_live_credentials(provider)
    assert creds.key == "live-key"


# ── §15: .gitignore excludes secrets / local data ──────────────────────


def test_gitignore_excludes_secrets_and_local_data() -> None:
    gitignore = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for needed in (".env", "secrets/", "freqtrade/user_data/data/"):
        assert needed in gitignore, f".gitignore must exclude {needed} (BRD §15 item 9)"


# ── §15 item 1/10: strict-msgpack checkpoint deserialization (Stage 11c F1) ──


def test_strict_msgpack_is_in_effect() -> None:
    """The §15/§6.6 checkpoint-RCE control is actually ON under the suite —
    conftest forces ``LANGGRAPH_STRICT_MSGPACK`` BEFORE the langgraph import, and
    ``orchestrator/__init__.py`` load_dotenv keeps the production path honest.
    Asserts LangGraph's RESOLVED flag (not the env var): the F1 hole was the env
    being set too late to be captured at langgraph import, so strict mode read
    silently OFF while the env var still said "true"."""
    from langgraph.checkpoint.serde._msgpack import STRICT_MSGPACK_ENABLED

    assert STRICT_MSGPACK_ENABLED is True


def test_startup_guard_refuses_when_strict_msgpack_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lifespan guard ``_assert_strict_msgpack_enabled`` fails LOUDLY when
    strict mode is not in effect, so a misconfigured / load-order-regressed deploy
    crashes at startup instead of running with checkpoint deserialization wide
    open (Stage 11c F1)."""
    import langgraph.checkpoint.serde._msgpack as msgpack_mod

    from orchestrator.main import _assert_strict_msgpack_enabled

    monkeypatch.setattr(msgpack_mod, "STRICT_MSGPACK_ENABLED", False)
    with pytest.raises(RuntimeError, match="LANGGRAPH_STRICT_MSGPACK"):
        _assert_strict_msgpack_enabled()
