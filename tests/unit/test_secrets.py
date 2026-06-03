"""Stage 8a unit tests — secrets boundary + live config render.

BRD §1.1 rule 5 is the non-negotiable under test: paper and live MUST use
SEPARATE keys. ``orchestrator/security/secrets.py`` is the single point where
exchange credentials enter the orchestrator; it loads live creds from the
environment and REFUSES to hand back live credentials that collide with the
paper credentials (the copy-paste mistake that points "live" at paper keys).

The config-write helper (``render_live_config`` in
``orchestrator.tools.freqtrade_lifecycle``) resolves ``live-base.json`` with
live creds at write-time and must emit ``dry_run: false`` carrying the LIVE
(not paper) key.

All offline — no Docker, no Freqtrade, no real exchange. Env is set per-test
via ``monkeypatch`` so the operator's real ``.env`` (loaded by conftest) can't
leak in.
"""

from __future__ import annotations

import pytest

from orchestrator.gates.thresholds import LIVE_CAPITAL_CAP_USD
from orchestrator.security.secrets import (
    EnvSecretProvider,
    MissingSecretError,
    OnePasswordSecretProvider,
    SecretCollisionError,
    SopsSecretProvider,
    load_live_credentials,
)
from orchestrator.tools.freqtrade_lifecycle import render_live_config

# Live + paper env var names the module reads. Kept local to the test so a
# rename in the module surfaces as a failure here, not a silent skip.
_LIVE_VARS = ("BINANCE_LIVE_API_KEY", "BINANCE_LIVE_API_SECRET", "BINANCE_LIVE_API_PASSWORD")
_PAPER_VARS = ("BINANCE_PAPER_API_KEY", "BINANCE_PAPER_API_SECRET", "PAPER_API_PASSWORD")


def _set_live(monkeypatch: pytest.MonkeyPatch, key: str = "live-key-aaa") -> None:
    monkeypatch.setenv("BINANCE_LIVE_API_KEY", key)
    monkeypatch.setenv("BINANCE_LIVE_API_SECRET", "live-secret-aaa")
    monkeypatch.setenv("BINANCE_LIVE_API_PASSWORD", "live-rest-pw")


def _set_paper(monkeypatch: pytest.MonkeyPatch, key: str = "paper-key-zzz") -> None:
    monkeypatch.setenv("BINANCE_PAPER_API_KEY", key)
    monkeypatch.setenv("BINANCE_PAPER_API_SECRET", "paper-secret-zzz")
    monkeypatch.setenv("PAPER_API_PASSWORD", "paper-rest-pw")


def _clear_all(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*_LIVE_VARS, *_PAPER_VARS):
        monkeypatch.delenv(name, raising=False)


# ─── 1. Live creds load from env ───────────────────────────────────────


def test_load_live_credentials_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_all(monkeypatch)
    _set_live(monkeypatch)
    creds = load_live_credentials()
    assert creds.key == "live-key-aaa"
    assert creds.secret == "live-secret-aaa"
    assert creds.api_password == "live-rest-pw"


def test_load_live_credentials_accepts_injected_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider seam: load_live_credentials routes reads through a
    SecretProvider, so a non-env backend can be injected without env vars."""
    _clear_all(monkeypatch)

    class _DictProvider:
        def __init__(self) -> None:
            self._d = {
                "BINANCE_LIVE_API_KEY": "k-from-provider",
                "BINANCE_LIVE_API_SECRET": "s-from-provider",
                "BINANCE_LIVE_API_PASSWORD": "p-from-provider",
            }

        def get(self, name: str) -> str | None:
            return self._d.get(name)

    creds = load_live_credentials(provider=_DictProvider())
    assert creds.key == "k-from-provider"


# ─── 2. Live creds distinct from paper ─────────────────────────────────


def test_collision_with_paper_key_raises_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BINANCE_LIVE_API_KEY pointed at the same value as BINANCE_PAPER_API_KEY is the
    catastrophic copy-paste — must raise, naming both vars (BRD §1.1 rule 5)."""
    _clear_all(monkeypatch)
    _set_live(monkeypatch, key="same-value")
    _set_paper(monkeypatch, key="same-value")
    with pytest.raises(SecretCollisionError) as exc:
        load_live_credentials()
    msg = str(exc.value)
    assert "BINANCE_LIVE_API_KEY" in msg
    assert "BINANCE_PAPER_API_KEY" in msg


def test_no_collision_when_paper_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live-only deployment (no paper keys configured) must not false-positive
    the collision check — there is nothing to collide with."""
    _clear_all(monkeypatch)
    _set_live(monkeypatch)
    creds = load_live_credentials()  # must not raise
    assert creds.key == "live-key-aaa"


def test_distinct_live_and_paper_values_load_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_all(monkeypatch)
    _set_live(monkeypatch, key="live-distinct")
    _set_paper(monkeypatch, key="paper-distinct")
    creds = load_live_credentials()
    assert creds.key == "live-distinct"


# ─── 3. Missing required live key raises clearly ───────────────────────


def test_missing_live_key_raises_named_exception_and_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_all(monkeypatch)
    monkeypatch.setenv("BINANCE_LIVE_API_SECRET", "s")
    monkeypatch.setenv("BINANCE_LIVE_API_PASSWORD", "p")
    # BINANCE_LIVE_API_KEY deliberately absent.
    with pytest.raises(MissingSecretError) as exc:
        load_live_credentials()
    assert "BINANCE_LIVE_API_KEY" in str(exc.value)


# ─── 4. Config-write helper emits dry_run:false + live key ─────────────


def test_render_live_config_emits_dry_run_false_and_live_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_all(monkeypatch)
    _set_live(monkeypatch, key="live-key-aaa")
    _set_paper(monkeypatch, key="paper-key-zzz")

    cfg = render_live_config(
        strategy_id="strat-1",
        pair_whitelist=["BTC/USDT"],
        stake_amount=125.0,
        strategy_class="GenStrategy_strat_1",
        port=8300,
    )

    assert cfg["dry_run"] is False
    assert cfg["exchange"]["key"] == "live-key-aaa"
    assert cfg["exchange"]["key"] != "paper-key-zzz"
    assert cfg["exchange"]["secret"] == "live-secret-aaa"
    # No leftover ${...} placeholders in the resolved config.
    assert "${" not in __import__("json").dumps(cfg)


def test_render_live_config_caps_stake_at_live_capital_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_all(monkeypatch)
    _set_live(monkeypatch)
    cfg = render_live_config(
        strategy_id="strat-1",
        pair_whitelist=["BTC/USDT"],
        stake_amount=10_000.0,  # absurdly high — must be capped
        strategy_class="GenStrategy_strat_1",
        port=8300,
    )
    assert cfg["stake_amount"] <= LIVE_CAPITAL_CAP_USD


# ─── adapter seams: declared but not implemented in v1 ─────────────────


def test_sops_and_1password_providers_are_seams_only() -> None:
    """v1 ships only the env provider; the sops / 1password adapters exist as
    signatures (SPEC Stage 8a decision) and raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        SopsSecretProvider().get("BINANCE_LIVE_API_KEY")
    with pytest.raises(NotImplementedError):
        OnePasswordSecretProvider().get("BINANCE_LIVE_API_KEY")


def test_env_provider_reads_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_LIVE_API_KEY", "via-env")
    assert EnvSecretProvider().get("BINANCE_LIVE_API_KEY") == "via-env"
    assert EnvSecretProvider().get("DEFINITELY_UNSET_VAR_XYZ") is None
