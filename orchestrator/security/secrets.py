"""Secret-loading boundary for exchange credentials (BRD §1.1 rule 5, §15).

Single point where exchange API credentials enter the orchestrator. The
non-negotiable this module enforces is BRD §1.1 rule 5: paper and live MUST
use SEPARATE keys (the live subaccount has withdrawals disabled at the
exchange). :func:`load_live_credentials` loads the live creds and REFUSES to
return them if they collide value-for-value with the paper creds — that
collision is the copy-paste mistake ("BINANCE_LIVE_API_KEY=${BINANCE_PAPER_API_KEY}")
that silently points a live config at paper keys, or worse.

Adapter seams: reads route through a :class:`SecretProvider`. v1 ships only
:class:`EnvSecretProvider` (``os.environ``). :class:`SopsSecretProvider` and
:class:`OnePasswordSecretProvider` are declared as seams — method signatures
that raise ``NotImplementedError`` — so Stage 11 hardening can wire a real
backend without touching call sites. No real sops/1password wiring in v1
(SPEC Stage 8a decision).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

# ─── Env var names (single source of truth for the boundary) ───────────
LIVE_API_KEY_VAR = "BINANCE_LIVE_API_KEY"
LIVE_API_SECRET_VAR = "BINANCE_LIVE_API_SECRET"
LIVE_API_PASSWORD_VAR = "BINANCE_LIVE_API_PASSWORD"

PAPER_API_KEY_VAR = "BINANCE_PAPER_API_KEY"
PAPER_API_SECRET_VAR = "BINANCE_PAPER_API_SECRET"
PAPER_API_PASSWORD_VAR = "PAPER_API_PASSWORD"


# ─── Exceptions ────────────────────────────────────────────────────────


class MissingSecretError(RuntimeError):
    """A required secret is absent. Message names the missing env var."""


class SecretCollisionError(RuntimeError):
    """Live credentials collide with paper credentials (BRD §1.1 rule 5)."""


# ─── Credential value object ───────────────────────────────────────────


@dataclass(frozen=True)
class ExchangeCredentials:
    """Resolved exchange creds: API key/secret + the Freqtrade REST password."""

    key: str
    secret: str
    api_password: str


# ─── Provider seams ────────────────────────────────────────────────────


class SecretProvider(Protocol):
    """Reads a secret by name, returning ``None`` when absent."""

    def get(self, name: str) -> str | None: ...


class EnvSecretProvider:
    """The v1 provider: reads ``os.environ`` at call time."""

    def get(self, name: str) -> str | None:
        return os.environ.get(name)


class SopsSecretProvider:
    """Seam for a future sops-encrypted backend (SPEC Stage 8a: signature only)."""

    def get(self, name: str) -> str | None:
        raise NotImplementedError(
            "SopsSecretProvider is a Stage 11 seam; v1 uses EnvSecretProvider"
        )


class OnePasswordSecretProvider:
    """Seam for a future 1Password backend (SPEC Stage 8a: signature only)."""

    def get(self, name: str) -> str | None:
        raise NotImplementedError(
            "OnePasswordSecretProvider is a Stage 11 seam; v1 uses EnvSecretProvider"
        )


# ─── Loading ───────────────────────────────────────────────────────────


def _require(provider: SecretProvider, name: str) -> str:
    """Return a required secret or raise :class:`MissingSecretError` naming it."""
    value = provider.get(name)
    if not value:
        raise MissingSecretError(
            f"required secret {name} is not set; see .env.example "
            "(Stage 8a live-key block) and BRD §1.1 rule 5"
        )
    return value


def load_paper_credentials(provider: SecretProvider | None = None) -> ExchangeCredentials:
    """Load the paper-subaccount credentials (BRD §1.1 rule 5)."""
    provider = provider or EnvSecretProvider()
    return ExchangeCredentials(
        key=_require(provider, PAPER_API_KEY_VAR),
        secret=_require(provider, PAPER_API_SECRET_VAR),
        api_password=_require(provider, PAPER_API_PASSWORD_VAR),
    )


def load_live_credentials(provider: SecretProvider | None = None) -> ExchangeCredentials:
    """Load the live-subaccount credentials, enforcing paper/live separation.

    Raises :class:`MissingSecretError` (naming the var) if a required live
    secret is absent, and :class:`SecretCollisionError` if the live key or
    secret value equals the corresponding paper value — the BRD §1.1 rule 5
    catastrophe. The collision check is skipped for any paper field that is
    not configured (a live-only deployment has nothing to collide with).
    """
    provider = provider or EnvSecretProvider()
    creds = ExchangeCredentials(
        key=_require(provider, LIVE_API_KEY_VAR),
        secret=_require(provider, LIVE_API_SECRET_VAR),
        api_password=_require(provider, LIVE_API_PASSWORD_VAR),
    )

    paper_key = provider.get(PAPER_API_KEY_VAR)
    if paper_key and paper_key == creds.key:
        raise SecretCollisionError(
            f"{LIVE_API_KEY_VAR} has the same value as {PAPER_API_KEY_VAR}; "
            "live and paper MUST use separate subaccount keys (BRD §1.1 rule 5)"
        )
    paper_secret = provider.get(PAPER_API_SECRET_VAR)
    if paper_secret and paper_secret == creds.secret:
        raise SecretCollisionError(
            f"{LIVE_API_SECRET_VAR} has the same value as {PAPER_API_SECRET_VAR}; "
            "live and paper MUST use separate subaccount keys (BRD §1.1 rule 5)"
        )
    return creds
