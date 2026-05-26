"""Stage 0 smoke test — keeps CI honest before real tests land.

Asserts: (a) the orchestrator package imports cleanly, (b) pyproject.toml exists
at repo root. Without this, `pytest` exits 5 (no tests collected) and CI looks
green for the wrong reason.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_orchestrator_package_imports() -> None:
    import orchestrator

    assert orchestrator.__version__


def test_pyproject_exists_at_repo_root() -> None:
    assert (REPO_ROOT / "pyproject.toml").is_file()


def test_brd_and_spec_present() -> None:
    """BRD.md and SPEC.md are load-bearing; both must be present at repo root."""
    assert (REPO_ROOT / "BRD.md").is_file()
    assert (REPO_ROOT / "SPEC.md").is_file()
