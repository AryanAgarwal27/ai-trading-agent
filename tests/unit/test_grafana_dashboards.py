"""Guard: Grafana dashboards + Prometheus scrape config stay valid and in sync
with the EXPORTED metric set (Stage 10g, BRD §14).

A dashboard that queries a metric the code does not export is dead on arrival —
it renders "No data" forever and nobody notices until an incident. This guard
makes that a build failure instead:

1. every JSON in ``ops/grafana/`` parses as valid JSON;
2. every ``ait_*`` metric token referenced ANYWHERE in a dashboard (panel
   ``expr`` queries, templating queries, descriptions) exists as a metric NAME
   declared in ``orchestrator/observability/metrics.py`` — so renaming a metric
   without updating the dashboards fails here;
3. conversely, every exported ``ait_*`` metric is referenced by at least one
   dashboard — so shipping a metric with no panel (a silent observability gap)
   also fails here;
4. the Prometheus scrape config is valid YAML with an ``orchestrator`` job
   hitting ``/metrics`` (skipped if PyYAML is unavailable — it is a transitive
   dep, not a hard requirement of the guard).

Source of truth for metric names is the metrics.py AST (the literal strings
passed to ``Counter()`` / ``Gauge()``), NOT a hand-maintained list — so the
guard cannot drift from the code it guards.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GRAFANA_DIR = _REPO_ROOT / "ops" / "grafana"
_PROMETHEUS_YML = _REPO_ROOT / "ops" / "prometheus" / "prometheus.yml"
_METRICS_PY = _REPO_ROOT / "orchestrator" / "observability" / "metrics.py"

# A Prometheus metric name token. Our metrics are all ``ait_<lower_snake>``.
_METRIC_TOKEN = re.compile(r"ait_[a-z_]+")


def _known_metric_names() -> set[str]:
    """Metric names declared in metrics.py — the literal first string arg to each
    ``Counter()`` / ``Gauge()`` call, harvested from the AST (no import side
    effects, no hand-maintained mirror list)."""
    tree = ast.parse(_METRICS_PY.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in {"Counter", "Gauge"} and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                if _METRIC_TOKEN.fullmatch(first.value):
                    names.add(first.value)
    return names


def _iter_strings(obj: Any) -> list[str]:
    """Every string value reachable in a parsed-JSON structure."""
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_iter_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_iter_strings(v))
    return out


def _dashboard_files() -> list[Path]:
    return sorted(_GRAFANA_DIR.glob("*.json"))


def test_grafana_dir_has_dashboards() -> None:
    files = _dashboard_files()
    assert files, f"no dashboard JSON found in {_GRAFANA_DIR}"


def test_dashboards_are_valid_json() -> None:
    for path in _dashboard_files():
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - failure detail
            pytest.fail(f"{path.name} is not valid JSON: {exc}")


def test_dashboard_metric_refs_all_exist() -> None:
    """No panel may reference a metric the code does not export."""
    known = _known_metric_names()
    assert known, "no ait_* metric names parsed from metrics.py"

    for path in _dashboard_files():
        doc = json.loads(path.read_text(encoding="utf-8"))
        refs = {tok for s in _iter_strings(doc) for tok in _METRIC_TOKEN.findall(s)}
        orphaned = refs - known
        assert not orphaned, (
            f"{path.name} references metric(s) not in metrics.py: {sorted(orphaned)}. "
            f"Known: {sorted(known)}"
        )


def test_every_exported_metric_is_on_a_dashboard() -> None:
    """No exported metric may be left without a panel (silent observability gap)."""
    known = _known_metric_names()
    referenced: set[str] = set()
    for path in _dashboard_files():
        doc = json.loads(path.read_text(encoding="utf-8"))
        referenced |= {tok for s in _iter_strings(doc) for tok in _METRIC_TOKEN.findall(s)}

    missing = known - referenced
    assert not missing, f"exported metrics with no dashboard panel: {sorted(missing)}"


def test_prometheus_scrape_config_targets_orchestrator_metrics() -> None:
    yaml = pytest.importorskip("yaml")
    assert _PROMETHEUS_YML.exists(), f"missing {_PROMETHEUS_YML}"
    cfg = yaml.safe_load(_PROMETHEUS_YML.read_text(encoding="utf-8"))

    jobs = {j["job_name"]: j for j in cfg["scrape_configs"]}
    assert "orchestrator" in jobs, f"no 'orchestrator' scrape job: {sorted(jobs)}"
    job = jobs["orchestrator"]
    assert job["metrics_path"] == "/metrics"

    targets = [t for sc in job["static_configs"] for t in sc["targets"]]
    assert any(":8000" in t for t in targets), f"no :8000 orchestrator target: {targets}"
    # BRD §15: the committed default binds to loopback.
    assert any(t.startswith("127.0.0.1") for t in targets), targets
