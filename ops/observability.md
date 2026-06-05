# Observability stack — Prometheus + Grafana (BRD §14)

This directory holds **version-controlled, importable definitions** for the
metrics dashboards and the Prometheus scrape config. They target the metrics the
orchestrator already exports (Stage 10e) — nothing here needs to change to start
using it; you only stand the two services up.

## v1 scope decision — DEFINITIONS-ONLY (operator decision, Stage 10g)

v1 does **not** run Grafana or Prometheus, and they are **not** added to
`docker-compose.yml`. We ship the dashboard JSON + scrape config as
deploy-time-importable artifacts instead. Stand them up whenever you actually
want the dashboards (a single-host deploy step, BRD §3).

**Why this is within BRD §14 intent (no SPEC §3 deviation entry).** BRD §14
lists Grafana dashboards as a Stage 10 *deliverable* ("optional in v1; required
in Stage 10") and names the panels; it never specifies a running service
topology. Shipping importable definitions that reference the real exported
metrics *is* delivering the dashboards — "required in Stage 10" is satisfied by
the artifacts existing and being correct, not by a Grafana process running in
the dev compose stack. This mirrors how the Stage 10e.2 §14 shape decision
(in-orchestrator collector instead of a per-container sidecar) was handled: BRD
§14 is observability guidance, not a §1.1/§6 non-negotiable, so the decision is
recorded here + in the commit message rather than as a SPEC §3 entry. If a later
stage decides to *run* the stack in compose, that is the change that would earn
a SPEC entry.

## What is exported (the single source of truth)

All metrics register on the prometheus_client default registry and are
serialized by the orchestrator's `GET /metrics` (FastAPI). Defined in
[`orchestrator/observability/metrics.py`](../orchestrator/observability/metrics.py):

| Metric | Type | Labels | Source |
|---|---|---|---|
| `ait_kill_switch_fires_total` | counter | — | kill_subscription (BRD §11, §5.6) |
| `ait_strategies_by_stage` | gauge | `stage` | supervisor portfolio snapshot (BRD §5.7) |
| `ait_supervisor_runs_total` | counter | `trigger` | run_supervisor (cron/event/manual) |
| `ait_freqtrade_up` | gauge | `strategy_id` | freqtrade_exporter (per container) |
| `ait_freqtrade_profit_closed_percent` | gauge | `strategy_id` | freqtrade_exporter `/api/v1/profit` |
| `ait_freqtrade_max_drawdown` | gauge | `strategy_id` | freqtrade_exporter `/api/v1/profit` |
| `ait_freqtrade_open_trades` | gauge | `strategy_id` | freqtrade_exporter `/api/v1/status` |

The per-container Freqtrade gauges are exposed **through** the orchestrator's
`/metrics` (the in-orchestrator collector scrapes each running container on every
scrape and re-exposes the gauges) — so there is a single Prometheus target, not
one per container.

A CI guard ([`tests/unit/test_grafana_dashboards.py`](../tests/unit/test_grafana_dashboards.py))
asserts the dashboard JSON is valid and that every `ait_*` metric a panel queries
exists in `metrics.py`, so renaming a metric without updating the dashboards
fails the build instead of silently orphaning a panel.

## Files

- `prometheus/prometheus.yml` — scrape config: one `orchestrator` job hitting
  `127.0.0.1:8000/metrics` at a 15s interval (rationale inline in the file).
- `grafana/orchestrator-overview.json` — kill-switch fires + rate,
  strategies-by-stage funnel, supervisor runs by trigger, containers-up summary.
- `grafana/strategy-trading.json` — per-strategy P&L, drawdown, open trades,
  container reachability, plus a placeholder note for per-host CPU/mem
  (node-exporter territory, intentionally not wired in v1).

Both dashboards use a `${DS_PROMETHEUS}` datasource input (Grafana's standard
"export for sharing" form), so import prompts you to pick your Prometheus
datasource — no hardcoded datasource UID to edit.

## Standing it up at deploy time

This is the one operational step that is not automated in v1.

### Option A — quick local check (Docker, ephemeral)

```bash
# From repo root. Prometheus reaches the orchestrator on the host via
# host.docker.internal (Linux: add --add-host=host.docker.internal:host-gateway).
docker run --rm -p 9090:9090 \
  -v "$PWD/ops/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  prom/prometheus

docker run --rm -p 3000:3000 grafana/grafana
```

Then, in Grafana (http://localhost:3000, admin/admin):
1. **Connections → Data sources → Add Prometheus**, URL `http://host.docker.internal:9090`.
2. **Dashboards → New → Import**, upload each JSON in `ops/grafana/`, and pick the
   Prometheus datasource when prompted.

> If Prometheus runs in a container while the orchestrator runs on the host, the
> `127.0.0.1:8000` target in `prometheus.yml` resolves to the *container's* own
> loopback, not the host's. Change the target to `host.docker.internal:8000`
> (or the orchestrator's compose service name) for that topology. The committed
> default is `127.0.0.1` because BRD §15 binds every port to loopback and the
> canonical deploy runs Prometheus on the same host/netns as the orchestrator.

### Option B — persistent deploy (provisioning)

For a real deploy, provision instead of clicking:
- Mount `ops/prometheus/prometheus.yml` into the Prometheus container.
- Use Grafana **file provisioning**: a datasource YAML pointing at Prometheus,
  and a dashboards provider that loads `ops/grafana/*.json` from a mounted path.
  (Provisioned dashboards ignore the `${DS_PROMETHEUS}` input — set the
  datasource `uid` in the provisioning datasource file and Grafana wires it.)

## Verifying it works

- `curl -s http://127.0.0.1:8000/metrics | grep ait_` shows the live metric set.
- In Prometheus, **Status → Targets** shows the `orchestrator` job `UP`.
- Each Grafana panel renders; a panel that stays "No data" usually means that
  metric has no series yet (e.g. `ait_freqtrade_*` is empty until a paper/live
  container is running) — not a broken query.
