# Disaster recovery — backups, restore, and the quarterly drill (BRD §16)

The orchestrator's durable state lives in **Postgres** (three logical DBs). This
doc covers how it is backed up off-box, how to restore it, what survives a
Postgres loss, and the **quarterly restore drill** BRD §16 requires. The
companion scripts are [`ops/backup.sh`](backup.sh) and
[`ops/restore.sh`](restore.sh).

## Backup target — Google Drive via rclone (SPEC §3.1, LOCKED)

Per **SPEC §3.1** (operator deviation from BRD §13's B2/S3 example), backups go
to **Google Drive via `rclone`**. The off-box requirement (BRD §3) is satisfied —
Google Drive is off-host. The payload is small (`pg_dump` of three logical DBs),
so rclone's lower gdrive throughput / occasional rate-limit retries are
acceptable.

- Remote: `gdrive:ai-trading-agent-backups/` (operator-created; default in the
  scripts via `BACKUP_RCLONE_REMOTE`).

## What the scripts do

### `ops/backup.sh`

`pg_dump --format=custom` of the three logical DBs — **one dump file per DB**, not
a combined `pg_dumpall`:

| DB | Managed by | Why a separate dump |
|---|---|---|
| `app` | our Alembic migrations | restore/inspect independently |
| `langgraph_checkpoints` | `PostgresSaver.setup()` | thread checkpoints; restore alone to replay |
| `langgraph_store` | `AsyncPostgresStore.setup()` (+ pgvector) | long-term Store; separate lifecycle |

`--format=custom` is a per-database format (pg_dump, not pg_dumpall), so per-DB
files are the natural unit **and** let `restore.sh` restore ONE DB to a scratch
target without touching the others. `pg_dumpall` would also replay cluster-wide
roles/tablespaces we don't want on restore.

Each dump is timestamped (`<db>_<UTC-ISO>.dump`), written to a `.partial` then
atomically renamed on success, then (unless `--no-upload`) `rclone copy`d to the
remote. Connection comes from the environment (it sources repo-root `.env`); it
**fails loudly** (`set -euo pipefail`) on any pg_dump or rclone error.

### `ops/restore.sh`

The inverse, with a safety model built around a **required `--target`**:

- `--target` has no default → you can't run it and clobber prod by accident.
- The three prod DBs are **guarded**: restoring onto one needs `--force-prod`
  **and** typing the DB name to confirm (then it restores in-place with
  `--clean`). Normal use restores to a fresh **scratch** DB.
- Scratch restore `createdb`s a new DB and refuses if it exists (use `--drop` to
  recreate — an explicit destructive opt-in).
- Source is a local file (`--source`) or pulled from gdrive (`--from-gdrive`).

## Execution modes (Windows operator vs Linux VPS)

The scripts assume **bash** (Git Bash / WSL on Windows; the VPS shell on deploy).
`.gitattributes` pins `*.sh` to LF so the shebang works on every checkout.

- **Linux VPS (deploy):** host has `pg_dump`/`pg_restore` + `rclone` on PATH;
  `POSTGRES_HOST`/`PORT` point at the local Postgres. Default mode.
- **Operator's Windows box (the DoD):** the host has **no** `pg_dump`/`rclone`
  installed, but the `ait-postgres` container ships the pg client tools. Set
  `AIT_PG_CONTAINER=ait-postgres` and the scripts run pg_dump/pg_restore/createdb
  **inside** that container, streaming dumps to/from host files over
  `docker exec`. For the local DoD, also pass `--no-upload` (no rclone needed).

> **rclone on PATH (Windows Git Bash).** To run `backup.sh` *with* the gdrive
> upload on Windows, rclone must be on Git Bash's PATH. A `winget`-installed
> rclone lands under
> `~/AppData/Local/Microsoft/WinGet/Packages/...rclone.../` which Git Bash does
> NOT pick up by default — `backup.sh` then fails with "rclone not found".
> Extend PATH for the session before running, e.g.
> `export PATH="$PATH:$HOME/AppData/Local/Microsoft/WinGet/Packages/$(ls ~/AppData/Local/Microsoft/WinGet/Packages | grep -i rclone)"`
> (or add the rclone dir to PATH permanently). The Linux VPS (rclone in
> `/usr/bin`) is unaffected. The `--no-upload` DoD path needs no rclone at all.

### Networking caveat (loopback vs container-internal port)

The dev stack maps the container's `5432` to **host `127.0.0.1:5433`**
(`POSTGRES_HOST_PORT=5433` in `.env`). This matters depending on where the pg
client runs:

- **Container mode** (`AIT_PG_CONTAINER` set): the tool runs *inside* the
  container, so it connects to the container's own server at `127.0.0.1:5432` —
  the scripts force this automatically; `POSTGRES_PORT` is ignored.
- **Host mode** on the operator's box: a host pg client must use the *mapped*
  port `127.0.0.1:5433` (`POSTGRES_PORT=5433`, already in `.env`).
- If Prometheus/other tooling runs in a *sibling* container, `127.0.0.1` is that
  container's own loopback — use `host.docker.internal` (or the compose service
  name) instead. (Same caveat noted in `ops/observability.md`.)

## reconcile.py on restart (Stage 10f)

A Postgres restore brings the LangGraph threads back, but the Freqtrade
**containers** may be gone (host reboot, crashed compose). On startup the
orchestrator's lifespan runs
[`orchestrator/ops/reconcile.py`](../orchestrator/ops/reconcile.py): it scans the
restored `strategy_registry`, pings each `freqtrade_api_url`, and for any
unreachable **LIVE** container writes a `kill_switch_events` row
(`reason="orchestrator_restart_no_freqtrade"`) and routes that thread to
`live_pause` for HITL review (via the 9f kill mechanism). So after a restore, a
live strategy whose container did not come back is parked safely, not left
trading blind. (This is `orchestrator/ops/` — runtime code — distinct from the
top-level `ops/` standalone shell scripts here.)

## What survives a Postgres death (BRD §16)

| State | Recoverable from | How |
|---|---|---|
| LangGraph threads | Postgres backup | restore + replay from last checkpoint |
| Freqtrade trade history | Freqtrade's own SQLite | always intact in the worker volume |
| Open positions | the exchange itself | re-attach via Freqtrade |
| OHLCV cache | feather files on disk | unaffected |
| Long-term Store | Postgres backup | restore |

The corollary: the **Postgres dumps are the only thing that must go off-box** —
trade history, open positions, and OHLCV are reconstructable from the exchange /
Freqtrade volumes / disk. That is why `backup.sh` only dumps the three DBs.

## rclone remote setup (fresh deploy)

So a fresh box can recreate the remote:

```bash
rclone config         # n) new remote → name: gdrive → storage: drive (Google Drive) → OAuth
rclone mkdir gdrive:ai-trading-agent-backups
rclone lsd gdrive:    # sanity: the remote lists
```

If `rclone.conf` is in a non-default location, point the scripts at it with
`BACKUP_RCLONE_CONFIG=/path/to/rclone.conf`. The backup folder default is
`BACKUP_RCLONE_REMOTE=gdrive:ai-trading-agent-backups` (override in `.env`).

Schedule on the VPS (BRD §16 "nightly `pg_dump`"): a cron entry, e.g.
`15 3 * * *  cd /opt/ai-trading-agent && ./ops/backup.sh >> /var/log/ait-backup.log 2>&1`.

> **The nightly schedule is a DEPLOY-TIME step (cron / systemd-timer), intentionally
> NOT in APScheduler.** The orchestrator's APScheduler only fires in-process graph
> work (paper/live wakes, regime, kill-switch poll, supervisor cron) — it does not
> run host-level ops, and `backup.sh` is a standalone shell script outside the
> Python process. So nothing in the app backs the DB up automatically: the operator
> MUST install the cron line (or an equivalent systemd timer) on the VPS at deploy
> time, or there is no nightly backup. This is the one operational step the app
> deliberately does not own.

## Quarterly DR drill (BRD §16 "test restore quarterly")

Run once a quarter; takes ~10 min. Record the date + result in your ops log.

1. **Take a fresh backup** and confirm it lands in gdrive:
   `./ops/backup.sh` → check `rclone lsf gdrive:ai-trading-agent-backups/` shows
   today's three `*.dump` files.
2. **Restore each DB to a scratch target** from the gdrive copy (proves the
   off-box artifact is restorable, not just the local file):
   `./ops/restore.sh --from-gdrive app_<ts>.dump --target app_drill` (repeat for
   `langgraph_checkpoints`, `langgraph_store`).
3. **Verify** schema + row counts match the live DB (see DoD below).
4. **Tear down** the scratch DBs (`dropdb ..._drill`).
5. **Log** the outcome. If any step failed, that is a real finding — fix before
   the next deploy.

A drill failure is the signal the backup pipeline rotted (rotated rclone token,
renamed DB, pg version skew) — exactly what the quarterly cadence exists to
catch before you need it for real.

---

## Stage 10 DoD — operator-runnable restore test

> BRD §13 Stage 10 DoD: *"backup script tested with restore to a scratch DB."*

Run from the **repo root in Git Bash** on the operator's Windows box. Uses the
dockerized pg tools (`AIT_PG_CONTAINER=ait-postgres`) and `--no-upload` so it
needs **neither a host pg_dump nor rclone**. `.env` supplies the credentials.

```bash
# 0. (once per shell) load .env so $POSTGRES_USER / $POSTGRES_PASSWORD are set
set -a; source .env; set +a

# 1. BACK UP — dump all three DBs locally via the container (no gdrive upload)
AIT_PG_CONTAINER=ait-postgres ./ops/backup.sh --no-upload
#    → writes backups/app_<ts>.dump, langgraph_checkpoints_<ts>.dump,
#      langgraph_store_<ts>.dump. Note the timestamp it prints.

# 2. RESTORE the app dump to a fresh SCRATCH db (NOT prod)
TS=<paste-the-timestamp-from-step-1>
AIT_PG_CONTAINER=ait-postgres ./ops/restore.sh \
  --source "backups/app_${TS}.dump" --target app_scratch

# 3. VERIFY the scratch db has the schema + the same data as prod
#    (tables present, and strategy_registry row count matches `app`)
docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" ait-postgres \
  psql -h 127.0.0.1 -p 5432 -U "$POSTGRES_USER" -d app_scratch -c '\dt'

echo "prod app:"; docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" ait-postgres \
  psql -h 127.0.0.1 -p 5432 -U "$POSTGRES_USER" -d app -tAc \
  'SELECT count(*) FROM strategy_registry;'
echo "scratch:";  docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" ait-postgres \
  psql -h 127.0.0.1 -p 5432 -U "$POSTGRES_USER" -d app_scratch -tAc \
  'SELECT count(*) FROM strategy_registry;'
#    → \dt lists strategy_registry, gate_audits, telemetry, kill_switch_events,
#      regime_log; the two counts match. DoD PASS.

# 4. CLEAN UP the scratch db
docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" ait-postgres \
  dropdb -h 127.0.0.1 -p 5432 -U "$POSTGRES_USER" --if-exists app_scratch
```

> A green DoD = the dump→restore round-trip preserves schema and data. If the
> live `app` DB is empty (fresh stack), both counts are `0` and `\dt` still
> proves the schema restored — a valid pass. To exercise data, optionally restore
> `langgraph_checkpoints` the same way after running at least one strategy.
