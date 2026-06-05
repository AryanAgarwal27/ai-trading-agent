#!/usr/bin/env bash
#
# ops/backup.sh — pg_dump the three logical DBs and rclone them off-box.
# Stage 10h, BRD §16 (disaster recovery), SPEC §3.1 (backup target = Google
# Drive via rclone, NOT the BRD B2/S3 example — operator deviation, LOCKED).
#
# WHAT IT DOES
#   For each of the three logical DBs (app, langgraph_checkpoints,
#   langgraph_store — BRD §5.8) it runs `pg_dump --format=custom` into a
#   timestamped file under the backup dir, then (unless --no-upload) `rclone
#   copy`s each file to gdrive:ai-trading-agent-backups/.
#
# ONE DUMP PER DB (not one combined dumpall) — deliberate:
#   * They are independent logical DBs with different managers: `app` is ours
#     (Alembic migrations), `langgraph_checkpoints` is owned by PostgresSaver
#     .setup(), `langgraph_store` by AsyncPostgresStore.setup() (+ pgvector).
#   * `--format=custom` is a per-database format (pg_dump, not pg_dumpall), so
#     per-DB files are the natural unit AND let restore.sh selectively restore
#     ONE DB to a scratch target without touching the others.
#   * pg_dumpall would also dump cluster-wide roles/tablespaces we don't want to
#     replay on restore. Per-DB custom dumps keep restore surgical.
#
# CONNECTION comes from the ENVIRONMENT (never hardcoded). The script sources
# repo-root .env if present (so the operator need not export), then reads:
#   POSTGRES_HOST / POSTGRES_PORT / POSTGRES_USER / POSTGRES_PASSWORD
# Override the DB list with AIT_BACKUP_DBS, the dir with --backup-dir /
# AIT_BACKUP_DIR, the remote with BACKUP_RCLONE_REMOTE.
#
# TWO EXECUTION MODES (so it runs on the operator's Windows box AND the VPS):
#   * Host pg client tools (default): uses `pg_dump` from PATH against
#     POSTGRES_HOST:POSTGRES_PORT. This is the Linux VPS deploy path.
#   * Via the Postgres container: set AIT_PG_CONTAINER=ait-postgres and the
#     script runs pg_dump INSIDE that container (connecting to the container's
#     own server on 127.0.0.1:5432), streaming the dump to the host file over
#     `docker exec`. This is the operator's Windows/Git-Bash path — the host has
#     no pg_dump installed, but the running postgres:15 container ships it.
#
# FAILS LOUDLY: set -euo pipefail; a pg_dump or rclone error aborts non-zero. A
# partial dump is written to <file>.partial and only renamed to <file> on
# success, so a crashed dump never leaves a valid-looking backup.
#
# See ops/DR.md for the rclone remote setup and the Stage 10 DoD commands.

set -euo pipefail

# ── locate repo root + load .env ───────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${AIT_ENV_FILE:-$REPO_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; . "$ENV_FILE"; set +a
fi

# ── defaults (env wins; .env.example documents these) ──────────────────
POSTGRES_HOST="${POSTGRES_HOST:-127.0.0.1}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-trading_agent}"
AIT_BACKUP_DBS="${AIT_BACKUP_DBS:-app langgraph_checkpoints langgraph_store}"
BACKUP_DIR="${AIT_BACKUP_DIR:-$REPO_ROOT/backups}"
BACKUP_RCLONE_REMOTE="${BACKUP_RCLONE_REMOTE:-gdrive:ai-trading-agent-backups}"
AIT_PG_CONTAINER="${AIT_PG_CONTAINER:-}"
AIT_PG_CONTAINER_PORT="${AIT_PG_CONTAINER_PORT:-5432}"
DO_UPLOAD=1

usage() {
  cat <<'EOF'
Usage: ops/backup.sh [--no-upload] [--backup-dir DIR] [-h|--help]

  --no-upload      pg_dump only; skip the rclone copy to gdrive. Use this for the
                   local DoD restore test (the operator's box has no rclone).
  --backup-dir DIR write dumps here (default: <repo>/backups, gitignored).

Env: POSTGRES_HOST/PORT/USER/PASSWORD, AIT_BACKUP_DBS, BACKUP_RCLONE_REMOTE,
     AIT_PG_CONTAINER (run pg_dump inside that container instead of host PATH).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-upload) DO_UPLOAD=0; shift ;;
    --backup-dir) BACKUP_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "backup.sh: unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${POSTGRES_PASSWORD:-}" ]]; then
  echo "backup.sh: POSTGRES_PASSWORD is not set (export it or put it in $ENV_FILE)" >&2
  exit 1
fi

# ── pg_dump invocation: host PATH vs inside the postgres container ─────
if [[ -n "$AIT_PG_CONTAINER" ]]; then
  CONN_HOST="127.0.0.1"; CONN_PORT="$AIT_PG_CONTAINER_PORT"
  echo "backup.sh: using pg_dump inside container '$AIT_PG_CONTAINER' (host:$CONN_HOST port:$CONN_PORT)"
  pg_dump_cmd() {
    docker exec -i -e PGPASSWORD="$POSTGRES_PASSWORD" "$AIT_PG_CONTAINER" \
      pg_dump "$@"
  }
else
  CONN_HOST="$POSTGRES_HOST"; CONN_PORT="$POSTGRES_PORT"
  echo "backup.sh: using host pg_dump (host:$CONN_HOST port:$CONN_PORT)"
  pg_dump_cmd() {
    PGPASSWORD="$POSTGRES_PASSWORD" pg_dump "$@"
  }
fi

mkdir -p "$BACKUP_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
echo "backup.sh: timestamp=$TS dir=$BACKUP_DIR dbs=[$AIT_BACKUP_DBS]"

declare -a PRODUCED=()
for db in $AIT_BACKUP_DBS; do
  out="$BACKUP_DIR/${db}_${TS}.dump"
  tmp="$out.partial"
  echo "backup.sh: dumping '$db' -> $out"
  # --format=custom (-Fc): compressed, selective-restore-capable, the per-DB
  # format restore.sh expects. Stream to a .partial then atomically rename.
  pg_dump_cmd -Fc -h "$CONN_HOST" -p "$CONN_PORT" -U "$POSTGRES_USER" -d "$db" > "$tmp"
  if [[ ! -s "$tmp" ]]; then
    echo "backup.sh: ERROR empty dump for '$db' ($tmp)" >&2
    rm -f "$tmp"
    exit 1
  fi
  mv "$tmp" "$out"
  size="$(wc -c < "$out" | tr -d ' ')"
  echo "backup.sh: wrote $out (${size} bytes)"
  PRODUCED+=("$out")
done

if [[ "$DO_UPLOAD" -eq 1 ]]; then
  if ! command -v rclone >/dev/null 2>&1; then
    echo "backup.sh: ERROR rclone not found but upload requested. Install rclone, or" >&2
    echo "           re-run with --no-upload for a local-only dump (DoD test)." >&2
    exit 1
  fi
  RCLONE_FLAGS=()
  [[ -n "${BACKUP_RCLONE_CONFIG:-}" ]] && RCLONE_FLAGS+=(--config "$BACKUP_RCLONE_CONFIG")
  for f in "${PRODUCED[@]}"; do
    echo "backup.sh: rclone copy $f -> $BACKUP_RCLONE_REMOTE/"
    rclone "${RCLONE_FLAGS[@]}" copy "$f" "$BACKUP_RCLONE_REMOTE/"
  done
  echo "backup.sh: uploaded ${#PRODUCED[@]} dump(s) to $BACKUP_RCLONE_REMOTE/"
else
  echo "backup.sh: --no-upload set; skipped rclone copy (${#PRODUCED[@]} local dump(s) kept)"
fi

echo "backup.sh: DONE (${#PRODUCED[@]} dump(s) at $TS)"
