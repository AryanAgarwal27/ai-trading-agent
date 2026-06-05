#!/usr/bin/env bash
#
# ops/restore.sh — restore a pg_dump (custom format) into a TARGET database.
# Stage 10h, BRD §16. The inverse of ops/backup.sh; the BRD §13 Stage 10 DoD is
# "backup script tested with restore to a scratch DB" — this script makes that
# safe and repeatable.
#
# SAFETY MODEL (the whole point of taking --target as an arg):
#   * --target is REQUIRED. There is no default, so you can never "just run it"
#     and clobber prod.
#   * The three prod DBs (app, langgraph_checkpoints, langgraph_store) are
#     GUARDED: restoring onto one is refused unless you pass --force-prod AND
#     type the DB name to confirm. Default usage restores to a fresh SCRATCH DB.
#   * Scratch restore creates a NEW database and refuses if it already exists
#     (use --drop to recreate it, an explicit destructive opt-in).
#
# SOURCE: either a local dump file (--source FILE) or a named object pulled from
# gdrive (--from-gdrive NAME → rclone copy from BACKUP_RCLONE_REMOTE).
#
# CONNECTION + EXECUTION MODES are identical to backup.sh: env-driven, with an
# optional AIT_PG_CONTAINER to run pg_restore/createdb INSIDE the postgres
# container (the operator's Windows path — no host pg client tools needed).
#
# FAILS LOUDLY: set -euo pipefail.
#
# Examples:
#   # DoD: restore the app dump to a scratch DB via the container, then verify.
#   AIT_PG_CONTAINER=ait-postgres ops/restore.sh \
#     --source backups/app_20260605T120000Z.dump --target app_scratch
#
#   # Pull the latest-named store dump from gdrive and restore to scratch.
#   ops/restore.sh --from-gdrive langgraph_store_20260605T120000Z.dump \
#     --target store_scratch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${AIT_ENV_FILE:-$REPO_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; . "$ENV_FILE"; set +a
fi

POSTGRES_HOST="${POSTGRES_HOST:-127.0.0.1}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-trading_agent}"
BACKUP_RCLONE_REMOTE="${BACKUP_RCLONE_REMOTE:-gdrive:ai-trading-agent-backups}"
AIT_PG_CONTAINER="${AIT_PG_CONTAINER:-}"
AIT_PG_CONTAINER_PORT="${AIT_PG_CONTAINER_PORT:-5432}"

PROD_DBS="app langgraph_checkpoints langgraph_store"

SOURCE=""
FROM_GDRIVE=""
TARGET=""
FORCE_PROD=0
DROP_FIRST=0

usage() {
  cat <<'EOF'
Usage: ops/restore.sh (--source FILE | --from-gdrive NAME) --target DB
                      [--drop] [--force-prod] [-h|--help]

  --source FILE     local pg_dump custom-format file to restore.
  --from-gdrive N   pull object N from BACKUP_RCLONE_REMOTE first, then restore.
  --target DB       REQUIRED. The database to restore INTO. Use a scratch name.
  --drop            dropdb --if-exists the target before recreating (scratch only;
                    explicit destructive opt-in).
  --force-prod      allow --target to be a prod DB (app/langgraph_*). Requires
                    typing the DB name to confirm; restores IN-PLACE (--clean).

Env: POSTGRES_HOST/PORT/USER/PASSWORD, BACKUP_RCLONE_REMOTE,
     AIT_PG_CONTAINER (run pg_restore/createdb inside that container).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --from-gdrive) FROM_GDRIVE="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --drop) DROP_FIRST=1; shift ;;
    --force-prod) FORCE_PROD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "restore.sh: unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$TARGET" ]]; then
  echo "restore.sh: --target is REQUIRED (refusing to guess a target)" >&2
  usage >&2; exit 2
fi
if [[ -z "$SOURCE" && -z "$FROM_GDRIVE" ]]; then
  echo "restore.sh: need --source FILE or --from-gdrive NAME" >&2
  usage >&2; exit 2
fi
if [[ -n "$SOURCE" && -n "$FROM_GDRIVE" ]]; then
  echo "restore.sh: --source and --from-gdrive are mutually exclusive" >&2
  exit 2
fi
if [[ -z "${POSTGRES_PASSWORD:-}" ]]; then
  echo "restore.sh: POSTGRES_PASSWORD is not set (export it or put it in $ENV_FILE)" >&2
  exit 1
fi

# ── prod guard ─────────────────────────────────────────────────────────
is_prod=0
for p in $PROD_DBS; do [[ "$TARGET" == "$p" ]] && is_prod=1; done
if [[ "$is_prod" -eq 1 ]]; then
  if [[ "$FORCE_PROD" -ne 1 ]]; then
    echo "restore.sh: REFUSING to restore onto prod DB '$TARGET' without --force-prod." >&2
    echo "            Restore to a scratch DB instead, e.g. --target ${TARGET}_scratch" >&2
    exit 1
  fi
  echo "restore.sh: !! about to restore IN-PLACE onto PROD DB '$TARGET' (--clean) !!" >&2
  printf "restore.sh: type the database name '%s' to confirm: " "$TARGET" >&2
  read -r confirm
  if [[ "$confirm" != "$TARGET" ]]; then
    echo "restore.sh: confirmation mismatch; aborting." >&2
    exit 1
  fi
fi

# ── pull from gdrive if requested ──────────────────────────────────────
CLEANUP_DIR=""
if [[ -n "$FROM_GDRIVE" ]]; then
  if ! command -v rclone >/dev/null 2>&1; then
    echo "restore.sh: ERROR rclone not found but --from-gdrive requested." >&2
    exit 1
  fi
  CLEANUP_DIR="$(mktemp -d)"
  trap '[[ -n "$CLEANUP_DIR" ]] && rm -rf "$CLEANUP_DIR"' EXIT
  RCLONE_FLAGS=()
  [[ -n "${BACKUP_RCLONE_CONFIG:-}" ]] && RCLONE_FLAGS+=(--config "$BACKUP_RCLONE_CONFIG")
  echo "restore.sh: rclone copy $BACKUP_RCLONE_REMOTE/$FROM_GDRIVE -> $CLEANUP_DIR/"
  rclone "${RCLONE_FLAGS[@]}" copy "$BACKUP_RCLONE_REMOTE/$FROM_GDRIVE" "$CLEANUP_DIR/"
  SOURCE="$CLEANUP_DIR/$FROM_GDRIVE"
fi

if [[ ! -s "$SOURCE" ]]; then
  echo "restore.sh: ERROR source dump not found or empty: $SOURCE" >&2
  exit 1
fi

# ── pg client invocation: host PATH vs inside the container ────────────
if [[ -n "$AIT_PG_CONTAINER" ]]; then
  CONN_HOST="127.0.0.1"; CONN_PORT="$AIT_PG_CONTAINER_PORT"
  echo "restore.sh: using pg client inside container '$AIT_PG_CONTAINER'"
  pg_tool() { docker exec -i -e PGPASSWORD="$POSTGRES_PASSWORD" "$AIT_PG_CONTAINER" "$@"; }
else
  CONN_HOST="$POSTGRES_HOST"; CONN_PORT="$POSTGRES_PORT"
  echo "restore.sh: using host pg client tools"
  pg_tool() { PGPASSWORD="$POSTGRES_PASSWORD" "$@"; }
fi
CONN=(-h "$CONN_HOST" -p "$CONN_PORT" -U "$POSTGRES_USER")

echo "restore.sh: source=$SOURCE target=$TARGET (prod=$is_prod)"

if [[ "$is_prod" -eq 1 ]]; then
  # In-place prod restore: --clean --if-exists drops+recreates objects inside the
  # existing DB. (Reached only after the typed confirmation above.)
  pg_tool pg_restore "${CONN[@]}" --no-owner --clean --if-exists -d "$TARGET" < "$SOURCE"
else
  # Scratch restore: (optionally drop, then) create a fresh DB and load into it.
  if [[ "$DROP_FIRST" -eq 1 ]]; then
    echo "restore.sh: dropdb --if-exists $TARGET"
    pg_tool dropdb "${CONN[@]}" --if-exists "$TARGET"
  fi
  echo "restore.sh: createdb $TARGET"
  # createdb fails if it already exists — that's the no-clobber guard. Use --drop
  # to recreate a scratch DB intentionally.
  pg_tool createdb "${CONN[@]}" -O "$POSTGRES_USER" "$TARGET"
  echo "restore.sh: pg_restore -> $TARGET"
  pg_tool pg_restore "${CONN[@]}" --no-owner -d "$TARGET" < "$SOURCE"
fi

echo "restore.sh: DONE. Verify, e.g.:"
if [[ -n "$AIT_PG_CONTAINER" ]]; then
  echo "  docker exec -e PGPASSWORD=\$POSTGRES_PASSWORD $AIT_PG_CONTAINER \\"
  echo "    psql -h $CONN_HOST -p $CONN_PORT -U $POSTGRES_USER -d $TARGET -c '\\dt'"
else
  echo "  PGPASSWORD=\$POSTGRES_PASSWORD psql -h $CONN_HOST -p $CONN_PORT -U $POSTGRES_USER -d $TARGET -c '\\dt'"
fi
