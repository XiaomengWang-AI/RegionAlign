#!/usr/bin/env bash
# SessionStart: drop a marker so the loop can tell "changed in this session"
# from "already dirty before Claude touched anything".
set -uo pipefail
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$RL_DIR/lib.sh"
: > "$LEDGER"
rl_iter_reset
rm -f "$LAST"
touch "$MARKER"
exit 0
