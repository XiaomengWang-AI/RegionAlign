#!/usr/bin/env bash
# Arm / disarm / inspect the automatic gate.
set -uo pipefail
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$RL_DIR/lib.sh"

case "${1:-status}" in
  on)
    touch "$ENABLEDFILE"; rl_iter_reset; rm -f "$LAST"; touch "$MARKER"
    echo "review-loop: ARMED (max $MAX_ITERATIONS rounds per change)"
    echo "Every turn that edits code now ends with a Codex review + self-check."
    ;;
  off)
    rm -f "$ENABLEDFILE"
    echo "review-loop: DISARMED. ./.claude/review-loop/review.sh still works on demand."
    ;;
  reset)
    rl_iter_reset; rm -f "$LAST"; : > "$LEDGER"; touch "$MARKER"
    echo "review-loop: counters and scope reset"
    ;;
  status)
    rl_enabled && echo "state      : ARMED" || echo "state      : disarmed"
    echo "rounds used: $(rl_iter) / $MAX_ITERATIONS"
    echo "reviewer   : codex ${CODEX_MODEL:-<codex default>} (effort ${CODEX_EFFORT:-<default>}, read-only)"
    echo "self-check : $([ "$SELFCHECK_ENABLED" = 1 ] && echo enabled || echo disabled)"
    echo "scope now  :"; rl_scope_files | sed 's/^/  /' | head -40
    if [ -f "$LAST" ]; then
      echo "last review:"
      python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print("  verdict:",d["verdict"],"| self-check:",d.get("selfcheck"),"| findings:",len(d.get("findings",[])))' "$LAST"
    fi
    ;;
  *) echo "usage: loop.sh {on|off|status|reset}"; exit 1 ;;
esac
