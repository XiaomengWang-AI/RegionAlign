#!/usr/bin/env bash
# Stop hook: the gate. Decides whether Claude is allowed to hand the turn back
# to the human, or must run another Codex review / fix round first.
#
# Exit 0 with no stdout      -> let Claude stop.
# Exit 0 with {"decision":"block","reason":...} -> Claude keeps working.
set -uo pipefail
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$RL_DIR/lib.sh"

INPUT=$(cat)

block() {
  python3 -c 'import json,sys; print(json.dumps({"decision":"block","reason":sys.stdin.read()}))' <<< "$1"
  exit 0
}

rl_enabled || exit 0

SCOPE=$(rl_scope_files)
[ -z "$SCOPE" ] && exit 0          # pure Q&A turn, nothing was edited

ITER=$(rl_iter)
if [ "$ITER" -ge "$MAX_ITERATIONS" ]; then
  # Budget spent. Stop looping and let Claude report honestly to the human.
  exit 0
fi

SIG=$(rl_signature)
CMD=".claude/review-loop/review.sh"

if [ -f "$LAST" ] && [ "$(rl_json_get "$LAST" signature)" = "$SIG" ]; then
  VERDICT=$(rl_json_get "$LAST" verdict)
  case "$VERDICT" in
    clean) exit 0 ;;
    error) exit 0 ;;   # reviewer itself broke; don't trap Claude in a loop
    *)
      rl_iter_bump
      block "Codex review round $((ITER+1))/$MAX_ITERATIONS: the findings from the last review are still open and the code has not changed since.

Blocking findings and the self-check result are in:
  .claude/review-loop/state/last_review.json

Fix every P0 and P1 finding (and make the project self-check pass if it failed), then finish your turn. The gate will re-run the review automatically. Do not summarise for the user yet — they asked for the final clean version only. If you believe a finding is wrong, say so explicitly in your next message with the reason, and move on."
      ;;
  esac
else
  rl_iter_bump
  block "Codex review round $((ITER+1))/$MAX_ITERATIONS: you changed code this session and it has not been reviewed yet.

Run this now with the Bash tool, then act on the result:
  $CMD

It runs Codex (read-only, it cannot edit) over just this session's changes and runs the AGENTS.md §4 self-check in parallel. Fix every P0 and P1 finding, then finish your turn — the gate will review again. Do not summarise for the user until the verdict is CLEAN."
fi
exit 0
