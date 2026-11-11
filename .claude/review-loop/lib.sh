#!/usr/bin/env bash
# Shared helpers: where the repo is, what counts as "this session's changes",
# and a content signature used to tell "already reviewed" from "changed since".

RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$RL_DIR/../.." && pwd)}"
STATE_DIR="$RL_DIR/state"
mkdir -p "$STATE_DIR"

# shellcheck source=config.sh
. "$RL_DIR/config.sh"

MARKER="$STATE_DIR/session_start.marker"
LEDGER="$STATE_DIR/touched.txt"
LAST="$STATE_DIR/last_review.json"
ITERFILE="$STATE_DIR/iterations"
ENABLEDFILE="$STATE_DIR/enabled"

rl_is_ignored() {
  local f="$1" p
  for p in $IGNORE_PREFIXES; do [[ "$f" == "$p"* ]] && return 0; done
  return 1
}

rl_is_frozen() {
  local f="$1" p
  for p in $FROZEN_PREFIXES; do [[ "$f" == "$p"* ]] && return 0; done
  return 1
}

# Files this session changed.
#
# Two independent sources, unioned, because neither alone is enough:
#   1. the ledger, written by the PostToolUse hook  -> catches Edit/Write
#   2. git-dirty files newer than the session marker -> catches edits made
#      through Bash (sed, heredocs), which no tool hook sees
# The marker is what keeps a pre-existing dirty tree out of the scope.
rl_scope_files() {
  {
    [ -f "$LEDGER" ] && cat "$LEDGER"
    if [ -f "$MARKER" ]; then
      git -C "$REPO_ROOT" status --porcelain --untracked-files=all 2>/dev/null \
      | sed 's/^...//' \
      | while IFS= read -r f; do
          case "$f" in *" -> "*) f="${f##* -> }" ;; esac
          f="${f%\"}"; f="${f#\"}"
          [ -e "$REPO_ROOT/$f" ] || continue
          [ "$REPO_ROOT/$f" -nt "$MARKER" ] && printf '%s\n' "$f"
        done
    fi
  } 2>/dev/null \
  | sed "s|^$REPO_ROOT/||" \
  | sort -u \
  | while IFS= read -r f; do
      [ -n "$f" ] || continue
      rl_is_ignored "$f" && continue
      printf '%s\n' "$f"
    done
}

# Content fingerprint of the scope. Changes the moment Claude edits anything
# in scope, which is exactly the "needs a fresh review" signal.
rl_signature() {
  { rl_scope_files | while IFS= read -r f; do
      if [ -f "$REPO_ROOT/$f" ]; then
        sha1sum "$REPO_ROOT/$f" 2>/dev/null || echo "UNREADABLE $f"
      else
        echo "GONE $f"
      fi
    done
  } | sha1sum | cut -d' ' -f1
}

# Unified diff for the scope: tracked changes vs HEAD, plus full text of
# untracked files (which have no HEAD side to diff against).
rl_build_diff() {
  local -a files=()
  while IFS= read -r f; do files+=("$f"); done < <(rl_scope_files)
  [ ${#files[@]} -eq 0 ] && return 0

  git -C "$REPO_ROOT" diff HEAD --no-color -- "${files[@]}" 2>/dev/null

  local f
  for f in "${files[@]}"; do
    if ! git -C "$REPO_ROOT" ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      [ -f "$REPO_ROOT/$f" ] || continue
      printf '\n--- /dev/null\n+++ b/%s\n' "$f"
      sed 's/^/+/' "$REPO_ROOT/$f"
    fi
  done
}

rl_enabled()   { [ -f "$ENABLEDFILE" ]; }
rl_iter()      { cat "$ITERFILE" 2>/dev/null || echo 0; }
rl_iter_bump() { echo $(( $(rl_iter) + 1 )) > "$ITERFILE"; }
rl_iter_reset(){ echo 0 > "$ITERFILE"; }

rl_json_get() { python3 -c 'import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
for k in sys.argv[2].split("."):
    d = d.get(k) if isinstance(d, dict) else None
    if d is None: print(""); sys.exit(0)
print(d if not isinstance(d,(dict,list)) else json.dumps(d))' "$1" "$2" 2>/dev/null
}
