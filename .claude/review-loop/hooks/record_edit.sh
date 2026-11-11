#!/usr/bin/env bash
# PostToolUse(Edit|Write|MultiEdit|NotebookEdit): append the edited path to the
# ledger. Edits made through Bash are caught separately, by mtime vs MARKER.
set -uo pipefail
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$RL_DIR/lib.sh"
python3 -c '
import json,sys,os
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
p=(d.get("tool_input") or {}).get("file_path") or (d.get("tool_input") or {}).get("notebook_path")
if p: print(os.path.relpath(p, sys.argv[1]) if os.path.isabs(p) else p)
' "$REPO_ROOT" >> "$LEDGER" 2>/dev/null
exit 0
