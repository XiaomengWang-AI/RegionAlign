#!/usr/bin/env bash
# Run one review round: Codex reads the diff (read-only), the project
# self-check runs in parallel, and both verdicts land in state/last_review.json.
#
# Codex NEVER writes to the tree. Claude is the only writer; Codex only judges.
# Usage: review.sh [--no-selfcheck] [--all] [--base <ref>] [-- <file>...]
set -uo pipefail

RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$RL_DIR/lib.sh"

NO_SELFCHECK=0; FORCE_ALL=0; BASE_REF=""; EXPLICIT=()
while [ $# -gt 0 ]; do
  case "$1" in
    --no-selfcheck) NO_SELFCHECK=1; shift ;;
    --all)          FORCE_ALL=1; shift ;;
    --base)         BASE_REF="${2:-}"; shift 2 ;;
    --)             shift; EXPLICIT=("$@"); break ;;
    *)              EXPLICIT+=("$1"); shift ;;
  esac
done

cd "$REPO_ROOT" || exit 1
RUN_ID="$(date +%Y%m%d-%H%M%S)"
PROMPT_FILE="$STATE_DIR/prompt-$RUN_ID.md"
CODEX_OUT="$STATE_DIR/codex-$RUN_ID.json"
CODEX_LOG="$STATE_DIR/codex-$RUN_ID.log"
SC_LOG="$STATE_DIR/selfcheck-$RUN_ID.log"

# ---------------------------------------------------------------- scope -----
if [ ${#EXPLICIT[@]} -gt 0 ]; then
  printf '%s\n' "${EXPLICIT[@]}" > "$STATE_DIR/scope.txt"
elif [ -n "$BASE_REF" ]; then
  git diff --name-only "$BASE_REF"...HEAD > "$STATE_DIR/scope.txt"
elif [ "$FORCE_ALL" = 1 ]; then
  git status --porcelain --untracked-files=all | sed 's/^...//' > "$STATE_DIR/scope.txt"
else
  rl_scope_files > "$STATE_DIR/scope.txt"
fi
FILES=$(grep -c . "$STATE_DIR/scope.txt" 2>/dev/null || echo 0)

if [ "$FILES" -eq 0 ]; then
  echo "review-loop: nothing changed this session — nothing to review."
  python3 - "$LAST" "$(rl_signature)" <<'PY'
import json,sys
json.dump({"verdict":"clean","signature":sys.argv[2],"summary":"empty scope",
           "findings":[],"selfcheck":"skipped","frozen_hits":[]},open(sys.argv[1],"w"))
PY
  rl_iter_reset; exit 0
fi

if [ -n "$BASE_REF" ] || [ "$FORCE_ALL" = 1 ] || [ ${#EXPLICIT[@]} -gt 0 ]; then
  SCOPE_ARR=(); while IFS= read -r f; do [ -n "$f" ] && SCOPE_ARR+=("$f"); done < "$STATE_DIR/scope.txt"
  DIFF=$(git diff HEAD --no-color -- "${SCOPE_ARR[@]}" 2>/dev/null)
else
  DIFF=$(rl_build_diff)
fi
DIFF_LINES=$(printf '%s' "$DIFF" | grep -c '' || echo 0)

if [ "$DIFF_LINES" -gt "$MAX_DIFF_LINES" ]; then
  echo "review-loop: diff is $DIFF_LINES lines (limit $MAX_DIFF_LINES)." >&2
  echo "Narrow the scope: .claude/review-loop/review.sh -- path/one.py path/two.py" >&2
  exit 2
fi

FROZEN_HITS=$(while IFS= read -r f; do rl_is_frozen "$f" && echo "$f"; done < "$STATE_DIR/scope.txt")

echo "review-loop: reviewing $FILES file(s), $DIFF_LINES diff lines"
sed 's/^/  - /' "$STATE_DIR/scope.txt"
[ -n "$FROZEN_HITS" ] && { echo "  !! FROZEN-zone files touched:"; echo "$FROZEN_HITS" | sed 's/^/     /'; }

# ------------------------------------------------- self-check in parallel ---
SC_PID=""; SC_STATUS="skipped"
if [ "$NO_SELFCHECK" = 0 ] && [ "$SELFCHECK_ENABLED" = 1 ] && grep -q '\.py$' "$STATE_DIR/scope.txt"; then
  echo "review-loop: project self-check started in background (AGENTS.md §4)"
  ( cd "$REPO_ROOT" && timeout "$SELFCHECK_TIMEOUT" bash -c "$SELFCHECK_CMD" ) >"$SC_LOG" 2>&1 &
  SC_PID=$!
fi

# ------------------------------------------------------------- prompt -------
{
  cat <<'HDR'
You are an adversarial reviewer for a research repository. Another AI agent
(Claude) just wrote the diff below. Your job is to find real defects in it.

The repository root holds AGENTS.md — read it. It states the one research
hypothesis (a preregistered region is the attacker's only constraint), the
append-only results/ zone, and the rules that must never be broken.
Read any file you need; you have read-only access. Do not modify anything.

Report ONLY defects introduced or left behind by this diff:
  P0  breaks correctness, or violates a documented invariant / zone rule
      (e.g. modifying or overwriting results/, importing code from outside
      the package, reintroducing a cell map or a per-bit/direction limit,
      using the test split to select an attack, letting a search judge its
      own legality)
  P1  a real bug: wrong result, crash, resource leak, broken contract,
      silently wrong edge case, test that cannot fail
  P2  minor; reported but never blocks

Hard rules:
- No style, naming, formatting or "consider adding a comment" findings.
- Every finding needs a concrete failure scenario: inputs/state -> wrong output.
- If you cannot name how it breaks, it is not a finding. Say clean instead.
- Do not invent work. A correct diff gets verdict "clean" and an empty list.
HDR
  if [ -n "$FROZEN_HITS" ]; then
    echo; echo "ATTENTION: the diff touches these FROZEN-zone paths, which is a P0"
    echo "(results/ is append-only evidence):"; echo "$FROZEN_HITS" | sed 's/^/  /'
  fi
  echo; echo "Files in scope:"; sed 's/^/  /' "$STATE_DIR/scope.txt"
  echo; echo "=== DIFF ==="; printf '%s\n' "$DIFF"; echo "=== END DIFF ==="
} > "$PROMPT_FILE"

# -------------------------------------------------------------- codex -------
CODEX_ARGS=(exec -s read-only --output-schema "$RL_DIR/schema.json" -o "$CODEX_OUT")
[ -n "$CODEX_MODEL" ]  && CODEX_ARGS+=(-m "$CODEX_MODEL")
[ -n "$CODEX_EFFORT" ] && CODEX_ARGS+=(-c "model_reasoning_effort=\"$CODEX_EFFORT\"")

echo "review-loop: codex reviewing..."
# `< /dev/null` is load-bearing: codex exec reads stdin and hangs forever
# without it when launched from a hook or a non-tty parent.
timeout "$CODEX_TIMEOUT" codex "${CODEX_ARGS[@]}" - < "$PROMPT_FILE" > "$CODEX_LOG" 2>&1
CODEX_RC=$?
[ "$CODEX_RC" -ne 0 ] && { echo "review-loop: codex exited $CODEX_RC (see $CODEX_LOG)"; tail -15 "$CODEX_LOG"; }

if [ -n "$SC_PID" ]; then
  echo "review-loop: waiting for self-check..."
  if wait "$SC_PID"; then SC_STATUS="pass"; else SC_STATUS="fail"; fi
  echo "review-loop: self-check $SC_STATUS"
fi

# ------------------------------------------------------------- verdict ------
SIG="$(rl_signature)"
python3 - "$CODEX_OUT" "$LAST" "$SIG" "$SC_STATUS" "$SC_LOG" "$BLOCKING_SEVERITIES" "$CODEX_RC" <<'PY'
import json, os, sys
codex_out, last, sig, sc_status, sc_log, blocking, rc = sys.argv[1:8]
blocking = blocking.split()
try:
    d = json.load(open(codex_out))
    findings = d.get("findings") or []
    summary  = d.get("summary") or ""
    cverdict = d.get("verdict", "issues")
except Exception:
    findings, summary, cverdict = [], "", "error"

sc_tail = ""
if sc_status == "fail" and os.path.exists(sc_log):
    sc_tail = "".join(open(sc_log, errors="replace").readlines()[-40:])

blockers = [f for f in findings if f.get("severity") in blocking]
if cverdict == "error" or rc != "0":
    verdict = "error"
elif blockers or sc_status == "fail":
    verdict = "issues"
else:
    verdict = "clean"

json.dump({"verdict": verdict, "codex_verdict": cverdict, "signature": sig,
           "summary": summary, "findings": findings, "blockers": len(blockers),
           "selfcheck": sc_status, "selfcheck_tail": sc_tail},
          open(last, "w"), indent=2)

print()
print("=" * 66)
print(f"  CODEX VERDICT: {verdict.upper()}   (self-check: {sc_status})")
print("=" * 66)
if summary: print(summary)
for f in sorted(findings, key=lambda x: x.get("severity", "P2")):
    print(f"\n[{f.get('severity')}] {f.get('file')}:{f.get('line')}  {f.get('title')}")
    print(f"      why: {f.get('why')}")
    print(f"      fix: {f.get('fix')}")
if not findings: print("No findings.")
if sc_tail:
    print("\n--- self-check FAILED, tail ---")
    print(sc_tail)
PY

VERDICT="$(rl_json_get "$LAST" verdict)"
[ "$VERDICT" = "clean" ] && rl_iter_reset
exit 0
