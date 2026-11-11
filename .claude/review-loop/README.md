# review-loop — Claude writes, Codex reviews, loop until clean

Claude is the only writer. Codex runs `-s read-only` and can only judge, so the
two models can never fight over the same file.

## Use it

    /review-loop on        # arm the automatic gate
    /codex-review          # one review round right now, on demand
    /review-loop status    # what is armed, what is in scope, last verdict
    /review-loop off

Plain shell, no Claude needed:

    .claude/review-loop/review.sh                  # this session's changes
    .claude/review-loop/review.sh -- a.py b.py     # explicit files
    .claude/review-loop/review.sh --base main      # a branch's changes
    .claude/review-loop/review.sh --no-selfcheck   # skip the unittest run

## How the gate closes

A `Stop` hook runs on every turn. It lets Claude stop unless all of these hold:
the loop is armed, code changed this session, and the round budget is not spent.
Otherwise it blocks and tells Claude to run `review.sh`.

`review.sh` fingerprints the scope (sha1 of every in-scope file). The gate
compares that fingerprint with the one stored in `state/last_review.json`:

| fingerprint | stored verdict | gate does |
|---|---|---|
| differs | — | block: code changed, review it |
| same | `issues` | block: findings still open |
| same | `clean` | allow stop |
| same | `error` | allow stop — a broken reviewer must not trap the loop |

So the loop ends on its own the moment Codex comes back clean, and it can never
spin more than `MAX_ITERATIONS` (default 3) rounds.

## What counts as "this session's changes"

The union of two sources, because neither alone is enough:

1. a ledger written by a `PostToolUse` hook — catches `Edit`/`Write`;
2. git-dirty files newer than `state/session_start.marker` — catches edits made
   through `Bash` (`sed`, heredocs), which no tool hook sees.

The marker is what keeps a tree that was already dirty out of
scope. `MAX_DIFF_LINES` (default 4000) is a second backstop: a larger diff is
refused rather than shipped to OpenAI.

## Verdict

`verdict = clean` requires **both**: no P0/P1 from Codex, and a passing
`AGENTS.md` §4 self-check. The self-check runs in parallel with Codex, so a
round costs about one self-check (~3 min), not two serial waits. It is skipped
when no `.py` file is in scope.

## Knobs

`config.sh`, all overridable from the environment: `CODEX_MODEL`,
`CODEX_EFFORT` (default `high`; `ultra` from `~/.codex/config.toml` is much
slower), `MAX_ITERATIONS`, `MAX_DIFF_LINES`, `BLOCKING_SEVERITIES`,
`SELFCHECK_ENABLED`, `SELFCHECK_CMD`, `FROZEN_PREFIXES`.

## Notes

- Codex reads the repo's `AGENTS.md` natively, so it reviews against the same
  zone rules and invariants Claude follows. Files in a FROZEN path are called
  out in the prompt as a P0 unless `AGENTS.md` allows them.
- Reviews send the diff to OpenAI and count against your Codex limits, not your
  Anthropic subscription.
- `codex exec` reads stdin and hangs forever if stdin is an open pipe. Every
  invocation here redirects stdin from a file. Do not remove that.
