#!/usr/bin/env bash
# Knobs for the Claude-writes / Codex-reviews loop.
# Every value can be overridden from the environment.

# --- Codex reviewer ---------------------------------------------------------
# Empty CODEX_MODEL = whatever ~/.codex/config.toml already selects.
: "${CODEX_MODEL:=}"
# Review is a bounded task; "high" is a good speed/quality point.
# Set to "" to keep the ~/.codex/config.toml default (currently "ultra", slow).
: "${CODEX_EFFORT:=high}"
: "${CODEX_TIMEOUT:=900}"

# --- Loop control -----------------------------------------------------------
# Max automatic review->fix rounds before the gate gives up and hands
# control back to the human. Prevents runaway token burn.
: "${MAX_ITERATIONS:=3}"
# Findings at these severities block the loop. P2 is reported, never blocks.
: "${BLOCKING_SEVERITIES:=P0 P1}"
# Refuse to ship an absurd diff to the reviewer.
: "${MAX_DIFF_LINES:=4000}"

# --- Project self-check (AGENTS.md section 4) -------------------------------
: "${SELFCHECK_ENABLED:=1}"
: "${SELFCHECK_TIMEOUT:=1800}"
# The package is imported as ``regionalign`` from its parent directory.
SELFCHECK_CMD=${SELFCHECK_CMD:-'cd .. && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest discover -s regionalign/tests -t .'}

# --- Repo zones (AGENTS.md section 3) ---------------------------------------
# Codex is told these are off limits; the gate flags any edit that lands here.
# results/ is the append-only evidence tree: an edit recorded there is flagged.
FROZEN_PREFIXES=${FROZEN_PREFIXES:-'results/'}
# Never part of a review scope: tooling state and caches, not package code.
# '../' also matters: the ledger can record paths outside the repository, and a
# single one of those makes `git diff HEAD -- <paths>` fail for the *entire*
# scope ("is outside repository"), so the reviewer silently received only the
# untracked files. A repository review scopes to the repository.
IGNORE_PREFIXES=${IGNORE_PREFIXES:-'.claude/ .git/ .cache/ ../'}
