---
description: Have Codex (GPT) adversarially review this session's changes, then fix every P0/P1 finding
argument-hint: "[--no-selfcheck] [--all] [--base <ref>] [-- <files...>]"
allowed-tools: Bash(.claude/review-loop/review.sh:*), Read, Edit, Write, Grep, Glob
---

Run one Codex review round over this session's changes:

!`.claude/review-loop/review.sh $ARGUMENTS`

Now act on the verdict above:

- Fix **every P0 and P1** finding. P2 is advisory — mention it, don't necessarily act.
- If the self-check failed, fix that too; its tail is in the output above.
- A finding you believe is wrong: say so explicitly with the reason, and skip it.
  Do not silently ignore one.
- Respect `AGENTS.md`: never edit anything under `results/` to satisfy a finding. If a
  finding can only be fixed there, stop and tell me instead.
- When you have fixed things, run the command again until the verdict is CLEAN.

Report to me only the final state: what Codex flagged, what you changed, and
anything you deliberately did not change.
