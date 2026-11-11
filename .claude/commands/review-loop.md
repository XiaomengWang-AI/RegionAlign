---
description: Arm or disarm the automatic Claude-writes / Codex-reviews gate
argument-hint: "on | off | status | reset"
allowed-tools: Bash(.claude/review-loop/loop.sh:*)
---

!`.claude/review-loop/loop.sh ${ARGUMENTS:-status}`

Tell me the result in one line. If it was just armed, also remind me that from
now on any turn in which you edit code will not end until Codex returns CLEAN
or the round budget runs out.
