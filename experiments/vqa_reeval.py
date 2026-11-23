"""Re-evaluate VQA reports that crashed in the summary, reusing the frozen artifacts.

``vqa._summary`` takes ``max`` over every valid budget's ``relative_score_drop``.
That drop is ``None`` whenever the deployed no-fault layout scores 0, so on
INT4 VLMs (clean VQA score 0) the summary raised ``TypeError`` after every
budget had already been measured, and the report was saved as
``evaluation_error`` without its results.

This module leaves the frozen plan and its sources untouched: it loads the
original plan with ``vqa.load_plan`` (so the source-drift check still binds),
re-runs ``vqa._evaluate_one`` on the original search artifacts with a summary
that skips ``None`` values, and publishes each report once under a separate
output directory.  Only items whose original report is exactly that crash are
re-evaluated; every other report is left as the original evaluation.  When no
value is ``None`` the summary is identical to ``vqa._summary``.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from regionalign import config as C
from regionalign.experiments import clean_mmlu as M
from regionalign.experiments import vqa as Q

SCOPE = "vqa_regionalign_reeval_none_safe_summary"
# Files that mark an existing run of any experiment type (VQA/LLM plans, layout
# validation, earlier re-evaluations); an output may not sit at or below one.
RUN_MARKERS = ("plan.json", "status.json", "artifacts", "report.json", "protocol.json",
               "binding.json")
TARGET_ERROR = "TypeError: '>' not supported between instances of 'NoneType' and 'NoneType'"


def summary_none_safe(results):
    """``vqa._summary`` with ``None`` damages skipped instead of compared."""
    valid = [row for row in results.values() if row.get("valid")]

    def largest(key):
        values = [row[key] for row in valid if row[key] is not None]
        return max(values, default=None)

    return {"valid_evaluations": len(valid), "registered_evaluations": len(Q.EVAL_K),
            "max_delta_nll": largest("delta_nll"),
            "max_relative_score_drop": largest("relative_score_drop"),
            "success_any": any(row.get("success") for row in valid)}


def needs_reeval(report):
    """True only for the summary crash this module repairs."""
    return (report.get("status") == "evaluation_error"
            and report.get("error") == TARGET_ERROR
            and "in _summary" in (report.get("traceback") or ""))


@contextmanager
def _summary_patched():
    original = Q._summary
    Q._summary = summary_none_safe
    try:
        yield
    finally:
        Q._summary = original


def _record(run, plan):
    return {"scope": SCOPE, "run": str(run), "plan_sha256": plan["sha256"],
            "module_sha256": M.sha256(Path(__file__)),
            "vqa_sha256": M.sha256(Path(Q.__file__))}


def _check_output(run, output):
    """The output must be a separate re-evaluation directory, never inside a run."""
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or run in output.parents or output in run.parents:
        raise ValueError("re-evaluation output must be outside the original run")
    # The output's own binding.json is allowed (init wrote it; shards verify it).
    own = [marker for marker in RUN_MARKERS if marker != "binding.json"]
    for directory, markers in ((output, own), *((parent, RUN_MARKERS)
                                                 for parent in output.parents)):
        for marker in markers:
            if (directory / marker).exists():
                raise ValueError(f"{output} is inside an experiment run ({directory})")


def init_output(run, output):
    """Create a fresh output directory and publish its binding; run once, before any shard."""
    run, plan = Q.load_plan(run)
    _check_output(run, C.output_path(output))
    target = C.fresh_output_dir(None, None, output)  # refuses any existing directory
    M.write_json_once(target / "binding.json", _record(run, plan))
    return target


def _verify_binding(run, plan, output):
    """Shards only write into a directory ``init_output`` created for this plan."""
    path = Path(output) / "binding.json"
    if not path.is_file():
        raise RuntimeError(f"{output} was not created by vqa_reeval init")
    if M.read_json(path) != _record(run, plan):
        raise RuntimeError(f"{path} binds a different plan or source; use a new output")


def run_reevaluate(run, condition_id, output, shard="0/1", device="cuda:0", log=print):
    run, plan = Q.load_plan(run)
    condition = Q._condition(plan, condition_id)
    Q._require_preflight(run, plan, condition)
    output = C.output_path(output)  # results/-contained, like every other command
    _check_output(run, output)
    _verify_binding(run, plan, output)
    todo = []
    for item in Q.shard_items(Q.planned_items(condition, plan), shard):
        old = Q._eval_path(run, condition_id, item["name"])
        if old.is_file() and needs_reeval(M.read_json(old)):
            todo.append((item, old))
    log(f"{condition_id} shard {shard}: {len(todo)} reports to re-evaluate")
    if not todo:
        return
    runtime, rows = Q._load_runtime(run, plan, condition, "dev", device)
    try:
        Q._assert_frozen(run, plan)
        logits = runtime.prefill_logits(rows[:Q.P.LOGIT_ROWS])
        clean = Q._task_metrics(runtime, rows)
        clean.update({"logits_sha256": Q._logits_digest(logits),
                      "quantization": runtime.quantization,
                      "load_timings": runtime.load_timing, "_logits": logits})
        fingerprint = Q.R.model_checksum(runtime.lm)
        public = {key: value for key, value in clean.items() if key != "_logits"}
        target = output / "eval" / Q._safe_component(condition_id)
        target.mkdir(parents=True, exist_ok=True)
        for item, old in todo:
            path = target / item["name"]
            source = Q._artifact_path(run, condition_id, item["name"])
            if path.exists() or not source.exists():
                continue
            with Q._item_lock(output, "reevaluate", condition_id, item["name"]):
                if path.exists():
                    continue
                Q._assert_frozen(run, plan)
                artifact = M.read_json(source)
                started = time.monotonic()
                try:
                    with _summary_patched():
                        report = Q._evaluate_one(runtime, rows,
                                                 {**public, "_logits": clean["_logits"]},
                                                 condition, item, artifact, M.sha256(source),
                                                 plan, log)
                except Exception as error:
                    if Q._resource_error(error):
                        raise
                    report = {"schema_version": Q.SCHEMA_VERSION, "scope": Q.SCOPE,
                              "condition": condition_id, "region": item["region"],
                              "repeat": item["repeat"], "seed": item["seed"],
                              "arm": item["arm"], "plan_sha256": plan["sha256"],
                              "artifact_file_sha256": M.sha256(source),
                              "status": "evaluation_error",
                              "error": f"{type(error).__name__}: {error}",
                              "traceback": traceback.format_exc()[-4000:],
                              "seconds": time.monotonic() - started}
                report["clean"] = public
                report["reevaluation"] = {"scope": SCOPE,
                                          "original_report": str(old),
                                          "original_report_sha256": M.sha256(old),
                                          "summary": "none_safe"}
                if Q.R.model_checksum(runtime.lm) != fingerprint:
                    raise RuntimeError("model state changed during VQA re-evaluation")
                M.write_json_once(path, report)
    finally:
        runtime.close()


def main(argv=None):
    root = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create the fresh output directory once")
    init.add_argument("--run", required=True)
    init.add_argument("--output", required=True)
    run = commands.add_parser("run", help="re-evaluate one shard into an initialized output")
    run.add_argument("--run", required=True)
    run.add_argument("--condition", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--shard", default="0/1")
    run.add_argument("--device", default="cuda:0")
    args = root.parse_args(argv)
    C.setup_environment()
    if args.command == "init":
        print(init_output(args.run, args.output), flush=True)
    else:
        run_reevaluate(args.run, args.condition, args.output, args.shard, args.device)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
