"""Tests for the None-safe VQA re-evaluation of summary crashes."""
import json
import random
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from regionalign.experiments import vqa as Q
from regionalign.experiments import vqa_reeval as E


def _row(valid, delta, drop, success=False):
    return {"valid": valid, "delta_nll": delta, "relative_score_drop": drop,
            "success": success}


class SummaryTests(unittest.TestCase):
    def test_matches_original_summary_without_none(self):
        rng = random.Random(7)
        for _ in range(200):
            results = {str(k): _row(rng.random() < 0.7, rng.uniform(-1, 3),
                                    rng.uniform(-0.5, 1), rng.random() < 0.3)
                       for k in Q.EVAL_K}
            self.assertEqual(E.summary_none_safe(results), Q._summary(results))
        self.assertEqual(E.summary_none_safe({}), Q._summary({}))

    def test_original_crashes_and_safe_version_skips_none(self):
        results = {"1": _row(True, 0.5, None), "16": _row(True, 2.0, None),
                   "4": _row(False, 9.0, 0.9)}
        with self.assertRaises(TypeError):
            Q._summary(results)
        summary = E.summary_none_safe(results)
        self.assertEqual(summary["max_delta_nll"], 2.0)
        self.assertIsNone(summary["max_relative_score_drop"])
        self.assertEqual(summary["valid_evaluations"], 2)
        self.assertFalse(summary["success_any"])
        mixed = {"1": _row(True, None, 0.2), "2": _row(True, 1.0, None)}
        self.assertEqual(E.summary_none_safe(mixed)["max_relative_score_drop"], 0.2)
        self.assertEqual(E.summary_none_safe(mixed)["max_delta_nll"], 1.0)

    def test_patch_is_restored(self):
        original = Q._summary
        with self.assertRaises(RuntimeError):
            with E._summary_patched():
                self.assertIs(Q._summary, E.summary_none_safe)
                raise RuntimeError
        self.assertIs(Q._summary, original)


class SelectionTests(unittest.TestCase):
    def test_only_the_summary_crash_is_selected(self):
        crash = {"status": "evaluation_error", "error": E.TARGET_ERROR,
                 "traceback": 'File "vqa.py", line 936, in _summary\n'}
        self.assertTrue(E.needs_reeval(crash))
        self.assertFalse(E.needs_reeval({**crash, "status": "complete"}))
        self.assertFalse(E.needs_reeval({**crash, "error": "TypeError: other"}))
        self.assertFalse(E.needs_reeval({**crash, "traceback": "in _evaluate_one"}))
        self.assertFalse(E.needs_reeval({"status": "artifact_error"}))


class OutputTests(unittest.TestCase):
    def test_output_must_be_a_separate_reeval_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = root / "plans" / "vqa"
            (run / "eval").mkdir(parents=True)
            (run / "plan.json").write_text("{}")
            for bad in (run, run / "eval", run / "reeval", root / "plans"):
                with self.assertRaises(ValueError):
                    E._check_output(run, bad)
            other = root / "other_run"
            other.mkdir()
            (other / "plan.json").write_text("{}")
            for bad in (other, other / "reeval", other / "a" / "b"):
                with self.assertRaises(ValueError):
                    E._check_output(run, bad)
            layout = root / "layout_validation" / "done"
            layout.mkdir(parents=True)
            (layout / "report.json").write_text("{}")
            (layout / "protocol.json").write_text("{}")
            with self.assertRaises(ValueError):
                E._check_output(run, layout / "reeval")
            E._check_output(run, root / "fresh")
            initialized = root / "initialized"
            initialized.mkdir()
            (initialized / "binding.json").write_text("{}")
            E._check_output(run, initialized)
            with self.assertRaises(ValueError):
                E._check_output(run, initialized / "nested")

    def test_shards_require_a_matching_init_binding(self):
        plan = {"sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "reeval"
            out.mkdir()
            (out / "report.json").write_text("{}")  # e.g. a completed layout run
            with self.assertRaises(RuntimeError):
                E._verify_binding("/run", plan, out)
            (out / "binding.json").write_text(json.dumps(E._record("/run", plan)))
            E._verify_binding("/run", plan, out)
            with self.assertRaises(RuntimeError):
                E._verify_binding("/run", {"sha256": "b" * 64}, out)

    def test_init_refuses_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            existing = root / "done"
            existing.mkdir()
            (existing / "report.json").write_text("{}")
            plan = {"sha256": "a" * 64}
            with mock.patch.object(E.Q, "load_plan", return_value=(root / "run", plan)), \
                    mock.patch.object(E.C, "output_path", side_effect=lambda p: Path(p)), \
                    mock.patch.object(E.C, "fresh_output_dir",
                                      side_effect=lambda g, l, p: (Path(p).mkdir(exist_ok=False)
                                                                  or Path(p))):
                with self.assertRaises(ValueError):
                    E.init_output(root / "run", existing)
                empty = root / "empty"
                empty.mkdir()
                with self.assertRaises(FileExistsError):
                    E.init_output(root / "run", empty)
                made = E.init_output(root / "run", root / "fresh")
            self.assertEqual(json.loads((made / "binding.json").read_text()),
                             E._record(root / "run", plan))


if __name__ == "__main__":
    unittest.main()
