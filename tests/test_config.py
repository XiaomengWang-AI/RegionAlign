"""Output containment, compatibility links, and delegated runtime settings."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alignment_attack import runtime_config as runtime
from regionalign import config


class RegionAlignConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.results = self.project / "regionalign" / "results"
        self.results.mkdir(parents=True)
        (self.project / "outputs").mkdir()
        project_patch = patch.object(config, "PROJECT", self.project)
        results_patch = patch.object(config, "RESULTS_ROOT", self.results)
        project_patch.start()
        results_patch.start()
        self.addCleanup(project_patch.stop)
        self.addCleanup(results_patch.stop)

    def test_accepts_new_and_legacy_output_paths(self):
        self.assertEqual(config.output_path("regionalign/results/study/run"),
                         self.results / "study" / "run")
        self.assertEqual(config.output_path("outputs/untouched/run"),
                         self.project / "outputs" / "untouched" / "run")

    def test_legacy_link_resolves_to_moved_results(self):
        group = self.results / "study"
        group.mkdir()
        (self.project / "outputs" / "study").symlink_to(
            Path("..") / "regionalign" / "results" / "study", target_is_directory=True)
        self.assertEqual(config.output_path("outputs/study/run"), group / "run")

    def test_rejects_roots_traversal_and_escape_links(self):
        (self.results / "escape").symlink_to(self.project, target_is_directory=True)
        for value in (self.results, self.project / "outputs", "records/new",
                      "regionalign/results/../../records/new",
                      "regionalign/results/escape/outside"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                config.output_path(value)

    def test_fresh_output_uses_results_and_refuses_overwrite(self):
        target = config.fresh_output_dir("study", "example")
        self.assertEqual(target.parent, self.results / "study")
        self.assertTrue(target.is_dir())
        with self.assertRaises(FileExistsError):
            config.fresh_output_dir("study", "example", requested=target)

    def test_rejects_group_traversal(self):
        with self.assertRaises(ValueError):
            config.fresh_output_dir("../../outside", "example")

    def test_result_root_follows_project_without_separate_override(self):
        # setUp intentionally overrides both attributes; remove only the
        # RESULTS_ROOT override temporarily to exercise ordinary PROJECT mocks.
        overridden = config.__dict__.pop("RESULTS_ROOT")
        try:
            other = self.project / "other-project"
            with patch.object(config, "PROJECT", other):
                self.assertEqual(config.RESULTS_ROOT, other / "regionalign" / "results")
                target = config.fresh_output_dir("study", "project_patch")
                self.assertEqual(target.parent, config.RESULTS_ROOT / "study")
        finally:
            config.RESULTS_ROOT = overridden

    def test_runtime_assignments_and_patches_reach_environment_owner(self):
        original = runtime.HUB
        replacement = self.project / "cache"
        with patch.object(config, "HUB", replacement):
            self.assertEqual(runtime.HUB, replacement)
            with patch.dict(os.environ, {}, clear=True), patch.object(runtime, "EXTRA_LIBS", []), \
                    patch.object(sys, "path", list(sys.path)):
                config.setup_environment()
                self.assertEqual(os.environ["HF_HOME"], str(replacement))
        self.assertEqual(runtime.HUB, original)


if __name__ == "__main__":
    unittest.main()
