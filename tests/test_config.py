"""Output containment, offline environment and self-contained path settings."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_package_owns_its_paths(self):
        package = Path(config.__file__).resolve().parent
        self.assertEqual(config.PACKAGE, package)
        self.assertEqual(config.CACHE, package / ".cache")
        self.assertTrue(config.CONFIG_PATH.is_absolute())

    def test_accepts_only_paths_below_the_result_root(self):
        self.assertEqual(config.output_path("regionalign/results/study/run"),
                         self.results / "study" / "run")
        self.assertEqual(config.output_path(self.results / "study" / "run"),
                         self.results / "study" / "run")
        with self.assertRaises(ValueError):
            config.output_path("outputs/untouched/run")

    def test_rejects_roots_traversal_and_escape_links(self):
        (self.results / "escape").symlink_to(self.project, target_is_directory=True)
        (self.project / "outputs" / "study").symlink_to(
            Path("..") / "regionalign" / "results", target_is_directory=True)
        for value in (self.results, self.project / "outputs", "records/new",
                      "regionalign/results/../../records/new",
                      "regionalign/results/escape/outside", "outputs/study"):
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

    def test_runtime_json_must_be_a_file_inside_the_package_with_only_hf_home(self):
        package = self.project / "package"
        package.mkdir()
        outside = self.project / "outside.json"
        outside.write_text('{"hf_home": "/elsewhere"}')
        with patch.object(config, "PACKAGE", package):
            path = package / "runtime.json"
            self.assertEqual(config._load_config(path), {})
            path.write_text('{"hf_home": "cache"}')
            self.assertEqual(config._load_config(path), {"hf_home": "cache"})
            self.assertEqual(config._path("cache"), package / "cache")
            for text in ('{"hf_home": "cache", "extra_libs": ["/elsewhere"]}', '["cache"]', '{"hf_home": 3}'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    config._load_config(path)
            path.unlink()
            path.symlink_to(outside)
            with self.assertRaises(ValueError):
                config._load_config(path)
            with self.assertRaises(ValueError):
                config._load_config(outside)
        for name in ("EXTRA_LIBS", "REGIONALIGN_CONFIG"):
            self.assertFalse(hasattr(config, name), name)

    def test_input_audit_uses_the_configured_cache_unless_overridden(self):
        from types import SimpleNamespace
        from regionalign.experiments import input_audit
        configured, explicit = self.project / "configured", self.project / "explicit"
        with patch.object(config, "HUB", configured), \
                patch.dict(os.environ, {"HF_HOME": str(self.project / "ambient")}):
            self.assertEqual(input_audit.cache_root(SimpleNamespace(hf_home=None)), configured)
            self.assertEqual(input_audit.cache_root(SimpleNamespace(hf_home=str(explicit))), explicit)

    def test_environment_is_offline_and_stays_inside_the_package(self):
        replacement = self.project / "cache"
        before = list(sys.path)
        with patch.object(config, "HUB", replacement), \
                patch.dict(os.environ, {"REGIONALIGN_EXTRA_LIBS": str(self.project)}, clear=True), \
                patch.object(sys, "path", list(sys.path)):
            config.setup_environment()
            self.assertEqual(sys.path, before)                 # no library path is ever injected
            self.assertEqual(os.environ["HF_HOME"], str(replacement))
            self.assertEqual(os.environ["HF_HUB_CACHE"], str(replacement / "hub"))
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_DATASETS_OFFLINE"], "1")
            for key in ("XDG_CACHE_HOME", "TORCH_HOME", "MPLCONFIGDIR"):
                self.assertIn(config.CACHE, (Path(os.environ[key]), *Path(os.environ[key]).parents), key)


if __name__ == "__main__":
    unittest.main()
