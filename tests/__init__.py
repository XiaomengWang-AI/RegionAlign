"""CPU protocol and relocation regression tests.

Pin the repository's shared test helpers before runtime setup prepends external
libraries that may also contain a top-level ``tests`` package.
"""
import importlib.util
from pathlib import Path
import sys

if "tests" not in sys.modules:
    _directory = Path(__file__).resolve().parents[2] / "tests"
    _spec = importlib.util.spec_from_file_location(
        "tests", _directory / "__init__.py", submodule_search_locations=[str(_directory)],
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["tests"] = _module
    _spec.loader.exec_module(_module)
