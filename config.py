"""RegionAlign output roots with the repository's existing offline settings.

Runtime paths and environment setup stay owned by ``runtime_config``.  This
module delegates those attributes, including assignments used by callers and
test patches, while keeping RegionAlign outputs in its own result tree.  Old
``outputs/`` paths remain accepted after resolving compatibility symlinks.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from uuid import uuid4

from alignment_attack import runtime_config as _runtime


PROJECT = _runtime.PROJECT
_RUNTIME_NAMES = frozenset(vars(_runtime))


def results_root():
    """Return the result root, following PROJECT patches used by CPU tests."""
    return getattr(sys.modules[__name__], "RESULTS_ROOT")


def output_path(path):
    """Resolve an output beneath the RegionAlign or legacy output root."""
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = PROJECT / target
    target = target.resolve()
    roots = (results_root().resolve(), (PROJECT / "outputs").resolve())
    if not any(target != root and root in target.parents for root in roots):
        raise ValueError(f"New outputs must be inside {roots[0]} or {roots[1]}: {target}")
    return target


def fresh_output_dir(group, label, requested=None):
    """Allocate a new result directory and refuse existing destinations."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    target = output_path(requested) if requested else output_path(
        results_root() / group / f"{stamp}_{label}_{uuid4().hex[:8]}")
    target.mkdir(parents=True, exist_ok=False)
    return target


class _RuntimeProxy(ModuleType):
    """Keep delegated runtime attributes patchable without copying globals."""

    def __getattr__(self, name):
        if name == "RESULTS_ROOT":
            return PROJECT / "regionalign" / "results"
        return getattr(_runtime, name)

    def __setattr__(self, name, value):
        if name not in self.__dict__ and name in _RUNTIME_NAMES:
            setattr(_runtime, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name not in self.__dict__ and name in _RUNTIME_NAMES:
            delattr(_runtime, name)
        else:
            super().__delattr__(name)


sys.modules[__name__].__class__ = _RuntimeProxy
