"""Use the shared layout/quantization primitives without creating a second state.

The module alias preserves the original harness object, including task/format
settings and test mocks. RegionAlign-specific policy lives in ``region``.
"""
import sys

from alignment_attack import harness as _implementation

sys.modules[__name__] = _implementation
