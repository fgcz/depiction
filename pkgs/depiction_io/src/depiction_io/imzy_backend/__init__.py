"""Adapter exposing `imzy <https://github.com/vandeplaslab/imzy>`_ behind the protocols in
``depiction_io.types``.

The adapter exists so that the hand-rolled imzML parser can eventually be deleted (see
``docs/refactoring/ROADMAP.md``). Until then imzy has gaps that this package works around;
every workaround carries an ``# upstream:`` marker naming the gap.
"""

from depiction_io.imzy_backend.imzml_scan import (
    ImzmlScan,
    UnsupportedCompressionError,
    raise_if_unsupported_compression,
    scan_imzml,
)
from depiction_io.imzy_backend.imzy_read_file import ImzyReadFile
from depiction_io.imzy_backend.imzy_reader import ImzyReader

__all__ = [
    "ImzmlScan",
    "ImzyReadFile",
    "ImzyReader",
    "UnsupportedCompressionError",
    "raise_if_unsupported_compression",
    "scan_imzml",
]
