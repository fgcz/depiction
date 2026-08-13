"""Adapter exposing `imzy <https://github.com/vandeplaslab/imzy>`_ behind the protocols in
``depiction_io.types``.

The adapter replaced a hand-rolled imzML parser, which has since been deleted. imzy has five
gaps that this package works around; every workaround carries an ``# upstream:`` marker naming
the gap by its number in ``docs/modules/depiction_io/imzy_backend.md``, which is also where the
kept remnants of that parser are explained.
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
