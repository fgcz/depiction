"""Adapter exposing `imzy <https://github.com/vandeplaslab/imzy>`_ behind the protocols in
``depiction_io.types``.

The adapter exists so that the hand-rolled imzML parser can eventually be deleted (see
``docs/refactoring/ROADMAP.md``). Until then imzy has gaps that this package works around;
every workaround carries an ``# upstream:`` marker naming the gap.
"""

from depiction_io.imzy_backend.compression_guard import UnsupportedCompressionError, raise_if_unsupported_compression

__all__ = ["UnsupportedCompressionError", "raise_if_unsupported_compression"]
