"""The zlib guard, checked against the corpus that already contains compressed twins.

This is the one property that makes the imzy migration safe to abandon half-way: as long
as compressed input raises, a partially migrated repository is inconvenient rather than
quietly wrong.
"""

from __future__ import annotations

import pytest

from depiction_io.imzy_backend import UnsupportedCompressionError, raise_if_unsupported_compression
from depiction_io.imzy_backend.compression_guard import find_unsupported_compression
from tests.differential.corpus import Case


def test_detects_compression_exactly_on_the_compressed_cases(case: Case) -> None:
    accession = find_unsupported_compression(case.path)
    if case.compressed:
        assert accession == "MS:1000574"
    else:
        assert accession is None


def test_raise_if_unsupported_compression(case: Case) -> None:
    if case.compressed:
        with pytest.raises(UnsupportedCompressionError, match="MS:1000574"):
            raise_if_unsupported_compression(case.path)
    else:
        raise_if_unsupported_compression(case.path)


def test_error_is_a_notimplementederror() -> None:
    # Callers outside this package are expected to catch `NotImplementedError`; keep the
    # subclassing relationship pinned so narrowing it later is a deliberate act.
    assert issubclass(UnsupportedCompressionError, NotImplementedError)


def test_scan_survives_a_missing_spectrum_list(tmp_path) -> None:
    # `spectrumList` is the element the scan empties as it goes; a file without one must
    # still scan rather than trip over the pruning logic.
    path = tmp_path / "no_spectra.imzML"
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<mzML xmlns="http://psi.hupo.org/ms/mzml">'
        '<cvParam accession="MS:1000576" name="no compression" />'
        "</mzML>"
    )
    assert find_unsupported_compression(path) is None
