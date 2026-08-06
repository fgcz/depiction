"""The imzML scan, checked against the corpus.

The compression half is the one property that makes the imzy migration safe to abandon
half-way: as long as compressed input raises, a partially migrated repository is
inconvenient rather than quietly wrong.

The z half is what keeps the two backends' `coordinates` the same shape; the corpus
contains one 3D case for exactly that reason.
"""

from __future__ import annotations

import pytest

from depiction_io.imzy_backend import UnsupportedCompressionError, raise_if_unsupported_compression, scan_imzml
from tests.differential.corpus import Case


def test_detects_compression_exactly_on_the_compressed_cases(case: Case) -> None:
    scan = scan_imzml(case.path)
    if case.compressed:
        assert scan.compression == "MS:1000574"
    else:
        assert scan.compression is None


def test_detects_z_exactly_on_the_3d_cases(case: Case) -> None:
    if case.compressed:
        pytest.skip("the scan stops early on compressed input, so declares_z is undefined")
    assert scan_imzml(case.path).declares_z == (case.spectra.coordinates.shape[1] == 3)


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
    scan = scan_imzml(path)
    assert scan.compression is None
    assert not scan.declares_z
