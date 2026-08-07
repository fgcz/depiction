"""The imzML scan, checked against the corpus.

The scan is what makes compressed files readable at all: imzy has the offsets but not the
encoded lengths, so everything downstream of `scan_imzml` is only as correct as the map it
produces. The corpus's zlib twins are rewritten from their uncompressed originals, so the
expected map is knowable exactly.

The z half is what keeps `coordinates` the right shape; the corpus contains one 3D case for
exactly that reason.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import pytest

from depiction_io.imzy_backend import UnsupportedCompressionError, raise_if_unsupported_compression, scan_imzml
from tests.differential.corpus import IBD_HEADER_SIZE, Case


def test_detects_compression_exactly_on_the_compressed_cases(case: Case) -> None:
    scan = scan_imzml(case.path)
    assert scan.unsupported_compression is None
    # `encoded_lengths` doubles as the "is this zlib" flag, so its presence is the assertion.
    assert (scan.encoded_lengths is not None) == case.compressed


def test_detects_z_exactly_on_the_3d_cases(case: Case) -> None:
    assert scan_imzml(case.path).declares_z == (case.spectra.coordinates.shape[1] == 3)


def test_encoded_lengths_locate_decompressible_blocks(case: Case) -> None:
    """Every entry in the map must actually decompress out of the .ibd.

    This is the property the reader relies on, checked without the reader: seek to the
    offset, read that many bytes, inflate. A map that is off by a block would still look
    plausible in isolation.
    """
    if not case.compressed:
        pytest.skip("only compressed cases carry encoded lengths")
    scan = scan_imzml(case.path)
    ibd = case.path.with_suffix(".ibd").read_bytes()
    assert scan.encoded_lengths
    for offset, length in scan.encoded_lengths.items():
        assert offset >= IBD_HEADER_SIZE, "a block must not overlap the UUID header"
        zlib.decompress(ibd[offset : offset + length])


def test_continuous_files_collapse_the_shared_mz_block(case: Case) -> None:
    # Keying the map on the offset means the m/z block every spectrum points at is stored
    # once. If that ever stops being true the map silently grows by a factor of n_spectra.
    if not case.compressed or case.name != "continuous_f64_f32_zlib":
        pytest.skip("needs the compressed continuous case")
    scan = scan_imzml(case.path)
    assert scan.encoded_lengths is not None
    assert len(scan.encoded_lengths) == case.spectra.n_spectra + 1


def test_raise_if_unsupported_compression_passes_zlib(case: Case) -> None:
    # zlib is read, not refused -- including for the compressed cases.
    raise_if_unsupported_compression(case.path)


def test_numpress_is_refused(case: Case, tmp_path: Path) -> None:
    """A compression that cannot be undone by inflating a block must still be refused.

    Numpress needs a codec rather than a length, so it fails the way zlib used to: as noise
    rather than as an error. No numpress payload is needed to pin that down -- the point is
    that the scan stops before anything is read.
    """
    path = tmp_path / "numpress.imzML"
    text = case.path.read_text()
    accession = "MS:1000574" if case.compressed else "MS:1000576"
    assert accession in text
    path.write_text(text.replace(accession, "MS:1002312"))

    with pytest.raises(UnsupportedCompressionError, match="MS:1002312"):
        raise_if_unsupported_compression(path)


def test_late_declared_compression_is_refused(tmp_path: Path) -> None:
    """Compression that only appears after uncompressed arrays cannot be served.

    The scan starts collecting encoded lengths when it first sees zlib, which is correct for
    every file that declares compression on a referenceableParamGroup -- the schema puts
    those before the run. A file that turns compression on mid-spectrumList would leave the
    earlier blocks without lengths, so it is rejected rather than half-mapped.
    """
    path = tmp_path / "late.imzML"
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<mzML xmlns="http://psi.hupo.org/ms/mzml"><run><spectrumList count="1"><spectrum index="0">'
        "<binaryDataArrayList>"
        '<binaryDataArray><cvParam accession="MS:1000576" name="no compression" />'
        '<cvParam accession="IMS:1000102" value="16" /><cvParam accession="IMS:1000104" value="8" /></binaryDataArray>'
        '<binaryDataArray><cvParam accession="MS:1000574" name="zlib compression" />'
        '<cvParam accession="IMS:1000102" value="24" /><cvParam accession="IMS:1000104" value="8" /></binaryDataArray>'
        "</binaryDataArrayList></spectrum></spectrumList></run></mzML>"
    )
    with pytest.raises(UnsupportedCompressionError, match="only after 1 uncompressed"):
        scan_imzml(path)


def test_error_is_a_notimplementederror() -> None:
    # Callers outside this package are expected to catch `NotImplementedError`; keep the
    # subclassing relationship pinned so narrowing it later is a deliberate act.
    assert issubclass(UnsupportedCompressionError, NotImplementedError)


def test_scan_survives_a_missing_spectrum_list(tmp_path: Path) -> None:
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
    assert scan.unsupported_compression is None
    assert scan.encoded_lengths is None
    assert not scan.declares_z
