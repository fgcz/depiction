"""What the writer puts on disk, now that `imzy` writes it instead of `pyimzml`.

The corpus in `conftest.py` is already written through `ImzmlWriteFile`, so
`test_reader_parity.py` covers the round trip. What is left is the writer's own output
shape, the inputs it must refuse, and the one place where writing and reading meet across
a process boundary.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from depiction.parallel_ops import ParallelConfig, WriteSpectraParallel
from depiction_io import ImzmlModeEnum, ImzmlReadFile, ImzmlWriteFile, ImzyReadFile
from depiction_io.types import GenericReadFile, GenericReader, GenericWriter
from tests.differential.corpus import Case, make_spectra, write_case

_NS = "{http://psi.hupo.org/ms/mzml}"
POSITION_Z = "IMS:1000052"


def test_z_is_written_only_for_3d_input(case: Case) -> None:
    # imzy's writer appends z = 1 to every 2D coordinate; `DepictionIMZMLWriter` undoes that.
    # Without the patch every acquisition the pipeline writes would gain a z axis, and
    # because `reader.coordinates[i]` goes straight back into `add_spectrum` in several
    # tools, one round trip would make it permanent.
    declares_z = POSITION_Z in case.path.read_text()
    assert declares_z == (case.spectra.coordinates.shape[1] == 3)


def test_written_file_declares_the_requested_mode(case: Case) -> None:
    if case.compressed:
        pytest.skip("the compressed twin is a rewrite, not a fresh write")
    tree = ElementTree.parse(case.path)
    accessions = {element.get("accession") for element in tree.iter(f"{_NS}cvParam")}
    # IMS:1000030 continuous / IMS:1000031 processed.
    expected = "IMS:1000030" if case.imzml_mode == ImzmlModeEnum.CONTINUOUS else "IMS:1000031"
    assert expected in accessions


def test_empty_spectra_are_refused(tmp_path: Path) -> None:
    # imzy warns and returns False here, which would drop the pixel and leave the spectrum
    # count out of step with the coordinates the caller believes it wrote. One good spectrum
    # is written first, both because that is the shape of the real case -- `filter_peaks`
    # emitting one empty spectrum among many -- and because closing an empty writer raises
    # an unrelated error.
    write_file = ImzmlWriteFile(tmp_path / "out.imzML", imzml_mode=ImzmlModeEnum.PROCESSED)
    with pytest.raises(ValueError, match="empty spectrum"), write_file.writer() as writer:
        writer.add_spectrum(np.array([100.0, 200.0]), np.array([1.0, 2.0]), (1, 1))
        writer.add_spectrum(np.array([]), np.array([]), (2, 1))


@pytest.fixture()
def file_with_an_empty_spectrum(tmp_path: Path) -> Path:
    """A file whose second spectrum has no peaks.

    The writer refuses to produce one, but files like this exist: `filter_peaks` could emit
    an empty spectrum and the old pyimzml writer wrote it. Both backends have to read them
    the same way, so the specimen is built by editing the XML -- pointing the empty spectrum
    at a zero-length slice of the existing .ibd, which leaves the file's checksum valid.
    """
    source = write_case(
        "with_empty",
        tmp_path,
        make_spectra(n_spectra=2, n_points=[3, 4], seed=7),
        ImzmlModeEnum.PROCESSED,
    )
    path = tmp_path / "with_empty_spectrum.imzML"
    shutil.copy(source.path.with_suffix(".ibd"), path.with_suffix(".ibd"))

    ElementTree.register_namespace("", "http://psi.hupo.org/ms/mzml")
    tree = ElementTree.parse(source.path)
    spectrum_list = tree.getroot().find(f"./{_NS}run/{_NS}spectrumList")
    spectra = spectrum_list.findall(f"{_NS}spectrum")
    empty = spectra[-1]
    for binary_array in empty.findall(f"{_NS}binaryDataArrayList/{_NS}binaryDataArray"):
        for accession in ("IMS:1000103", "IMS:1000104"):
            binary_array.find(f"{_NS}cvParam[@accession='{accession}']").set("value", "0")
        binary_array.find(f"{_NS}cvParam[@accession='IMS:1000102']").set("value", "16")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


@pytest.mark.parametrize("read_file_cls", [ImzmlReadFile, ImzyReadFile])
def test_empty_spectra_read_back_as_empty_arrays(file_with_an_empty_spectrum: Path, read_file_cls: type) -> None:
    read_file = read_file_cls(file_with_an_empty_spectrum)
    assert read_file.n_spectra == 2
    with read_file.reader() as reader:
        assert len(reader.get_spectrum_mz(0)) == 3
        assert len(reader.get_spectrum_mz(1)) == 0
        assert len(reader.get_spectrum_int(1)) == 0


def _copy_chunk(reader: GenericReader, spectra_indices: list[int], writers: list[GenericWriter]) -> None:
    writers[0].copy_spectra(reader=reader, spectra_indices=spectra_indices)


def test_write_spectra_parallel_round_trip(case: Case, tmp_path: Path) -> None:
    """Chunks written by the new writer, merged back by the legacy reader.

    This is the sharp edge of the writer flip: `WriteSpectraParallel` writes one file per
    chunk in a worker process and `MergeImzml` reads them all back in the parent, so any
    disagreement between the two halves shows up as lost or reordered spectra rather than
    as an exception.
    """
    if case.compressed:
        pytest.skip("the imzy backend refuses compressed input, and the legacy path is covered by the twin")
    read_file: GenericReadFile = ImzmlReadFile(case.path)
    output = tmp_path / "merged.imzML"
    parallel = WriteSpectraParallel.from_config(ParallelConfig(n_jobs=2, task_size=2))
    parallel.map_chunked_to_files(
        read_file=read_file,
        write_files=[ImzmlWriteFile(output, imzml_mode=case.imzml_mode)],
        operation=_copy_chunk,
    )

    merged = ImzmlReadFile(output)
    assert merged.n_spectra == case.spectra.n_spectra
    np.testing.assert_array_equal(read_file.coordinates, merged.coordinates)
    # The chunk files that `WriteSpectraParallel` writes are opened with `ImzmlWriteFile`'s
    # default dtypes -- it does not propagate the input's -- so a float64 intensity array
    # comes back as float32. That is a pre-existing wart of the parallel machinery, not of
    # the writer, and the expectations follow it rather than hide it.
    with merged.reader() as reader:
        for i, (mz, intensity) in enumerate(zip(case.expected_mz, case.expected_int)):
            np.testing.assert_array_equal(mz.astype(np.float64), reader.get_spectrum_mz(i), err_msg=f"mz {i}")
            np.testing.assert_array_equal(intensity.astype(np.float32), reader.get_spectrum_int(i), err_msg=f"int {i}")
