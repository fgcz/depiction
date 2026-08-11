import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from depiction_io import GenericReadFile, ImzmlWriteFile
from pytest_mock import MockerFixture

from depiction.tools.create_imzml_pool import CreateImzmlPool


@pytest.fixture()
def source_files(mocker: MockerFixture, tmp_path) -> list[MagicMock]:
    """Three sources with distinct paths, which is what the pool's lookup keys on.

    `RamReadFile` is not usable here: its `imzml_file` is `Path("/dev/null")` for every
    instance, so all three would collide on `abs_path`.
    """
    files = []
    for i, n_spectra in enumerate([10, 20, 30]):
        file = mocker.MagicMock(name=f"source_{i}", spec=GenericReadFile)
        file.imzml_file = tmp_path / f"source_{i}.imzML"
        file.n_spectra = n_spectra
        files.append(file)
    return files


@pytest.fixture()
def write_file(mocker: MockerFixture, tmp_path) -> MagicMock:
    file = mocker.MagicMock(name="write_file", spec=ImzmlWriteFile)
    file.imzml_file = tmp_path / "pool" / "pool.imzML"
    return file


def test_write_pool(source_files, write_file) -> None:
    pool = CreateImzmlPool(source_files=source_files, n_spectra_per_file=4)

    pool.write_pool(write_file)

    writer = write_file.writer.return_value.__enter__.return_value
    assert writer.copy_spectra.call_count == 3
    for call, source in zip(writer.copy_spectra.call_args_list, source_files):
        reader, spectrum_ids = call.args
        assert reader is source.reader.return_value.__enter__.return_value
        # every id is one this source actually sampled, and each is in range for it
        assert len(spectrum_ids) == 4
        assert set(spectrum_ids) <= set(range(source.n_spectra))


def test_write_pool_when_ids_differ_per_file(source_files, write_file) -> None:
    """Each source must receive its own ids, not the first source's.

    A lookup that silently matches nothing -- or matches the wrong row -- is the failure
    mode this guards; with 4 of 10 vs 4 of 30 spectra the samples do not coincide.
    """
    pool = CreateImzmlPool(source_files=source_files, n_spectra_per_file=4)

    pool.write_pool(write_file)

    writer = write_file.writer.return_value.__enter__.return_value
    passed = [call.args[1] for call in writer.copy_spectra.call_args_list]
    expected = list(pool.pool_source_df["source_spectrum_id"])
    assert [list(ids) for ids in passed] == [list(ids) for ids in expected]


def test_write_pool_writes_metadata(source_files, write_file, tmp_path) -> None:
    pool = CreateImzmlPool(source_files=source_files, n_spectra_per_file=4)

    pool.write_pool(write_file)

    source = json.loads((tmp_path / "pool" / "pool_source.json").read_text())
    content = json.loads((tmp_path / "pool" / "pool_content.json").read_text())
    assert [Path(path).name for path in source["rel_path"].values()] == [
        "source_0.imzML",
        "source_1.imzML",
        "source_2.imzML",
    ]
    assert len(content["pool_spectrum_id"]) == 12


def test_write_pool_when_too_few_spectra(source_files, write_file) -> None:
    pool = CreateImzmlPool(source_files=source_files, n_spectra_per_file=11)

    with pytest.raises(ValueError, match="has only 10 spectra"):
        pool.write_pool(write_file)


def test_pool_source_df(source_files) -> None:
    pool = CreateImzmlPool(source_files=source_files, n_spectra_per_file=4, random_seed=42)

    df = pool.pool_source_df

    assert list(df["file_id"]) == [0, 1, 2]
    assert list(df["n_spectra"]) == [10, 20, 30]
    for indices in df["source_spectrum_id"]:
        assert len(set(indices)) == 4
        assert np.all(np.diff(indices) > 0)


if __name__ == "__main__":
    pytest.main()
