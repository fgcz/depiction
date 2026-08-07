from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from depiction_io import get_read_file
from tests.real_data.datasets import CACHE_DIR_ENV_VAR, DATASETS, DATASETS_BY_NAME, PublicDataset, cache_dir

if TYPE_CHECKING:
    from depiction_io.types import GenericReadFile


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrises over dataset *names*.

    Session scope, because `read_file` below is session-scoped and pytest will not let a
    session fixture depend on a narrower parametrisation. Names rather than objects, for
    readable test ids.
    """
    if "dataset_name" in metafunc.fixturenames:
        metafunc.parametrize("dataset_name", [dataset.name for dataset in DATASETS], scope="session")


@pytest.fixture(scope="session")
def dataset(dataset_name: str) -> PublicDataset:
    """The requested dataset, skipping the test when it has not been downloaded.

    Skip, never fail: these files are 1.24 GB together and are deliberately not in the
    repository, so their absence is the normal case in CI and on a fresh clone.
    """
    dataset = DATASETS_BY_NAME[dataset_name]
    if not dataset.is_available:
        missing = [file.filename for file in dataset.files if not file.local_path.is_file()]
        wrong_size = [
            file.filename
            for file in dataset.files
            if file.local_path.is_file() and file.local_path.stat().st_size != file.size_bytes
        ]
        detail = f"missing {', '.join(missing)}" if missing else f"wrong size: {', '.join(wrong_size)}"
        pytest.skip(
            f"{dataset.name} not in {cache_dir()} ({detail}). Run "
            f"`uv run python -m tests.real_data.fetch {dataset.name}`, or point "
            f"{CACHE_DIR_ENV_VAR} at an existing copy."
        )
    return dataset


@pytest.fixture(scope="session")
def read_file(dataset: PublicDataset) -> GenericReadFile:
    """Opened once per session: the first read parses an 8.9 MB imzML for the larger file."""
    return get_read_file(dataset.imzml.local_path)


@pytest.fixture(scope="session")
def sample_indices(dataset: PublicDataset) -> list[int]:
    """A fixed spread of spectra to read.

    Fixed rather than random so a failure is reproducible, and a handful rather than all of
    them because reading the whole rapifleX acquisition is 1.1 GB of I/O that buys no
    coverage the first, last and a few interior spectra do not already give.
    """
    n_spectra = dataset.expected.n_spectra
    return sorted({0, 1, n_spectra // 3, n_spectra // 2, n_spectra - 1})
