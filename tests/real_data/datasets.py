"""The public imzML acquisitions this repository tests against, and what they must read as.

Two redistributable imzML/`.ibd` pairs, identified and measured in
[`test-data.md`](../../docs/test-data.md). Together they are 1.24 GB, so they are not in the
repository: `fetch.py` downloads them into a gitignored cache and the tests skip when it is
empty.

`ExpectedReading` is the part that earns its keep. Those numbers were measured on
2026-08-07 through *both* the hand-rolled imzML parser and imzy, which agreed on all of
them; the parser was deleted shortly afterwards. Pinning them here is what survives of that
comparison. It is a different kind of evidence from `tests/differential/`,
which checks imzy against files imzy's own writer produced -- these are third-party vendor
exports, and their expected values predate the migration that would have to be wrong for
them to be wrong.

This module is deliberately stdlib-only, so that `fetch.py` can import it and run before
anything is installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Repo-local rather than under `XDG_CACHE_HOME`, so that a 1.18 GB fixture cannot quietly
#: accumulate somewhere nobody looks; `.gitignore` covers it.
_DEFAULT_CACHE_DIR = Path(__file__).parents[2] / ".test-data"

#: Environment variable pointing the cache somewhere else -- an existing copy, or an
#: external disk. Also how the tests are made to skip: point it at an empty directory.
CACHE_DIR_ENV_VAR = "DEPICTION_TEST_DATA_DIR"


def cache_dir() -> Path:
    """The directory holding the downloaded acquisitions."""
    override = os.environ.get(CACHE_DIR_ENV_VAR)
    return Path(override) if override else _DEFAULT_CACHE_DIR


@dataclass(frozen=True)
class RemoteFile:
    """One downloadable file, pinned by size and hash so it cannot change underneath us."""

    filename: str
    url: str
    size_bytes: int
    sha256: str

    @property
    def local_path(self) -> Path:
        return cache_dir() / self.filename


@dataclass(frozen=True)
class ExpectedReading:
    """What a correct reader must report for an acquisition.

    See the module docstring for where these came from and why that matters. `mz_tolerance`
    exists because the recorded m/z bounds are rounded to the precision they were printed
    at; comparing more tightly than they were written down would be asserting on noise.
    """

    n_spectra: int
    #: `ImzmlModeEnum` member name, as a string, to keep this module import-free.
    imzml_mode: str
    #: `coordinates.max(0) - coordinates.min(0) + 1` over x and y, as in `compact_metadata`.
    coordinate_extent: tuple[int, int]
    mz_dtype: str
    int_dtype: str
    n_bins: int
    mz_min: float
    mz_max: float
    mz_tolerance: float
    #: `None` means the file is expected to declare no pixel size, not that we did not look.
    pixel_size_um: float | None


@dataclass(frozen=True)
class PublicDataset:
    name: str
    description: str
    source: str
    licence: str
    citation: str
    imzml: RemoteFile
    ibd: RemoteFile
    expected: ExpectedReading

    @property
    def files(self) -> tuple[RemoteFile, RemoteFile]:
        return (self.imzml, self.ibd)

    @property
    def total_bytes(self) -> int:
        return sum(file.size_bytes for file in self.files)

    @property
    def is_available(self) -> bool:
        """Whether both files are present at their recorded size.

        Size only -- hashing 1.18 GB to decide whether to skip a test would cost more than
        the test. `test_sha256_matches_the_deposit` does the real check, and `fetch.py`
        never renames a file into place without having verified it.
        """
        return all(
            file.local_path.is_file() and file.local_path.stat().st_size == file.size_bytes for file in self.files
        )


_PRIDE_PXD048809 = "https://ftp.pride.ebi.ac.uk/pride/data/archive/2024/09/PXD048809"
_ZENODO_1560646 = "https://zenodo.org/records/1560646/files"

RAPIFLEX_CTRLS = PublicDataset(
    name="rapiflex_ctrls",
    description="Murine tibialis anterior, metabolites and phospholipids, 20 um raster",
    source="PRIDE PXD048809",
    licence="CC0",
    citation=(
        "Spatial multi-omics in skeletal muscle unravel complex myofiber architecture. "
        "Commun Biol (2024). doi:10.1038/s42003-024-06949-1"
    ),
    imzml=RemoteFile(
        filename="20230130_rf1_msi2023001_slide001_ctrls.imzML",
        url=f"{_PRIDE_PXD048809}/20230130_rf1_msi2023001_slide001_ctrls.imzML",
        size_bytes=8_903_764,
        sha256="706a42243e95dca0d84f650840fd94d6a781c971857e96e244b10adb6ce03bb8",
    ),
    ibd=RemoteFile(
        filename="20230130_rf1_msi2023001_slide001_ctrls.ibd",
        url=f"{_PRIDE_PXD048809}/20230130_rf1_msi2023001_slide001_ctrls.ibd",
        size_bytes=1_169_811_216,
        sha256="acbccad504442682c8e5426af718e3de8f9e613992a32ccf2d80265cd2f8fe02",
    ),
    # The one fixture with SCiLS Lab / Bruker Container header conventions, which is the
    # combination the FGCZ tonsil acquisition also has and the synthetic corpus does not.
    expected=ExpectedReading(
        n_spectra=3887,
        imzml_mode="CONTINUOUS",
        coordinate_extent=(240, 668),
        mz_dtype="float64",
        int_dtype="float32",
        n_bins=75200,
        mz_min=79.985,
        mz_max=1000.008,
        mz_tolerance=5e-4,
        pixel_size_um=20.0,
    ),
)

MOUSE_KIDNEY = PublicDataset(
    name="mouse_kidney",
    description="FFPE mouse kidney tryptic peptides, cropped teaching export, 150 um raster per the deposit",
    source="Zenodo 10.5281/zenodo.1560646",
    licence="MIT",
    citation="MALDI imaging of mouse kidney peptides - test dataset. doi:10.5281/zenodo.1560646",
    imzml=RemoteFile(
        filename="mouse_kidney_cut.imzML",
        url=f"{_ZENODO_1560646}/mouse_kidney_cut.imzML?download=1",
        size_bytes=2_276_557,
        sha256="779fa4cb718cc8b19a11c9e8ddeb90e3e7fef421852ccbf7e049a8f6d61aa1dc",
    ),
    ibd=RemoteFile(
        filename="mouse_kidney_cut.ibd",
        url=f"{_ZENODO_1560646}/mouse_kidney_cut.ibd?download=1",
        size_bytes=57_034_280,
        sha256="8740034a9734a2713cce0de78b3ce03e49cba524e9614e14e9304ace45ce015b",
    ),
    # The only real fixture with **float32 m/z**. The differential corpus parametrises over
    # that; nothing else measured does.
    expected=ExpectedReading(
        n_spectra=1581,
        imzml_mode="CONTINUOUS",
        coordinate_extent=(31, 51),
        mz_dtype="float32",
        int_dtype="float32",
        n_bins=9013,
        mz_min=1220.0382,
        mz_max=1624.9987,
        mz_tolerance=5e-5,
        # None, not 150: the file's `scanSettings` carries only `max count of pixel x/y`
        # and no `IMS:1000046`. The 150 um raster is documented by the deposit, not by the
        # file, and `public-test-data.md` conflated the two until this test was written.
        pixel_size_um=None,
    ),
)

DATASETS: tuple[PublicDataset, ...] = (MOUSE_KIDNEY, RAPIFLEX_CTRLS)

DATASETS_BY_NAME: dict[str, PublicDataset] = {dataset.name: dataset for dataset in DATASETS}
