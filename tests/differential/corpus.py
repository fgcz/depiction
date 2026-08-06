"""Builders for the differential-test corpus.

The corpus is generated at test time rather than committed, because ``.gitignore``
excludes ``*.imzML`` and ``*.ibd`` -- and because a generated corpus stays honest when
the writer changes.

The point of this module is to make backend swaps verifiable: every case here is read
back through whichever ``GenericReadFile`` implementations are available and the results
are required to agree. See ``test_reader_parity.py``.
"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from numpy.typing import NDArray

from depiction_io import ImzmlModeEnum, ImzmlWriteFile

MZML_NS = "http://psi.hupo.org/ms/mzml"
_NS = f"{{{MZML_NS}}}"

# cvParam accessions used when rewriting a file to be zlib-compressed.
ACC_NO_COMPRESSION = "MS:1000576"
ACC_ZLIB_COMPRESSION = "MS:1000574"
ACC_EXTERNAL_OFFSET = "IMS:1000102"
ACC_EXTERNAL_ENCODED_LENGTH = "IMS:1000104"
ACC_IBD_SHA1 = "IMS:1000091"

# The .ibd begins with a 16-byte UUID header that must survive recompression untouched.
IBD_HEADER_SIZE = 16


@dataclass(frozen=True)
class Spectra:
    """The source data for one corpus case, before it is written to disk."""

    mz: list[NDArray[np.float64]]
    int: list[NDArray[np.float64]]
    coordinates: NDArray[np.int64]

    @property
    def n_spectra(self) -> int:
        return len(self.mz)


@dataclass(frozen=True)
class Case:
    """One corpus entry: source data plus the file written from it."""

    name: str
    spectra: Spectra
    #: The mode the file was written in.
    imzml_mode: ImzmlModeEnum
    path: Path
    mz_dtype: np.typing.DTypeLike
    int_dtype: np.typing.DTypeLike
    #: The mode a reader is expected to report, which is not always the mode the file was
    #: written in -- see ``Spec.expected_read_mode``.
    expected_read_mode: ImzmlModeEnum
    #: True when the file on disk has zlib-compressed binary arrays.
    compressed: bool = False
    #: Set for a compressed case: the name of the uncompressed case it was derived from.
    derived_from: str | None = field(default=None)

    @property
    def expected_mz(self) -> list[NDArray[np.float64]]:
        """The m/z arrays as they should read back, i.e. after the writer's dtype cast."""
        return [np.asarray(arr, dtype=self.mz_dtype) for arr in self.spectra.mz]

    @property
    def expected_int(self) -> list[NDArray[np.float64]]:
        """The intensity arrays as they should read back, i.e. after the writer's dtype cast."""
        return [np.asarray(arr, dtype=self.int_dtype) for arr in self.spectra.int]


def make_spectra(
    n_spectra: int,
    n_points: int | list[int],
    n_dim: int = 2,
    shared_mz: bool = False,
    seed: int = 0,
) -> Spectra:
    """Builds deterministic synthetic spectra.

    Args:
        n_spectra: number of spectra.
        n_points: points per spectrum; a list gives a ragged (processed-mode) file.
        n_dim: 2 or 3 spatial coordinate dimensions.
        shared_mz: when True every spectrum gets the identical m/z array, as continuous
            mode requires.
        seed: seed for the intensity values.
    """
    rng = np.random.default_rng(seed)
    counts = [n_points] * n_spectra if isinstance(n_points, int) else n_points
    if len(counts) != n_spectra:
        raise ValueError(f"n_points has {len(counts)} entries but n_spectra is {n_spectra}")
    if shared_mz and len(set(counts)) != 1:
        raise ValueError("shared_mz requires every spectrum to have the same number of points")

    if shared_mz:
        shared = np.sort(rng.uniform(100.0, 1000.0, counts[0]))
        mz = [shared.copy() for _ in range(n_spectra)]
    else:
        mz = [np.sort(rng.uniform(100.0, 1000.0, n)) for n in counts]
    intensities = [rng.uniform(1.0, 1000.0, n) for n in counts]

    coords = np.array([[i % 4 + 1, i // 4 + 1] for i in range(n_spectra)], dtype=np.int64)
    if n_dim == 3:
        coords = np.hstack([coords, np.full((n_spectra, 1), 3, dtype=np.int64)])
    elif n_dim != 2:
        raise ValueError(f"n_dim must be 2 or 3, got {n_dim}")

    return Spectra(mz=mz, int=intensities, coordinates=coords)


def write_case(
    name: str,
    directory: Path,
    spectra: Spectra,
    imzml_mode: ImzmlModeEnum,
    mz_dtype: np.typing.DTypeLike = np.float64,
    int_dtype: np.typing.DTypeLike = np.float32,
    expected_read_mode: ImzmlModeEnum | None = None,
) -> Case:
    """Writes one corpus case to disk using the production writer."""
    path = directory / f"{name}.imzML"
    write_file = ImzmlWriteFile(path, imzml_mode=imzml_mode, mz_dtype=mz_dtype, intensity_dtype=int_dtype)
    with write_file.writer() as writer:
        for mz, intensity, coords in zip(spectra.mz, spectra.int, spectra.coordinates):
            writer.add_spectrum(mz, intensity, tuple(int(c) for c in coords))
    return Case(
        name=name,
        spectra=spectra,
        imzml_mode=imzml_mode,
        path=path,
        mz_dtype=mz_dtype,
        int_dtype=int_dtype,
        expected_read_mode=imzml_mode if expected_read_mode is None else expected_read_mode,
    )


def compress_case(case: Case, directory: Path) -> Case:
    """Returns a zlib-compressed twin of ``case``.

    Nothing in the toolchain can *produce* a compressed .ibd -- depiction's writer passes
    no compression argument to pyimzml, and imzy emits "no compression" unconditionally.
    So the only way to get a compressed specimen is to rewrite an uncompressed one, which
    has the pleasant side effect that the twin's expected contents are exactly the
    original's: any read discrepancy is a decompression bug and nothing else.
    """
    name = f"{case.name}_zlib"
    out_imzml = directory / f"{name}.imzML"
    out_ibd = out_imzml.with_suffix(".ibd")
    src_ibd = case.path.with_suffix(".ibd")

    ElementTree.register_namespace("", MZML_NS)
    tree = ElementTree.parse(case.path)
    root = tree.getroot()

    src_bytes = src_ibd.read_bytes()
    out_bytes = bytearray(src_bytes[:IBD_HEADER_SIZE])

    binary_arrays = root.findall(
        f"./{_NS}run/{_NS}spectrumList/{_NS}spectrum/{_NS}binaryDataArrayList/{_NS}binaryDataArray"
    )
    if not binary_arrays:
        raise ValueError(f"No binaryDataArray elements found in {case.path}")

    # In continuous mode every spectrum's m/z array points at the *same* (offset, length)
    # block, and the reader infers the mode from exactly that sharing. Compressing each
    # binaryDataArray independently would write that block N times at N offsets and
    # silently turn a continuous file into a processed one, so identical blocks are
    # emitted once and reused.
    relocated: dict[tuple[int, int], tuple[int, int]] = {}
    for binary_array in binary_arrays:
        offset_param = binary_array.find(f"{_NS}cvParam[@accession='{ACC_EXTERNAL_OFFSET}']")
        length_param = binary_array.find(f"{_NS}cvParam[@accession='{ACC_EXTERNAL_ENCODED_LENGTH}']")
        source_block = (int(offset_param.attrib["value"]), int(length_param.attrib["value"]))

        if source_block not in relocated:
            offset, length = source_block
            compressed = zlib.compress(src_bytes[offset : offset + length])
            relocated[source_block] = (len(out_bytes), len(compressed))
            out_bytes.extend(compressed)

        new_offset, new_length = relocated[source_block]
        offset_param.attrib["value"] = str(new_offset)
        length_param.attrib["value"] = str(new_length)

    # The compression flag lives on the referenceableParamGroups, not on the individual
    # arrays, which is why one rewrite here covers every spectrum.
    rewritten = 0
    for param in root.iter(f"{_NS}cvParam"):
        if param.attrib.get("accession") == ACC_NO_COMPRESSION:
            param.attrib["accession"] = ACC_ZLIB_COMPRESSION
            param.attrib["name"] = "zlib compression"
            rewritten += 1
    if rewritten == 0:
        raise ValueError(f"No {ACC_NO_COMPRESSION} cvParam found in {case.path}")

    out_ibd.write_bytes(bytes(out_bytes))

    # Recompute the ibd checksum, otherwise is_checksum_valid would report a spurious
    # failure that has nothing to do with compression.
    for param in root.iter(f"{_NS}cvParam"):
        if param.attrib.get("accession") == ACC_IBD_SHA1:
            param.attrib["value"] = hashlib.sha1(bytes(out_bytes)).hexdigest().upper()

    tree.write(out_imzml, encoding="utf-8", xml_declaration=True)

    return Case(
        name=name,
        spectra=case.spectra,
        imzml_mode=case.imzml_mode,
        path=out_imzml,
        mz_dtype=case.mz_dtype,
        int_dtype=case.int_dtype,
        expected_read_mode=case.expected_read_mode,
        compressed=True,
        derived_from=case.name,
    )


@dataclass(frozen=True)
class Spec:
    """A declarative corpus entry. ``CORPUS_SPECS`` is the single source of truth for
    both the files that get written and the ids the tests parametrise over."""

    name: str
    imzml_mode: ImzmlModeEnum
    n_spectra: int
    n_points: int | list[int]
    n_dim: int = 2
    shared_mz: bool = False
    seed: int = 0
    mz_dtype: np.typing.DTypeLike = np.float64
    int_dtype: np.typing.DTypeLike = np.float32
    #: When True, a zlib-compressed twin named ``f"{name}_zlib"`` is also built.
    compress_twin: bool = False
    #: Override when a reader cannot recover the written mode. ``ImzmlReader`` derives the
    #: mode from whether all spectra share one m/z offset, so a one-spectrum file is
    #: always reported as continuous regardless of what the imzML declares. That heuristic
    #: is worth pinning down: a backend that reads the IMS:1000030/31 cvParam instead
    #: would disagree here, and this is where that shows up.
    expected_read_mode: ImzmlModeEnum | None = None


#: The matrix: continuous vs processed, float32 vs float64 on both axes, 2D vs 3D
#: coordinates, and the degenerate shapes (one spectrum, one point per spectrum) that
#: tend to break offset arithmetic.
CORPUS_SPECS: list[Spec] = [
    Spec("continuous_f64_f32", ImzmlModeEnum.CONTINUOUS, 6, 8, shared_mz=True, compress_twin=True),
    Spec(
        "continuous_f32_f64",
        ImzmlModeEnum.CONTINUOUS,
        6,
        8,
        shared_mz=True,
        seed=1,
        mz_dtype=np.float32,
        int_dtype=np.float64,
    ),
    Spec("processed_ragged", ImzmlModeEnum.PROCESSED, 5, [3, 7, 1, 12, 4], compress_twin=True),
    Spec(
        "processed_f32_f32",
        ImzmlModeEnum.PROCESSED,
        4,
        [5, 6, 5, 9],
        seed=2,
        mz_dtype=np.float32,
        int_dtype=np.float32,
    ),
    Spec("processed_3d_coords", ImzmlModeEnum.PROCESSED, 4, [4, 5, 4, 6], n_dim=3, seed=3),
    Spec(
        "single_spectrum",
        ImzmlModeEnum.PROCESSED,
        1,
        5,
        seed=4,
        expected_read_mode=ImzmlModeEnum.CONTINUOUS,
    ),
    Spec("single_point_spectra", ImzmlModeEnum.CONTINUOUS, 3, 1, shared_mz=True, seed=5),
]

#: Every case name, uncompressed twins first. Tests parametrise over this.
CASE_NAMES: list[str] = [spec.name for spec in CORPUS_SPECS] + [
    f"{spec.name}_zlib" for spec in CORPUS_SPECS if spec.compress_twin
]


def build_corpus(directory: Path) -> list[Case]:
    """Builds every case in ``CORPUS_SPECS`` (plus compressed twins) in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    cases = [
        write_case(
            spec.name,
            directory,
            make_spectra(
                n_spectra=spec.n_spectra,
                n_points=spec.n_points,
                n_dim=spec.n_dim,
                shared_mz=spec.shared_mz,
                seed=spec.seed,
            ),
            spec.imzml_mode,
            mz_dtype=spec.mz_dtype,
            int_dtype=spec.int_dtype,
            expected_read_mode=spec.expected_read_mode,
        )
        for spec in CORPUS_SPECS
    ]
    by_name = {case.name: case for case in cases}
    cases += [compress_case(by_name[spec.name], directory) for spec in CORPUS_SPECS if spec.compress_twin]

    built = [case.name for case in cases]
    if built != CASE_NAMES:
        raise AssertionError(f"build_corpus produced {built}, but CASE_NAMES says {CASE_NAMES}")
    return cases
