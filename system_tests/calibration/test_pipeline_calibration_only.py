"""End-to-end `depiction_targeted_preproc` runs, checked against their own inputs.

Every expectation here is derived from what the pipeline was handed -- the imzML's
coordinate list, the standardized panel, the metadata the pipeline itself exported -- so
that swapping the acquisition is a change to `system_tests/fixtures.py` and nothing else.
The previous version of this file asserted `{"x": 128, "y": 137, "c": 118}` and
`n_nonzero == 10131`, four constants copied from one run on one non-redistributable
acquisition. They are all still checked; they are just no longer written down.

That matters beyond tidiness: constants copied from an output cannot distinguish a correct
pipeline from a pipeline that reproduces its own past mistake, and they made the public
fixture impossible to add without re-deriving them by hand.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from depiction.image.multi_channel_image import MultiChannelImage
from depiction.image.ome_tiff import OmeTiff
from depiction_io import GenericReadFile, get_read_file
from depiction_io.imzml.metadata import Metadata
from depiction_targeted_preproc.app_interface.process_chunk import process_chunk
from depiction_targeted_preproc.pipeline.prepare_params import Params
from depiction_targeted_preproc.pipeline_config.artifacts_mapping import get_result_files_new
from system_tests.fixtures import FIXTURES, FIXTURES_BY_NAME, PipelineFixture

_CONFIG_DIR = Path(__file__).parent / "configs"

#: `position z`. Absent from both fixtures, and imzy's writer emits it unconditionally, so
#: whether it reappears in the output is a real question -- see `test_calibration_...` below.
_POSITION_Z = b'accession="IMS:1000052"'


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrises over fixture *names*, at session scope.

    Session scope because `work_dir` below is session-scoped -- a full pipeline run is
    minutes on the tonsil -- and pytest will not let a session fixture depend on a narrower
    parametrisation.
    """
    if "fixture_name" in metafunc.fixturenames:
        metafunc.parametrize("fixture_name", [fixture.name for fixture in FIXTURES], scope="session")


@pytest.fixture(scope="session")
def pipeline_fixture(fixture_name: str) -> PipelineFixture:
    """The requested acquisition, skipping when it is not on this machine.

    Skip, never fail: the tonsil is 1.26 GB and not redistributable, and the public pair is
    deliberately not in the repository, so absence is the normal case rather than an error.
    """
    fixture = FIXTURES_BY_NAME[fixture_name]
    if fixture.missing:
        pytest.skip(fixture.skip_reason)
    return fixture


@pytest.fixture(scope="session")
def work_dir(pipeline_fixture: PipelineFixture, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A staged chunk directory with the pipeline already run in it."""
    directory = tmp_path_factory.mktemp(pipeline_fixture.name) / "work"
    (directory / "panels").mkdir(parents=True)
    shutil.copy(pipeline_fixture.imzml_path, directory / "raw.imzML")
    shutil.copy(pipeline_fixture.ibd_path, directory / "raw.ibd")
    shutil.copy(pipeline_fixture.panel_path, directory / "panels" / "unstandardized_full.csv")
    shutil.copy(_CONFIG_DIR / "params.yml", directory / "params.yml")
    shutil.copy(_CONFIG_DIR / "global_constant_shift.yml", directory / "pipeline_params.yml")
    process_chunk(chunk_dir=directory)
    return directory


@pytest.fixture(scope="session")
def acquisition(work_dir: Path) -> GenericReadFile:
    """The input the pipeline was given -- the source of truth for every geometric claim."""
    return get_read_file(work_dir / "raw.imzML")


@pytest.fixture(scope="session")
def acquired_coordinates(acquisition: GenericReadFile) -> np.ndarray:
    """The acquired pixel positions, as an (n, 2) array of x/y."""
    return np.asarray(acquisition.coordinates_array_2d.values)


@pytest.fixture(scope="session")
def panel(work_dir: Path) -> pl.DataFrame:
    """The panel as `vis_images` consumed it, after standardization."""
    return pl.read_csv(work_dir / "panels" / "full_visualize.csv")


@pytest.fixture(scope="session")
def image(work_dir: Path) -> MultiChannelImage:
    return OmeTiff.read_image(work_dir / "images_default.ome.tiff", bg_value=0.0)


def test_requested_artifacts_are_produced(work_dir: Path) -> None:
    """Every file `params.yml` asked for, plus the zip `process_chunk` bundles them into.

    Resolved through `get_result_files_new`, the same call `process_chunk` makes, so changing
    `requested_artifacts` changes what is checked instead of silently checking less.
    """
    params = Params.model_validate(yaml.safe_load((work_dir / "params.yml").read_text()))
    requested = get_result_files_new(requested_artifacts=params.requested_artifacts, sample_dir=work_dir)
    assert requested, "the fixture config requested no artifacts at all"
    assert [path for path in requested if not path.exists()] == []
    assert (work_dir / "outputs" / f"{work_dir.name}.zip").is_file()


def test_image_geometry_matches_the_acquisition_and_the_panel(
    image: MultiChannelImage, acquired_coordinates: np.ndarray, panel: pl.DataFrame
) -> None:
    extent = acquired_coordinates.max(axis=0) - acquired_coordinates.min(axis=0) + 1
    assert image.sizes == {"x": int(extent[0]), "y": int(extent[1]), "c": len(panel)}


def test_channel_names_are_the_panel_labels(image: MultiChannelImage, panel: pl.DataFrame) -> None:
    assert image.channel_names == panel["label"].to_list()


def test_foreground_matches_the_acquired_pixels(
    image: MultiChannelImage, acquired_coordinates: np.ndarray, pipeline_fixture: PipelineFixture
) -> None:
    """The foreground mask against the acquisition's coordinate list.

    This is what the old `n_nonzero == 10131` was standing in for, and it is strictly
    stronger: a count is satisfied by the right number of *wrong* pixels, a transposed or
    shifted image among them.

    The OME-TIFF round trip drops the x/y coordinate labels -- `OmeTiff.read` assigns only
    `c` -- so the mask comes back indexed from zero and the acquisition's origin has to be
    added back before the two can be compared.
    """
    rows, columns = np.nonzero(image.fg_mask.transpose("y", "x").values)
    origin = acquired_coordinates.min(axis=0)
    foreground = set(zip((columns + origin[0]).tolist(), (rows + origin[1]).tolist(), strict=True))
    acquired = set(map(tuple, acquired_coordinates.tolist()))

    assert foreground <= acquired
    if pipeline_fixture.expect_every_pixel_has_signal:
        assert foreground == acquired


def test_image_values_are_finite(image: MultiChannelImage) -> None:
    """No NaNs anywhere in the output.

    `CalibrationMethodConstantGlobalShift` derives one shift for the whole image with
    `np.nanmedian` over per-reference peak distances. A panel whose masses find no peaks
    makes that NaN, and every m/z in the file follows -- a failure that otherwise surfaces
    only as an image that is quietly all background.
    """
    assert np.isfinite(image.data_spatial.values).all()


def test_pixel_size_matches_the_exported_raw_metadata(work_dir: Path) -> None:
    """The OME-TIFF's physical pixel size against what `proc_export_raw_metadata` wrote.

    For the mouse kidney this pins a fallback nothing else exercises: the file declares no
    `IMS:1000046`, `Metadata.pixel_size` is not optional, so the export step takes its
    `ValidationError` branch and substitutes a dummy 1 um. See ROADMAP, "Known, still
    unfixed" -- this asserts the current behaviour rather than endorsing it.
    """
    expected = Metadata.model_validate(json.loads((work_dir / "raw_metadata.json").read_text())).pixel_size
    written = OmeTiff.read(work_dir / "images_default.ome.tiff").attrs["pixel_size"]
    assert (written.size_x, written.size_y) == (expected.size_x, expected.size_y)


def test_calibration_preserves_the_pixel_grid(work_dir: Path, acquisition: GenericReadFile) -> None:
    """Calibration is a per-spectrum m/z transform, so the pixel grid must survive it.

    imzy's writer drops empty spectra with a warning rather than an error, which would
    desynchronise the count from the coordinates. The adapter refuses instead; this is that
    guard on a real acquisition rather than on the synthetic corpus.
    """
    calibrated = get_read_file(work_dir / "calibrated.imzML")
    assert calibrated.n_spectra == acquisition.n_spectra
    assert np.array_equal(calibrated.coordinates, acquisition.coordinates)


def test_calibration_does_not_invent_a_z_axis(work_dir: Path) -> None:
    """A 2D acquisition must stay 2D through a pipeline round trip.

    imzy's writer emits `IMS:1000052` unconditionally, and `reader.coordinates[i]` is handed
    straight back to `writer.add_spectrum` in five tools -- so one round trip through a
    pipeline is exactly what would have made an invented z column permanent.
    `DepictionIMZMLWriter` strips it again; this checks the output XML, not the reader that
    would have to agree with it.
    """
    declares_z = _POSITION_Z in (work_dir / "raw.imzML").read_bytes()
    assert (_POSITION_Z in (work_dir / "calibrated.imzML").read_bytes()) == declares_z
