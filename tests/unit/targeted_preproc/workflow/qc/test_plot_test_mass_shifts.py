"""Covers the calibration mass-shift QC chain end to end on a non-square image.

Issue #50: the predecessor of this chain built its per-pixel coordinates by stacking the two
axis coordinate vectors, so it raised on any acquisition that was not square -- and produced a
silently wrong image on the ones that were. Nothing caught it for over a year because the
artifact was in no `requested_artifacts` list and had no test, so the shape assertions below are
the point of this module, not incidental.
"""

import matplotlib
import numpy as np
import polars as pl
import pytest
import xarray as xr
import yaml

matplotlib.use("Agg")

from depiction.image import MultiChannelImage  # noqa: E402
from depiction_targeted_preproc.workflow.qc.plot_test_mass_shifts import qc_plot_test_mass_shifts  # noqa: E402
from depiction_targeted_preproc.workflow.vis.test_mass_shifts import vis_test_mass_shifts  # noqa: E402

#: Deliberately not square, and in the shape of the `mouse_kidney` fixture named in issue #50.
N_Y, N_X = 51, 31

#: The shift `ConstantGlobalShift` will recover, in m/z.
GLOBAL_SHIFT = 0.05


@pytest.fixture
def calib_data_path(tmp_path):
    """A `model_coefs` group as `CalibrateImage` writes it: dense (y, x, c) plus an alpha channel."""
    coords = {"y": np.arange(N_Y), "x": np.arange(N_X)}
    coefs = xr.DataArray(np.full((N_Y, N_X, 1), GLOBAL_SHIFT), dims=("y", "x", "c"), coords={**coords, "c": ["0"]})
    is_foreground = xr.DataArray(np.ones((N_Y, N_X), dtype=bool), dims=("y", "x"), coords=coords)

    path = tmp_path / "calib_data.hdf5"
    MultiChannelImage(coefs, is_foreground=is_foreground).write_hdf5(path, group="model_coefs")
    return path


@pytest.fixture
def panel_path(tmp_path):
    path = tmp_path / "panel.csv"
    pl.DataFrame({"mass": [900.0, 1200.0, 1850.0], "label": ["a", "b", "c"]}).write_csv(path)
    return path


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "proc_calibrate.yml"
    path.write_text(yaml.safe_dump({"method": {"calibration_method": "ConstantGlobalShift"}, "n_jobs": 1}))
    return path


@pytest.fixture
def mass_shifts_path(tmp_path, calib_data_path, panel_path, config_path):
    path = tmp_path / "test_mass_shifts.hdf5"
    vis_test_mass_shifts(
        calib_hdf5_path=calib_data_path, mass_list_path=panel_path, config_path=config_path, output_hdf5_path=path
    )
    return path


def test_vis_test_mass_shifts_probes_the_panel_range(mass_shifts_path) -> None:
    image = MultiChannelImage.read_hdf5(mass_shifts_path)

    assert image.channel_names == ["900.00", "1375.00", "1850.00"]


def test_vis_test_mass_shifts_keeps_the_image_orientation(mass_shifts_path) -> None:
    """A transposed result also round-trips, so assert the axes individually rather than the area."""
    image = MultiChannelImage.read_hdf5(mass_shifts_path)

    assert image.data_spatial.sizes["y"] == N_Y
    assert image.data_spatial.sizes["x"] == N_X


def test_vis_test_mass_shifts_recovers_the_applied_shift(mass_shifts_path) -> None:
    image = MultiChannelImage.read_hdf5(mass_shifts_path)

    assert image.data_spatial.values == pytest.approx(GLOBAL_SHIFT)


def test_vis_test_mass_shifts_deduplicates_a_degenerate_panel(tmp_path, calib_data_path, config_path) -> None:
    """Masses closer together than the channel-name precision must not become duplicate names.

    They write out fine and only raise when the QC rule reads them back, which would leave a
    stale-looking artifact behind rather than failing the step that produced it.
    """
    panel_path = tmp_path / "narrow_panel.csv"
    pl.DataFrame({"mass": [800.0, 800.005], "label": ["a", "b"]}).write_csv(panel_path)
    output_path = tmp_path / "narrow.hdf5"

    vis_test_mass_shifts(
        calib_hdf5_path=calib_data_path,
        mass_list_path=panel_path,
        config_path=config_path,
        output_hdf5_path=output_path,
    )

    # collapsing to fewer rows is fine; colliding names are not, so assert uniqueness rather than a
    # particular count, which depends on how the panel's span rounds
    channel_names = MultiChannelImage.read_hdf5(output_path).channel_names
    assert len(set(channel_names)) == len(channel_names)


@pytest.mark.parametrize("n_test_masses", [1, 3])
def test_qc_plot_test_mass_shifts_renders_every_test_mass(
    tmp_path, calib_data_path, panel_path, config_path, n_test_masses
) -> None:
    """`n=1` exercises the `squeeze=False` axes indexing, which a 1-D `axs` would break."""
    shifts_path = tmp_path / "shifts.hdf5"
    vis_test_mass_shifts(
        calib_hdf5_path=calib_data_path,
        mass_list_path=panel_path,
        config_path=config_path,
        output_hdf5_path=shifts_path,
        n_test_masses=n_test_masses,
    )
    output_pdf = tmp_path / "plot.pdf"

    qc_plot_test_mass_shifts(input_mass_shifts=shifts_path, output_pdf=output_pdf)

    assert output_pdf.stat().st_size > 0
