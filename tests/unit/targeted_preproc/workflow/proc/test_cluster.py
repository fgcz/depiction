"""Covers the two clustering rules reachable from `PipelineArtifact.DEBUG`, through to the PNG.

Both scripts ended with `MultiChannelImage(cluster_data.unstack("i"))`, and `is_foreground` has
been a required argument for long enough that neither could ever have run: requesting `DEBUG`
died with `TypeError: MultiChannelImage.__init__() missing 1 required positional argument`, after
calibration and image generation had already been paid for. There was no test on either script.

It is tempting to read the clustering surface as one problem. It is not: these two scripts call
sklearn and hdbscan directly and never touch `depiction.tools.clustering` or
`clustering/maxmin_sampling.py`, so they are repairable without reviving anything of doubtful
correctness. That is why they are tested here while the clustering *tools* stay broken and
unsupported -- see the repository README, which says so on the front page.

Nothing asserts particular labels -- neither script sets a `random_state`, so the partition
differs run to run. The assertions are on the shape of the result and on the foreground mask,
which is what the missing argument was about.
"""

import matplotlib
import numpy as np
import pytest
import xarray
from xarray import DataArray

matplotlib.use("Agg")

from depiction.image import MultiChannelImage  # noqa: E402
from depiction_targeted_preproc.workflow.proc.cluster_kmeans import cluster_kmeans  # noqa: E402
from depiction_targeted_preproc.workflow.vis.clustering import vis_clustering  # noqa: E402

N_Y, N_X, N_C = 16, 20, 12

#: Both scripts reduce to at most 50 features via NMF with 5 components, so the image needs more
#: channels than that has components, and enough pixels for `BisectingKMeans(n_clusters=7)`.
assert N_C > 5 and N_Y * N_X > 7


def _image(*, background: bool) -> MultiChannelImage:
    """A synthetic image, optionally with a background region the cluster image must reproduce."""
    rng = np.random.default_rng(0)
    data = DataArray(
        rng.random((N_Y, N_X, N_C)),
        dims=("y", "x", "c"),
        coords={"y": np.arange(N_Y), "x": np.arange(N_X), "c": [f"marker_{i}" for i in range(N_C)]},
    )
    mask = np.ones((N_Y, N_X), dtype=bool)
    if background:
        # a corner block, so the foreground is neither everything nor a full rectangle
        mask[:4, :5] = False
    is_foreground = DataArray(mask, dims=("y", "x"), coords={"y": np.arange(N_Y), "x": np.arange(N_X)})
    return MultiChannelImage(data, is_foreground=is_foreground)


@pytest.fixture
def input_hdf5(tmp_path, request):
    """`images_default.hdf5` as `proc_cluster_kmeans` receives it from the workflow."""
    background = getattr(request, "param", False)
    path = tmp_path / "images_default.hdf5"
    _image(background=background).write_hdf5(path)
    return path


def _assert_is_a_cluster_image(path, source: MultiChannelImage) -> None:
    result = MultiChannelImage.read_hdf5(path)
    assert result.channel_names == ["cluster"]
    np.testing.assert_array_equal(source.data_spatial.coords["y"], result.data_spatial.coords["y"])
    np.testing.assert_array_equal(source.data_spatial.coords["x"], result.data_spatial.coords["x"])
    # one label per acquired pixel, and the background left as background rather than clustered
    np.testing.assert_array_equal(source.fg_mask.values, result.fg_mask.values)
    assert np.isfinite(result.data_flat.values).all()
    assert result.data_flat.sizes["i"] == int(source.fg_mask.values.sum())


class TestClusterKmeans:
    def test_writes_a_readable_cluster_image(self, input_hdf5, tmp_path) -> None:
        output = tmp_path / "cluster_default_kmeans.hdf5"
        cluster_kmeans(input_netcdf_path=input_hdf5, output_netcdf_path=output)
        _assert_is_a_cluster_image(output, _image(background=False))

    @pytest.mark.parametrize("input_hdf5", [True], indirect=True)
    def test_preserves_a_background_region(self, input_hdf5, tmp_path) -> None:
        """The case the missing `is_foreground` argument existed for."""
        output = tmp_path / "cluster_default_kmeans.hdf5"
        cluster_kmeans(input_netcdf_path=input_hdf5, output_netcdf_path=output)
        _assert_is_a_cluster_image(output, _image(background=True))

    def test_output_renders_to_a_png(self, input_hdf5, tmp_path) -> None:
        """`vis_clustering` is the rule that turns the hdf5 into the requested artifact."""
        netcdf = tmp_path / "cluster_default_kmeans.hdf5"
        png = tmp_path / "cluster_default_kmeans.png"
        cluster_kmeans(input_netcdf_path=input_hdf5, output_netcdf_path=netcdf)

        # `vis_clustering` reads the file as a plain DataArray and plots channel 0, so it depends
        # on `is_foreground` being appended after the data channels rather than before them.
        assert list(xarray.open_dataarray(netcdf).coords["c"].values) == ["cluster", "is_foreground"]

        vis_clustering(input_netcdf_path=netcdf, output_png_path=png)
        assert png.stat().st_size > 0


class TestClusterHdbscan:
    """`hdbscan` is declared only in the `dev` extra, so this skips under `nox -s tests_depiction`.

    Dropping `cluster_default_hdbscan.png` from `DEBUG` is what makes that acceptable: no
    artifact requests this script any more, and promoting `hdbscan` to a runtime dependency would
    make every clean Linux install compile it -- it publishes no manylinux wheel.
    """

    def test_writes_a_readable_cluster_image(self, input_hdf5, tmp_path) -> None:
        pytest.importorskip("hdbscan", reason="hdbscan ships only in the dev extra")
        from depiction_targeted_preproc.workflow.proc.cluster_hdbscan import cluster_dbscan

        output = tmp_path / "cluster_default_hdbscan.hdf5"
        cluster_dbscan(input_netcdf_path=input_hdf5, output_netcdf_path=output)
        _assert_is_a_cluster_image(output, _image(background=False))
