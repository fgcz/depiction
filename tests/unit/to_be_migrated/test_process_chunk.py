import yaml
import pytest
from pytest_mock import MockerFixture

from depiction_targeted_preproc.app_interface import process_chunk as process_chunk_module
from depiction_targeted_preproc.app_interface.process_chunk import process_chunk


@pytest.fixture
def chunk_dir(tmp_path):
    """A chunk directory with only the file `process_chunk` reads before invoking snakemake."""
    directory = tmp_path / "sample"
    directory.mkdir()
    (directory / "params.yml").write_text(
        yaml.safe_dump(
            {
                "config_preset": "does_not_exist",
                "requested_artifacts": ["CALIB_IMAGES"],
                "n_jobs": 7,
            }
        )
    )
    return directory


def test_process_chunk_gives_snakemake_the_configured_core_count(mocker: MockerFixture, chunk_dir) -> None:
    """The core count snakemake is told must be the one the tools parallelise over.

    Left unset it defaults to 1, and snakemake then schedules as if it had a single core while
    `process_spectra`, `proc_calibrate` and `vis_images` each fork `n_jobs` workers underneath
    it -- which is issue #7.
    """
    mock_invoke = mocker.patch.object(process_chunk_module, "SnakemakeInvoke")

    process_chunk(chunk_dir)

    config = mock_invoke.call_args.args[0]
    assert config.n_cores == 7
    assert config.snakefile_path.name == "Snakefile"
    mock_invoke.return_value.invoke.assert_called_once()
