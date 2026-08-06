import pytest
import shutil
from pathlib import Path
from collections.abc import Generator

from depiction.image.ome_tiff import OmeTiff
from depiction_targeted_preproc.app_interface.process_chunk import process_chunk


#: Files this test needs, none of which are in the repository: the imzML/ibd pair is a
#: 1.3 GB acquisition and `panel.csv` accompanies it. See `system_tests/README.md` for
#: where to get them.
REQUIRED_INPUTS = ("tonsil.imzML", "tonsil.ibd", "panel.csv")


@pytest.fixture()
def config_yaml_path() -> Path:
    return Path(__file__).parent / "configs" / "global_constant_shift.yml"


@pytest.fixture()
def input_dir() -> Path:
    """The local fixture directory, skipping the test when it has not been populated."""
    source_dir = Path(__file__).parents[1] / "inputs"
    missing = [name for name in REQUIRED_INPUTS if not (source_dir / name).exists()]
    if missing:
        pytest.skip(
            f"Missing system test inputs in {source_dir}: {', '.join(missing)}. "
            f"See system_tests/README.md for how to obtain them."
        )
    return source_dir


def copy_input_files(source_dir: Path, target_dir: Path):
    shutil.copy(source_dir / "tonsil.imzML", target_dir / "raw.imzML")
    shutil.copy(source_dir / "tonsil.ibd", target_dir / "raw.ibd")
    (target_dir / "panels").mkdir(parents=True, exist_ok=True)
    shutil.copy(source_dir / "panel.csv", target_dir / "panels" / "unstandardized_full.csv")


@pytest.fixture()
def work_dir(config_yaml_path: Path, input_dir: Path, tmp_path: Path) -> Generator[Path]:
    dir = tmp_path / "work"
    dir.mkdir()
    shutil.copy(Path(__file__).parent / "configs" / "params.yml", dir / "params.yml")
    shutil.copy(config_yaml_path, dir / "pipeline_params.yml")
    copy_input_files(source_dir=input_dir, target_dir=dir)
    yield dir


def test_run_pipeline(work_dir: Path):
    process_chunk(chunk_dir=work_dir)
    # basic checks of the .ome.tiff image -- the expected values below are pinned to the
    # tonsil acquisition named in system_tests/README.md and mean nothing for any other input
    image = OmeTiff.read_image(work_dir / "images_default.ome.tiff", bg_value=0.0)
    assert image.sizes == {"x": 128, "y": 137, "c": 118}
    assert image.n_nonzero == 10131
