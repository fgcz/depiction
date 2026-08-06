import pytest
import shutil
from pathlib import Path
from collections.abc import Generator

from depiction.image.ome_tiff import OmeTiff
from depiction_targeted_preproc.app_interface.process_chunk import process_chunk


@pytest.fixture()
def config_yaml_path() -> Path:
    return Path(__file__).parent / "configs" / "global_constant_shift.yml"


def copy_input_files(target_dir: Path):
    # TODO make this more generic and runnable by non fgcz
    source_dir = Path(__file__).parents[1] / "inputs"
    shutil.copy(source_dir / "tonsil.imzML", target_dir / "raw.imzML")
    shutil.copy(source_dir / "tonsil.ibd", target_dir / "raw.ibd")
    (target_dir / "panels").mkdir(parents=True, exist_ok=True)
    shutil.copy(source_dir / "panel.csv", target_dir / "panels" / "unstandardized_full.csv")


@pytest.fixture()
def work_dir(config_yaml_path: Path, tmp_path: Path) -> Generator[Path]:
    dir = tmp_path / "work"
    dir.mkdir()
    shutil.copy(Path(__file__).parent / "configs" / "params.yml", dir / "params.yml")
    shutil.copy(config_yaml_path, dir / "pipeline_params.yml")
    copy_input_files(target_dir=dir)
    yield dir


def test_run_pipeline(work_dir: Path):
    process_chunk(chunk_dir=work_dir)
    # basic checks of the .ome.tiff image
    image = OmeTiff.read_image(work_dir / "images_default.ome.tiff", bg_value=0.0)
    assert image.sizes == {"x": 128, "y": 137, "c": 118}
    assert image.n_nonzero == 10131
