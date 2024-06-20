import shutil
from pathlib import Path

import yaml

from depiction_targeted_preproc.pipeline_config.artifacts_mapping import get_result_files
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters
from depiction_targeted_preproc.workflow.snakemake_invoke import SnakemakeInvoke


def setup_sim_dir(path: Path) -> None:
    source_configs_dir = Path(__file__).parents[1] / "pipeline_config"
    path.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(source_configs_dir / "default.yml", path / "pipeline_params.yml")
    (path / "config").mkdir(exist_ok=True)
    shutil.copyfile(source_configs_dir / "default_simulate.yml", path / "config" / "simulate.yml")

    # path_source_mass_list = Path(__file__).parents[1] / "example" / "data-raw" / "mass_list_vend.csv"
    # shutil.copyfile(path_source_mass_list, path / "mass_list.raw.csv")


# TODO why does it not work?


def main() -> None:
    dir_work = Path(__file__).parent / "data-work"
    # dir_output = Path(__file__).parent / "data-output"
    dir_work.mkdir(exist_ok=True, parents=True)
    # dir_output.mkdir(exist_ok=True, parents=True)

    sample_name = "dummy01_sim"
    setup_sim_dir(dir_work / sample_name)
    params_file = Path(__file__).parent / "default.yml"
    params = PipelineParameters.model_validate(yaml.safe_load(params_file.read_text()))
    result_files = get_result_files(params, dir_work, sample_name)
    SnakemakeInvoke().invoke(work_dir=dir_work, result_files=result_files)


if __name__ == "__main__":
    main()
