import os
import shutil
import subprocess
from pathlib import Path

import yaml
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters, PipelineArtifact
from depiction_targeted_preproc.pipeline_config.artifacts_mapping import ARTIFACT_FILES_MAPPING
from depiction_targeted_preproc.workflow.snakemake_invoke import SnakemakeInvoke
from loguru import logger


def main() -> None:
    dir_raw = Path(__file__).parent / "data-raw"
    dir_work = Path(__file__).parent / "data-work"
    dir_output = Path(__file__).parent / "data-output"
    dir_work.mkdir(exist_ok=True, parents=True)
    dir_output.mkdir(exist_ok=True, parents=True)
    # sample_name = "menzha_20231208_s607930_64074-b20-30928-a"
    # sample_name = "menzha_20231210_s607943_64005-b20-47740-g"
    sample_name = "menzha_20231208_s607923_tonsil-repro-sample-01"
    # sample_name = "menzha_20231208_s607923_tonsil-repro-sample-01_mcc"
    # sample_name = "menzha_20231208_s607923_tonsil-repro-sample-01_peptnoise"

    params_file = Path(__file__).parents[1] / "pipeline_config" / "default.yml"
    params = PipelineParameters.model_validate(yaml.safe_load(params_file.read_text()))
    logger.info("Pipeline parameters: {params}", params=params.dict())

    if not (dir_work / sample_name / "raw.imzML").exists():
        initial_setup(
            input_imzml=dir_raw / f"{sample_name}.imzML",
            input_mass_list=dir_raw / "mass_list_vend.csv",
            params_file=params_file,
            dir=dir_work / sample_name,
        )

    result_files = get_result_files(params, dir_work, sample_name)
    SnakemakeInvoke().invoke(work_dir=dir_work, result_files=result_files)
    export_results(
        work_dir=dir_work,
        output_dir=dir_output,
        sample_name=sample_name,
        requested_artifacts=params.requested_artifacts,
        result_file_mapping=ARTIFACT_FILES_MAPPING,
    )


def export_results(
    work_dir: Path,
    output_dir: Path,
    sample_name: str,
    requested_artifacts: list[PipelineArtifact],
    result_file_mapping: dict[PipelineArtifact, list[str]],
) -> None:
    for artifact in requested_artifacts:
        if artifact == PipelineArtifact.DEBUG:
            logger.info(f"Skipping export of {artifact}")
            continue
        for file in result_file_mapping[artifact]:
            (output_dir / sample_name / file).parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(work_dir / sample_name / file, output_dir / sample_name / file)


if __name__ == "__main__":
    main()
