import shutil
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile

import typer
import yaml
from depiction_targeted_preproc.example.run import (
    export_results,
)
from loguru import logger

from depiction.misc.find_file_util import find_one_by_extension
from depiction.persistence import ImzmlReadFile
from depiction_targeted_preproc.pipeline.setup import setup_workdir
from depiction_targeted_preproc.pipeline_config.artifacts_mapping import ARTIFACT_FILES_MAPPING, get_result_files
from depiction_targeted_preproc.pipeline_config.model import (
    PipelineParameters,
    PipelineArtifact,
    PipelineParametersPreset,
)
from depiction_targeted_preproc.workflow.snakemake_invoke import SnakemakeInvoke


def entrypoint(
    script_version: Annotated[str, typer.Option(..., help="The version of the script to run.")] = "",
    input_dir: Annotated[Path, typer.Option(..., help="The input directory.")] = Path("/data/input"),
    output_dir: Annotated[Path, typer.Option(..., help="The output directory.")] = Path("/data/output"),
) -> None:
    if script_version != "":
        raise ValueError(f"Unknown script version: {script_version}")

    # TODO extend in the future
    # Find input files
    input_imzml_file = find_one_by_extension(input_dir=input_dir, extension=".imzML")
    input_panel_file = find_one_by_extension(input_dir=input_dir, extension=".csv")
    workunit_yaml_file = input_dir / "workunit.yaml"

    if not input_imzml_file or not input_panel_file:
        raise RuntimeError(
            f"Input files were not found: input_imzml_file={input_imzml_file}, input_panel_file={input_panel_file}"
        )
    if not workunit_yaml_file.exists():
        raise RuntimeError(f"Workunit yaml file not found: {workunit_yaml_file}")

    # Ensure the input file's checksum passes
    check_imzml_file(ImzmlReadFile(input_imzml_file))

    # Parse the params
    params = parse_parameters(workunit_yaml_file)

    # Set up the workdir for the Snakemake workflow
    work_dir = output_dir / "work"
    setup_workdir(
        params=params,
        input_imzml_file=Path(input_imzml_file),
        input_panel_file=Path(input_panel_file),
        work_dir=work_dir,
    )

    # Execute the snakemake workflow
    sample_name = Path(input_imzml_file).stem
    result_files = get_result_files(params=params, work_dir=work_dir, sample_name=sample_name)
    SnakemakeInvoke().invoke(work_dir=work_dir, result_files=result_files)

    # Export the results
    export_results(
        work_dir=work_dir,
        output_dir=output_dir,
        sample_name=sample_name,
        requested_artifacts=params.requested_artifacts,
        result_file_mapping=ARTIFACT_FILES_MAPPING,
    )
    export_pipeline_params(work_dir=work_dir, output_dir=output_dir, sample_name=sample_name)

    # Zip the results
    zip_results(output_dir=output_dir, sample_name=sample_name)


def export_pipeline_params(work_dir: Path, output_dir: Path, sample_name: str) -> None:
    shutil.copy(work_dir / sample_name / "pipeline_params.yml", output_dir / sample_name / "pipeline_params.yml")


def check_imzml_file(imzml_file: ImzmlReadFile) -> None:
    # TODO this is not very efficient, but is better than not checking the file at all
    logger.info(f"Checking the integrity of the input file: {imzml_file.imzml_file}")
    logger.info(imzml_file.summary())
    if not imzml_file.is_checksum_valid:
        raise RuntimeError(f"Checksum validation failed for the input file: {imzml_file.imzml_file}")


def zip_results(output_dir: Path, sample_name: str) -> None:
    with ZipFile(output_dir / f"{sample_name}.zip", "w") as zipf:
        for file in (output_dir / sample_name).rglob("*"):
            if file.is_file() and file.name != f"{sample_name}.zip":
                zipf.write(file, file.relative_to(output_dir))


def parse_parameters(yaml_file: Path) -> PipelineParameters:
    data = yaml.safe_load(yaml_file.read_text())

    # Find the correct preset
    preset_name = Path(data["application"]["parameters"]["config_preset"]).name
    preset_path = Path(__file__).parent / "config_presets" / f"{preset_name}.yml"
    # Load the presets
    preset = PipelineParametersPreset.validate(yaml.safe_load(preset_path.read_text()))

    # Add n_jobs and requested_artifacts information to build a PipelineParameters
    requested_artifacts = []
    if parse_app_boolean_parameter(data, "output_activate_calibrated_imzml"):
        requested_artifacts.append(PipelineArtifact.CALIB_IMZML)
    if parse_app_boolean_parameter(data, "output_activate_calibrated_ometiff"):
        requested_artifacts.append(PipelineArtifact.CALIB_IMAGES)
    if parse_app_boolean_parameter(data, "output_activate_calibration_qc"):
        requested_artifacts.append(PipelineArtifact.CALIB_QC)
    return PipelineParameters.from_preset_and_settings(
        preset=preset, requested_artifacts=requested_artifacts, n_jobs=32
    )


def parse_app_boolean_parameter(data: dict, key: str) -> bool:
    str_value = data["application"]["parameters"][key]
    if str_value == "true":
        return True
    elif str_value == "false":
        return False
    else:
        logger.warning(f"Unknown boolean value: {str_value}")
        return False


def main() -> None:
    """Provides the CLI around `entrypoint`."""
    typer.run(entrypoint)


if __name__ == "__main__":
    main()
