import os
import shutil
import subprocess
from pathlib import Path

import yaml
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters, PipelineArtifact
from loguru import logger

RESULT_FILE_MAPPING = {
    PipelineArtifact.CALIB_IMZML: ["calibrated.imzML", "calibrated.ibd"],
    # PipelineArtifact.CALIB_QC: ["calib_qc.pdf"],
    # TODO
    PipelineArtifact.CALIB_QC: [
        "qc/plot_marker_presence.pdf",
        "qc/plot_peak_density_combined.pdf",
        "qc/plot_peak_density_grouped.pdf",
        "qc/plot_calibration_map.pdf",
    ],
    PipelineArtifact.CALIB_IMAGES: ["images_default.ome.tiff", "images_default_norm.ome.tiff"],
    # PipelineArtifact.CALIB_HEATMAP: ["images_calib_heatmap.ome.tiff"],
    # TODO
    PipelineArtifact.CALIB_HEATMAP: [],
    PipelineArtifact.DEBUG: [
        "qc/plot_marker_presence_cv.pdf",
        "qc/plot_spectra_for_marker.pdf",
        "qc/plot_sample_spectra_before_after.pdf",
        "qc/plot_peak_counts.pdf",
    ],
}


def main() -> None:
    dir_raw = Path(__file__).parent / "data-raw"
    dir_work = Path(__file__).parent / "data-work"
    dir_output = Path(__file__).parent / "data-output"
    dir_work.mkdir(exist_ok=True, parents=True)
    dir_output.mkdir(exist_ok=True, parents=True)
    # sample_name = "menzha_20231208_s607930_64074-b20-30928-a"
    # sample_name = "menzha_20231210_s607943_64005-b20-47740-g"
    sample_name = "menzha_20231208_s607923_tonsil-repro-sample-01"
    # sample_name = "menzha_20231208_s607923_tonsil-repro-sample-01_peptnoise"

    params_file = Path(__file__).parents[1] / "pipeline_config" / "default.yml"
    params = PipelineParameters.model_validate(yaml.safe_load(params_file.read_text()))
    logger.info("Pipeline parameters: {params}", params=params.dict())

    if not (dir_work / sample_name / "raw.imzML").exists():
        initial_setup(
            input_imzml=dir_raw / f"{sample_name}.imzML",
            input_mass_list=dir_raw / "mass_list_vend.csv",
            params_file=params_file,
            work_dir=dir_work,
        )

    result_files = get_result_files(params, dir_work, sample_name)
    snakemake_invoke(work_dir=dir_work, result_files=result_files)
    export_results(
        work_dir=dir_work,
        output_dir=dir_output,
        sample_name=sample_name,
        requested_artifacts=params.requested_artifacts,
        result_file_mapping=RESULT_FILE_MAPPING,
    )


def get_result_files(params: PipelineParameters, work_dir: Path, sample_name: str):
    result_files = list(
        {
            work_dir / sample_name / file
            for artifact in params.requested_artifacts
            for file in RESULT_FILE_MAPPING[artifact]
        }
    )
    return result_files


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


def initial_setup(input_imzml: Path, input_mass_list: Path, params_file: Path, work_dir: Path) -> None:
    sample_name = input_imzml.stem
    sample_dir = work_dir / sample_name
    sample_dir.mkdir(exist_ok=True)
    shutil.copy(input_imzml, sample_dir / "raw.imzML")
    shutil.copy(input_imzml.with_suffix(".ibd"), sample_dir / "raw.ibd")
    shutil.copy(input_mass_list, sample_dir / "images_default_mass_list.csv")
    shutil.copy(params_file, sample_dir / "pipeline_params.yml")


def snakemake_invoke(work_dir: Path, result_files: list[Path]) -> None:
    snakefile_path = Path(__file__).parents[1] / "workflow" / "Snakefile"
    workflow_dir = Path(__file__).parents[1] / "workflow"

    snakemake_bin = shutil.which("snakemake")
    if snakemake_bin is None:
        raise RuntimeError(f"snakemake not found, check PATH: {os.environ['PATH']}")
    command = [
        snakemake_bin,
        "-d",
        str(work_dir),
        "--cores",
        "1",
        "--snakefile",
        str(snakefile_path),
        *[str(file) for file in result_files],
    ]
    logger.info("Executing {command}", command=command)
    subprocess.run(
        command,
        cwd=workflow_dir,
        check=True,
    )


if __name__ == "__main__":
    main()
