from pathlib import Path

from depiction_targeted_preproc.example.run import initial_setup, RESULT_FILE_MAPPING

from depiction_targeted_preproc.pipeline_config.model import PipelineArtifact
from depiction_targeted_preproc.workflow.snakemake_invoke import SnakemakeInvoke


def get_configs() -> dict[str, Path]:
    config_dir = Path(__file__).parent / "configs"
    return {path.stem: path for path in config_dir.glob("*.yml")}


def prepare_tasks(input_imzml_path: Path, work_dir: Path) -> list[Path]:
    input_mass_list = input_imzml_path.parent / "mass_list.raw.csv"
    folders = set_up_work_dir(work_dir, input_imzml_path, input_mass_list)
    requested_files = get_all_output_files(folders)

    combined_dir = work_dir / input_imzml_path.stem
    exp_files = [combined_dir / "exp_compare_cluster_stats.pdf",
                 combined_dir / "exp_plot_compare_peak_density.pdf"]
    return requested_files + exp_files


def main() -> None:
    work_dir = Path(__file__).parent / "data-work"
    data_raw_dir = Path(__file__).parent / "data-raw"

    imzmls = [
        "menzha_20231208_s607923_tonsil-repro-sample-01.imzML",
        "menzha_20231208_s607930_64074-b20-30928-a.imzML",
        "menzha_20240212_tonsil_06-50.imzML",
    ]

    requested_files = []
    for imzml in imzmls:
        requested_files += prepare_tasks(data_raw_dir / imzml, work_dir=work_dir)

    SnakemakeInvoke().invoke(work_dir=work_dir, result_files=requested_files, n_cores=4)


def get_all_output_files(folders: list[Path]) -> list[Path]:
    artifacts = [PipelineArtifact.CALIB_QC, PipelineArtifact.CALIB_IMAGES, PipelineArtifact.DEBUG]
    filenames = [name for artifact in artifacts for name in RESULT_FILE_MAPPING[artifact]]

    all_files = []
    for folder in folders:
        for filename in filenames:
            all_files.append(folder / filename)

    return all_files


def set_up_work_dir(work_dir: Path, input_imzml: Path, input_mass_list: Path) -> list[Path]:
    configs = get_configs()
    folders = []
    sample_name = input_imzml.stem
    for config_name, config_path in configs.items():
        dir = work_dir / sample_name / config_name
        initial_setup(
            input_imzml=input_imzml,
            input_mass_list=input_mass_list,
            params_file=config_path,
            dir=dir,
            mass_list_filename="mass_list.raw.csv",
        )
        folders.append(dir)
    return folders


if __name__ == "__main__":
    main()
