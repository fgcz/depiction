from pathlib import Path

from depiction_targeted_preproc.example.run import initial_setup, RESULT_FILE_MAPPING, snakemake_invoke

from depiction_targeted_preproc.pipeline_config.model import PipelineArtifact


def get_configs() -> dict[str, Path]:
    config_dir = Path(__file__).parent / "configs"
    return {path.stem: path for path in config_dir.glob("*.yml")}


def main() -> None:
    data_raw_path = Path(__file__).parent / "data-raw"
    input_imzml = data_raw_path / "menzha_20231208_s607923_tonsil-repro-sample-01.imzML"
    input_mass_list = data_raw_path / "mass_list_vend.csv"

    work_dir = Path(__file__).parent / "data-work"
    folders = set_up_work_dir(work_dir, input_imzml, input_mass_list)
    requested_files = get_all_output_files(folders)

    snakemake_invoke(work_dir=work_dir, result_files=requested_files)


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
    for config_name, config_path in configs.items():
        dir = work_dir / config_name
        initial_setup(input_imzml=input_imzml, input_mass_list=input_mass_list, params_file=config_path, dir=dir)
        folders.append(dir)
    return folders


if __name__ == "__main__":
    main()
