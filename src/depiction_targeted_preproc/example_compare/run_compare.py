from pathlib import Path

from depiction_targeted_preproc.pipeline.setup_old import initial_setup
from depiction_targeted_preproc.pipeline_config.artifacts_mapping import get_all_output_files
from depiction_targeted_preproc.workflow.snakemake_invoke import SnakemakeInvoke


def get_configs() -> dict[str, Path]:
    config_dir = Path(__file__).parent / "configs"
    return {path.stem: path for path in config_dir.glob("*.yml")}


def prepare_tasks(input_imzml_path: Path, work_dir: Path) -> list[Path]:
    input_mass_list = input_imzml_path.parent / "mass_list.raw.csv"
    folders = set_up_work_dir(work_dir, input_imzml_path, input_mass_list)
    requested_files = get_all_output_files(folders)

    combined_dir = work_dir / input_imzml_path.stem
    exp_files = [
        combined_dir / "exp_compare_cluster_stats.pdf",
        combined_dir / "exp_plot_compare_peak_density.pdf",
        combined_dir / "exp_plot_map_comparison.pdf",
    ]
    return requested_files + exp_files


def main() -> None:
    work_dir = Path(__file__).parent / "data-work"
    data_raw_dir = Path(__file__).parent / "data-raw"

    imzmls = [
        "menzha_20231208_s607923_tonsil-repro-sample-01.imzML",
        # "menzha_20231208_s607923_tonsil-repro-sample-02.imzML",
        # "menzha_20231208_s607930_64074-b20-30928-a.imzML",
        # "menzha_20240212_tonsil_06-50.imzML",
    ]

    requested_files = []
    for imzml in imzmls:
        requested_files += prepare_tasks(data_raw_dir / imzml, work_dir=work_dir)

    ## TODO quick hack
    requested_files = [f for f in requested_files if "mini" in str(f) or "presence" in str(f)]

    SnakemakeInvoke(continue_on_error=True).invoke(work_dir=work_dir, result_files=requested_files, n_cores=4)


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
