from pathlib import Path

from depiction_targeted_preproc.example.run import initial_setup


def get_configs() -> dict[str, Path]:
    config_dir = Path(__file__).parent / "configs"
    return {path.stem: path for path in config_dir.glob("*.yml")}


def run_compare() -> None:
    pass


def set_up_work_dir(work_dir: Path, input_imzml: Path, input_mass_list: Path) -> None:
    configs = get_configs()
    for config_name, config_path in configs.items():
        dir = work_dir / config_name
        initial_setup(input_imzml=input_imzml, input_mass_list=input_mass_list, params_file=config_path, dir=dir)


if __name__ == "__main__":
    run_compare()
