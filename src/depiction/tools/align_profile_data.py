import shutil
import typer

from depiction.estimate_ppm_error import EstimatePPMError
from depiction.spectrum.evaluate_bins import EvaluateBins
from depiction.parallel_ops import ParallelConfig
from depiction.persistence import ImzmlModeEnum, ImzmlWriteFile, ImzmlReadFile
from depiction.tools.align_imzml import main_align_imzml
from pathlib import Path


def align_profile_data(input_imzml_path: str, output_imzml_path: str) -> None:
    aligned_path = main_align_imzml(
        input_imzml=input_imzml_path, output_imzml=output_imzml_path, method="FIRST_MZ_ARR", n_jobs=20
    )
    if aligned_path == Path(input_imzml_path):
        shutil.copy(input_imzml_path, output_imzml_path)
        shutil.copy(Path(input_imzml_path).with_suffix(".ibd"), Path(output_imzml_path).with_suffix(".ibd"))


def main(input_imzml_path: Path, output_imzml_path: Path, ppm_res: int = 100, n_jobs: int = 20) -> None:
    input_imzml = ImzmlReadFile(input_imzml_path)
    output_imzml = ImzmlWriteFile(output_imzml_path, imzml_mode=ImzmlModeEnum.CONTINUOUS)
    parallel_config = ParallelConfig(n_jobs=n_jobs, task_size=None)

    # TODO this method should be moved to a nicer place
    with input_imzml.reader() as reader:
        n_spectra = reader.n_spectra
        mz_range = reader.get_spectra_mz_range(list(range(0, n_spectra, n_spectra // 10)))
    mz_bins = EstimatePPMError.ppm_to_mz_values(ppm_error=ppm_res, mz_min=mz_range[0], mz_max=mz_range[1])
    eval_bins = EvaluateBins.from_mz_values(mz_bins)
    eval_bins.evaluate_file(read_file=input_imzml, write_file=output_imzml, parallel_config=parallel_config)


if __name__ == "__main__":
    typer.run(main)
