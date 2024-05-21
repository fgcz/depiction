# TODO deprecated

import argparse
import enum
import logging
from pathlib import Path

from tqdm import tqdm

from ionplotter.estimate_ppm_error import EstimatePPMError
from ionplotter.evaluate_bins import EvaluateBins
from ionplotter.parallel_ops.parallel_config import ParallelConfig
from ionplotter.parallel_ops.write_spectra_parallel import WriteSpectraParallel
from ionplotter.persistence import (
    ImzmlReadFile,
    ImzmlModeEnum,
    ImzmlWriteFile,
    ImzmlReader,
    ImzmlWriter,
)


class AlignImzmlMethod(enum.Enum):
    # broken for picked input data
    CARDINAL_ESTIMATE_PPM = "CARDINAL_ESTIMATE_PPM"
    # broken for picked input data
    FIRST_MZ_ARR = "FIRST_MZ_ARR"



class AlignImzml:
    """Aligns the m/z values of an imzml file to a common set of bins."""

    def __init__(
        self,
        input_file: ImzmlReadFile,
        output_file_path: str,
        method: AlignImzmlMethod,
        parallel_config: ParallelConfig = None,
    ) -> None:
        self._input_file = input_file
        self._output_file_path = output_file_path
        self._method = method
        self._parallel_config = parallel_config

    def evaluate(self) -> ImzmlReadFile:
        # Check if the input file is already in "continuous" mode.
        if self._input_file.imzml_mode == ImzmlModeEnum.CONTINUOUS:
            logging.info('No binning required, input file is already in "continuous" mode.')
            return self._input_file

        # Determine the bins for alignment.
        bin_eval = self._get_bin_eval(input_file=self._input_file)

        # Apply the alignment.
        self._apply_alignment(
            input_file=self._input_file,
            output_file=ImzmlWriteFile(self._output_file_path, imzml_mode=ImzmlModeEnum.CONTINUOUS),
            bin_eval=bin_eval,
        )
        return ImzmlReadFile(self._output_file_path)

    def _apply_alignment(
        self,
        input_file: ImzmlReadFile,
        output_file: ImzmlWriteFile,
        bin_eval: EvaluateBins,
    ) -> None:
        def chunk_operation(reader: ImzmlReader, spectra_ids: list[int], writer: ImzmlWriter) -> None:
            mz_arr_new = bin_eval.mz_values
            for spectrum_id in tqdm(spectra_ids):
                mz_arr_orig, int_arr_orig = reader.get_spectrum(spectrum_id)
                int_arr_new = bin_eval.evaluate(mz_arr_orig, int_arr_orig)
                writer.add_spectrum(mz_arr_new, int_arr_new, reader.coordinates[spectrum_id])

        parallelize = WriteSpectraParallel.from_config(self._parallel_config)
        parallelize.map_chunked_to_file(read_file=input_file, write_file=output_file, operation=chunk_operation)

    def _get_bin_eval(self, input_file: ImzmlReadFile) -> EvaluateBins:
        if self._method == AlignImzmlMethod.CARDINAL_ESTIMATE_PPM:
            print('Estimating PPM error and creating bins for "continuous" mode.')
            estimate_ppm = EstimatePPMError()
            results = estimate_ppm.estimate(input_file)

            ppm_error = results["ppm_median"]
            mz_min = results["mz_min"]
            mz_max = results["mz_max"]
            print(f"Estimated PPM error: {ppm_error:.3f}")
            print(f"m/z range: {mz_min:.3f} - {mz_max:.3f}")

            mz_arr_new = EstimatePPMError().ppm_to_mz_values(ppm_error, mz_min, mz_max)
            print(f"Number of bins: {len(mz_arr_new)}")
            return EvaluateBins.from_mz_values(mz_arr_new)
        elif self._method == AlignImzmlMethod.FIRST_MZ_ARR:
            with input_file.reader() as reader:
                return EvaluateBins.from_mz_values(reader.get_spectrum_mz(0))
        else:
            raise ValueError(f"Unknown method: {self._method}")


def main_align_imzml(input_imzml: str, output_imzml: str, method: str, n_jobs: int) -> Path:
    align = AlignImzml(
        input_file=ImzmlReadFile(input_imzml),
        output_file_path=output_imzml,
        method=AlignImzmlMethod(method),
        parallel_config=ParallelConfig(n_jobs=n_jobs, task_size=None, verbose=10),
    )
    aligned_file = align.evaluate()
    print(f"Aligned file: {aligned_file.imzml_file}")
    return aligned_file.imzml_file


def main() -> None:
    parser = argparse.ArgumentParser()
    # Currently there is only one algorithm, but it makes sense to already make it an argument, because in the future
    # different methods might be a lot more appropriate.
    parser.add_argument(
        "--method",
        choices=["CARDINAL_ESTIMATE_PPM", "FIRST_MZ_ARR"],
        default="CARDINAL_ESTIMATE_PPM",
        type=str,
    )
    parser.add_argument("input_imzml", type=str)
    parser.add_argument("output_imzml", type=str)
    parser.add_argument("--n-jobs", type=int, default=32)
    args = vars(parser.parse_args())
    main_align_imzml(**args)


if __name__ == "__main__":
    main()
