from pathlib import Path

from matplotlib import pyplot as plt
from numpy.typing import NDArray
from xarray import DataArray

from depiction.persistence import ImzmlReadFile, ImzmlWriteFile
from depiction.spectrum.peak_picking import BasicInterpolatedPeakPicker


class NewCalibrationMethod:
    def extract_spectrum_features(self, peak_mz_arr: NDArray[float], peak_int_arr: NDArray[float]) -> DataArray:
        return DataArray([1, 2, 3], dims=["c"])

    def apply_spectrum_model(
        self, spectrum_mz_arr: NDArray[float], spectrum_int_arr: NDArray[float], model_coef: DataArray
    ) -> tuple[NDArray[float], NDArray[float]]:
        pass


def main() -> None:
    with ImzmlReadFile("sample.imzML").reader() as reader:
        spectra_original = []
        for spectrum_id in range(reader.n_spectra):
            mz_arr, int_arr = reader.get_spectrum(spectrum_id)
            spectra_original.append((mz_arr, int_arr))

    # TODO adjust params?
    peak_picker = BasicInterpolatedPeakPicker(min_prominence=0.5, min_distance=0.9, min_distance_unit="mz")
    spectra_picked = [
        peak_picker.pick_peaks(mz_arr, int_arr) for mz_arr, int_arr in spectra_original
    ]

    calib = NewCalibrationMethod()
    for mz_arr, int_arr in spectra_picked:
        features = calib.extract_spectrum_features(mz_arr, int_arr)
        print(features)

    plt.stem(mz_arr, int_arr)
    plt.xlim(1030, 1080)
    plt.show()


def create_data():
    original_file = "/Users/leo/code/msi/code/msi_targeted_preproc/example/data-work/menzha_20231208_s607930_64074-b20-30928-a/baseline_adj.imzML"
    imzml_file = ImzmlReadFile(original_file)
    output_file = Path("sample.imzML")
    if output_file.exists():
        print(f"{output_file} file already exists, skipping")
        return
    spectra_ids = range(0, imzml_file.n_spectra, imzml_file.n_spectra // 10)

    write_file = ImzmlWriteFile(output_file, imzml_mode=imzml_file.imzml_mode)
    with imzml_file.reader() as reader, write_file.writer() as writer:
        for spectrum_id in spectra_ids:
            mz_arr, int_arr, coords = reader.get_spectrum_with_coords(spectrum_id)
            writer.add_spectrum(mz_arr, int_arr, coords)


if __name__ == "__main__":
    # create_data()
    main()
