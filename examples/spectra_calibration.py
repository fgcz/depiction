from pathlib import Path

from matplotlib import pyplot as plt
from numpy.typing import NDArray
from xarray import DataArray
import numpy as np
import scipy
import statsmodels.api as sm
from statsmodels.robust.norms import HuberT
from depiction.persistence import ImzmlReadFile, ImzmlWriteFile
from depiction.spectrum.peak_filtering import FilterNHighestIntensityPartitioned
from depiction.spectrum.peak_picking import BasicInterpolatedPeakPicker
import scipy.stats as stats


class NewCalibrationMethod:
    def extract_spectrum_features(self, peak_mz_arr: NDArray[float], peak_int_arr: NDArray[float]) -> DataArray:
        l_none = 1.000482
        # c_0 = 0.029
        # c_1 = 4.98*10e-4

        plt.scatter(peak_mz_arr, peak_int_arr, color="blue", alpha=0.5, marker="x", label="delta_lambda vs x", s=0.3)
        plt.xlim(1300, 1400)
        plt.show()

        # compute all differences for elements in peak_mz_arr amd store in a DataArray
        # delta = scipy.spatial.distance_matrix(np.expand_dims(peak_mz_arr,1), np.expand_dims(peak_mz_arr,1), p = 1)
        delta = scipy.spatial.distance.pdist(np.expand_dims(peak_mz_arr, 1), metric="cityblock")
        # get all distances smaller then 500

        delta = delta[delta < 500]
        # for each x compute
        # Compute delta_lambda for each x
        delta_lambda = self.compute_distance_from_MCC(delta, l_none)

        # sorted_indices = np.argsort(delta)
        # delta = delta[sorted_indices]
        # delta_lambda = delta_lambda[sorted_indices]

        # Add a constant term with the intercept set to zero
        X = delta.reshape(-1, 1)

        # Fit the model
        robust_model = sm.RLM(delta_lambda, X, M=HuberT())
        results = robust_model.fit()

        if False:
            plt.scatter(delta, delta_lambda, color="blue", alpha=0.5, marker="x", label="delta_lambda vs x", s=0.3)
            plt.plot(delta, results.predict(X), color="red", label="fit")
            plt.show()

        slope = results.params[0]
        peak_mz_corrected = peak_mz_arr + results.predict(np.expand_dims(peak_mz_arr, 1))
        peak_mz_corrected2 = peak_mz_arr * (1 - slope)

        delta_intercept = self.compute_distance_from_MCC(peak_mz_corrected2, l_none)
        intercept_coef = stats.trim_mean(delta_intercept, 0.3)
        # add histogram of delta_intercept
        if False:
            plt.hist(delta_intercept, bins=100)
            # add vertical abline at intercept_coef
            plt.axvline(intercept_coef, color="red")
            plt.show()

        return DataArray([intercept_coef, slope], dims=["c"])

    def compute_distance_from_MCC(self, delta: NDArray[float], l_none=1.000482) -> NDArray[float]:
        delta_lambda = np.zeros_like(delta)
        for i, mi in enumerate(delta):
            term1 = mi % l_none
            if term1 < 0.5:
                delta_lambda[i] = term1
            else:
                delta_lambda[i] = -1 + term1
        return delta_lambda

    def apply_spectrum_model(
        self, spectrum_mz_arr: NDArray[float], spectrum_int_arr: NDArray[float], model_coef: DataArray
    ) -> tuple[NDArray[float], NDArray[float]]:
        spectrum_corrected = spectrum_mz_arr * (1 - model_coef.values[1]) - model_coef.values[0]

        # check calibration here
        delta_intercept = self.compute_distance_from_MCC(spectrum_corrected)
        # intercept_coef = np.mean(delta_intercept)
        intercept_coef = stats.trim_mean(delta_intercept, 0.3)
        # add histogram of delta_intercept
        plt.hist(delta_intercept, bins=100)
        # add vertical abline at intercept_coef
        plt.axvline(intercept_coef, color="red")
        plt.show()
        return (spectrum_corrected, spectrum_int_arr)


def main() -> None:
    with ImzmlReadFile("sample.imzML").reader() as reader:
        spectra_original = []
        for spectrum_id in range(reader.n_spectra):
            mz_arr, int_arr = reader.get_spectrum(spectrum_id)
            spectra_original.append((mz_arr, int_arr))

    # TODO adjust params?
    peak_picker = BasicInterpolatedPeakPicker(
        min_prominence=2,
        min_distance=0.9,
        min_distance_unit="mz",
        peak_filtering=FilterNHighestIntensityPartitioned(max_count=200, n_partitions=8),
    )
    spectra_picked = [peak_picker.pick_peaks(mz_arr, int_arr) for mz_arr, int_arr in spectra_original]

    # show picker result
    mz_arr, int_arr = spectra_picked[0]
    plt.figure()
    ax = plt.gca()
    ax.stem(mz_arr, int_arr, basefmt=" ", markerfmt="x", linefmt="C0-")
    ax.set_xlim(1030, 1080)
    plt.show()

    calib = NewCalibrationMethod()
    all_features = []
    print("Extracting features...")
    for mz_arr, int_arr in spectra_picked:
        features = calib.extract_spectrum_features(mz_arr, int_arr)
        assert features.dims == ("c",)
        print(features)
        all_features.append(features)

    # apply the models to the spectra
    print("Applying models...")
    spectra_calib = []
    for (mz_arr, int_arr), features in zip(spectra_picked, all_features):
        mz_arr, int_arr = calib.apply_spectrum_model(
            spectrum_mz_arr=mz_arr, spectrum_int_arr=int_arr, model_coef=features
        )
        spectra_calib.append((mz_arr, int_arr))

    # compare the spectra somehow...
    print("Plot comparison...")

    mz_arr_original, int_arr_original = spectra_picked[0]
    mz_arr_calib, int_arr_calib = spectra_calib[0]

    print("shape comparison")
    print(mz_arr_original.shape)
    print(mz_arr_calib.shape)

    plt.figure()
    ax = plt.gca()
    ax.stem(mz_arr_original, int_arr_original, label="original", basefmt=" ", markerfmt="x", linefmt="C0-")
    ax.stem(mz_arr_calib, int_arr_calib, label="calibrated", basefmt=" ", markerfmt="x", linefmt="C1-")
    mass_target = 997.53
    ax.axvline(mass_target, color="red", label="target")
    ax.legend()
    ax.set_xlim(mass_target - 5, mass_target + 5)
    ax.set_ylim(0, 1)
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
