import numpy as np
import pytest
from depiction.spectrum.peak_filtering.filter_by_snr_threshold import FilterBySnrThresholdConfig, FilterBySnrThreshold
from depiction.spectrum.unit_conversion import WindowSize


@pytest.fixture
def mock_filter_config() -> FilterBySnrThresholdConfig:
    return FilterBySnrThresholdConfig(
        snr_threshold=1.0,
        window_size={"size": 5, "unit": "index"},
    )


@pytest.fixture()
def mock_filter(mock_filter_config: FilterBySnrThresholdConfig) -> FilterBySnrThreshold:
    return FilterBySnrThreshold(config=mock_filter_config)


@pytest.fixture
def sample_spectrum():
    # Create a sample spectrum with 100 elements
    mz = np.linspace(100, 1000, 100)
    intensity = np.zeros(100)

    # Add some peaks
    peak_indices = [10, 30, 50, 70, 90]
    for idx in peak_indices:
        intensity[idx] = 100

    # Add some noise
    noise = np.random.normal(0, 5, 100)
    intensity += noise
    intensity = np.abs(intensity)  # Ensure all intensities are non-negative

    return mz, intensity


def test_filter_peaks_basic(mock_filter, sample_spectrum):
    mz, intensity = sample_spectrum
    peak_mz = mz[intensity > 50]
    peak_intensity = intensity[intensity > 50]

    filtered_mz, filtered_intensity = mock_filter.filter_peaks(mz, intensity, peak_mz, peak_intensity)

    assert len(filtered_mz) > 0
    assert len(filtered_mz) == len(filtered_intensity)
    assert len(filtered_mz) <= len(peak_mz)


def test_filter_peaks_all_below_threshold(mock_filter, sample_spectrum):
    mz, intensity = sample_spectrum
    # Set a very high SNR threshold
    mock_filter.config.snr_threshold = 1000

    peak_mz = mz[intensity > 50]
    peak_intensity = intensity[intensity > 50]

    filtered_mz, filtered_intensity = mock_filter.filter_peaks(mz, intensity, peak_mz, peak_intensity)

    assert len(filtered_mz) == 0
    assert len(filtered_intensity) == 0


def test_filter_peaks_all_above_threshold(mock_filter, sample_spectrum):
    mz, intensity = sample_spectrum
    # Set a very low SNR threshold
    mock_filter.config.snr_threshold = 0.1

    peak_mz = mz[intensity > 50]
    peak_intensity = intensity[intensity > 50]

    filtered_mz, filtered_intensity = mock_filter.filter_peaks(mz, intensity, peak_mz, peak_intensity)

    assert len(filtered_mz) == len(peak_mz)
    assert len(filtered_intensity) == len(peak_intensity)


def test_estimate_noise_level(mock_filter):
    # Create a simple signal with known noise
    signal = np.array([0, 0, 10, 0, 0, 20, 0, 0, 30, 0, 0])
    noise_level = mock_filter._estimate_noise_level(signal, kernel_size=3)

    assert len(noise_level) == len(signal)
    assert np.allclose(noise_level[2::3], [10, 20, 30], atol=1e-6)  # Peak positions
    assert np.all(noise_level[np.arange(len(signal)) % 3 != 2] < 1e-6)  # Non-peak positions


def test_window_size_conversion(mock_filter, sample_spectrum):
    mz, intensity = sample_spectrum

    # Test with index-based window size
    mock_filter.config.window_size = WindowSize(size=5, unit="index")
    filtered_mz, filtered_intensity = mock_filter.filter_peaks(mz, intensity, mz, intensity)
    assert len(filtered_mz) > 0

    # Test with m/z-based window size
    mock_filter.config.window_size = WindowSize(size=50, unit="mz")
    filtered_mz, filtered_intensity = mock_filter.filter_peaks(mz, intensity, mz, intensity)
    assert len(filtered_mz) > 0


# def test_edge_cases(mock_filter):
#    # Test with empty arrays
#    empty_mz, empty_intensity = np.array([]), np.array([])
#    filtered_mz, filtered_intensity = mock_filter.filter_peaks(empty_mz, empty_intensity, empty_mz, empty_intensity)
#    assert len(filtered_mz) == 0
#    assert len(filtered_intensity) == 0
#
#    # Test with single-element arrays
#    single_mz, single_intensity = np.array([100]), np.array([10])
#    filtered_mz, filtered_intensity = mock_filter.filter_peaks(single_mz, single_intensity, single_mz, single_intensity)
#    assert len(filtered_mz) <= 1
#    assert len(filtered_intensity) <= 1
#
#    # Test with arrays where no peaks meet the SNR threshold
#    mz = np.linspace(100, 1000, 100)
#    intensity = np.ones(100)  # All intensities are 1
#    peak_mz = np.array([300, 600, 900])
#    peak_intensity = np.array([1.5, 1.5, 1.5])  # Below the SNR threshold of 2
#    filtered_mz, filtered_intensity = mock_filter.filter_peaks(mz, intensity, peak_mz, peak_intensity)
#    assert len(filtered_mz) == 0
#    assert len(filtered_intensity) == 0
#
#    # Test with arrays where all peaks meet the SNR threshold
#    mz = np.linspace(100, 1000, 100)
#    intensity = np.ones(100)  # All intensities are 1
#    peak_mz = np.array([300, 600, 900])
#    peak_intensity = np.array([10, 10, 10])  # Well above the SNR threshold of 2
#    filtered_mz, filtered_intensity = mock_filter.filter_peaks(mz, intensity, peak_mz, peak_intensity)
#    assert len(filtered_mz) == len(peak_mz)
#    assert len(filtered_intensity) == len(peak_intensity)
#    np.testing.assert_array_equal(filtered_mz, peak_mz)
#    np.testing.assert_array_equal(filtered_intensity, peak_intensity)
#
