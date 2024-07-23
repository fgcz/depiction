import numpy as np
import pytest
from depiction.clustering.extrapolate import get_cluster_centers, extrapolate_labels
from numpy.testing import assert_array_almost_equal, assert_array_equal


@pytest.fixture
def basic_features():
    return np.array([[1, 2], [3, 4], [5, 6], [7, 8]])


@pytest.fixture
def basic_labels():
    return np.array([0, 0, 1, 1])


@pytest.fixture
def basic_expected_centers():
    return np.array([[2, 3], [6, 7]])


@pytest.fixture
def high_dim_features():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])


@pytest.fixture
def full_features(basic_features):
    return np.vstack([basic_features, [[9, 10]]])


def test_extrapolate_labels_basic(mocker, basic_features, basic_labels, full_features, basic_expected_centers):
    mocker.patch("depiction.clustering.extrapolate.get_cluster_centers", return_value=basic_expected_centers)

    expected_labels = np.array([0, 0, 1, 1, 1])
    result = extrapolate_labels(basic_features, basic_labels, full_features)

    assert_array_equal(result, expected_labels)


def test_extrapolate_labels_single_cluster(mocker, basic_features, full_features):
    mocker.patch("depiction.clustering.extrapolate.get_cluster_centers", return_value=np.array([[4, 5]]))

    sample_labels = np.array([0, 0, 0, 0])
    expected_labels = np.array([0, 0, 0, 0, 0])

    result = extrapolate_labels(basic_features, sample_labels, full_features)
    assert_array_equal(result, expected_labels)


def test_extrapolate_labels_high_dimensionality(mocker, high_dim_features):
    sample_features = high_dim_features[:3]
    sample_labels = np.array([0, 1, 1])
    full_features = high_dim_features
    mock_centers = np.array([[1, 2, 3], [5.5, 6.5, 7.5]])

    mocker.patch("depiction.clustering.extrapolate.get_cluster_centers", return_value=mock_centers)

    expected_labels = np.array([0, 1, 1, 1])
    result = extrapolate_labels(sample_features, sample_labels, full_features)

    assert_array_equal(result, expected_labels)


def test_extrapolate_labels_empty_full_features(mocker, basic_features, basic_labels):
    mocker.patch("depiction.clustering.extrapolate.get_cluster_centers", return_value=np.array([[2, 3], [6, 7]]))

    full_features = np.empty((0, 2))
    expected_labels = np.array([], dtype=int)

    result = extrapolate_labels(basic_features, basic_labels, full_features)
    assert_array_equal(result, expected_labels)


def test_extrapolate_labels_different_feature_count(basic_features, basic_labels):
    with pytest.raises(ValueError, match="Number of features must be the same"):
        full_features = np.array([[1, 2, 3], [4, 5, 6]])
        extrapolate_labels(basic_features, basic_labels, full_features)


def test_extrapolate_labels_mock_calls(mocker, basic_features, basic_labels, full_features):
    mock_get_centers = mocker.patch("depiction.clustering.extrapolate.get_cluster_centers")
    mock_norm = mocker.patch("numpy.linalg.norm")
    mock_argmin = mocker.patch("numpy.argmin")

    mock_get_centers.return_value = np.array([[2, 3], [6, 7]])
    mock_norm.return_value = np.array([1, 2])
    mock_argmin.return_value = 0

    extrapolate_labels(basic_features, basic_labels, full_features)

    mock_get_centers.assert_called_once_with(features=basic_features, labels=basic_labels)
    assert mock_norm.call_count == full_features.shape[0]
    assert mock_argmin.call_count == full_features.shape[0]


def test_get_cluster_centers_basic(basic_features, basic_labels, basic_expected_centers):
    result = get_cluster_centers(basic_features, basic_labels)
    assert_array_almost_equal(result, basic_expected_centers)


def test_get_cluster_centers_single_cluster():
    features = np.array([[1, 2], [3, 4], [5, 6]])
    labels = np.array([0, 0, 0])
    expected_centers = np.array([[3, 4]])

    result = get_cluster_centers(features, labels)
    assert_array_almost_equal(result, expected_centers)


def test_get_cluster_centers_non_zero_min_label(basic_features, basic_expected_centers):
    labels = np.array([1, 1, 2, 2])

    result = get_cluster_centers(basic_features, labels)
    assert_array_almost_equal(result, basic_expected_centers)


def test_get_cluster_centers_high_dimensionality(high_dim_features):
    labels = np.array([0, 0, 1, 1])
    expected_centers = np.array([[2.5, 3.5, 4.5], [8.5, 9.5, 10.5]])

    result = get_cluster_centers(high_dim_features, labels)
    assert_array_almost_equal(result, expected_centers)


def test_get_cluster_centers_empty_cluster():
    features = np.array([[1, 2], [3, 4], [5, 6]])
    labels = np.array([0, 2, 2])
    expected_centers = np.array([[1, 2], [np.nan, np.nan], [4, 5]])

    result = get_cluster_centers(features, labels)
    assert_array_almost_equal(result, expected_centers)


def test_get_cluster_centers_input_validation(basic_features):
    with pytest.raises(IndexError):
        get_cluster_centers(basic_features, np.array([0, 1, 2]))


def test_get_cluster_centers_mock_mean(mocker, basic_features, basic_labels):
    mock_mean = mocker.patch("numpy.mean")
    mock_mean.return_value = np.array([10, 20])

    expected_centers = np.array([[10, 20], [10, 20]])

    result = get_cluster_centers(basic_features, basic_labels)
    assert_array_almost_equal(result, expected_centers)
    assert mock_mean.call_count == 2


if __name__ == "__main__":
    pytest.main()
