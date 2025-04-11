import numpy as np

from depiction_image_ops.smoothing.percentile_filter import PercentileFilter, KernelShape


def test_get_circle_kernel_mask():
    expected = np.array(
        [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], dtype=bool
    )
    filter_instance = PercentileFilter(kernel_size=5, kernel_shape=KernelShape.Circle)
    np.testing.assert_array_equal(filter_instance.get_circle_kernel_mask(), expected)
