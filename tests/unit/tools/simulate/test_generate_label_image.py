import numpy as np
import pytest

from depiction.tools.simulate.generate_label_image import GenerateLabelImage


@pytest.fixture
def generate() -> GenerateLabelImage:
    return GenerateLabelImage(100, 200, 3)


def test_shape(generate) -> None:
    assert generate.shape == (100, 200, 3)


def test_sample_circles_when_indices(generate) -> None:
    channel_indices = [1, 2]
    circles = generate.sample_circles(channel_indices)
    assert len(circles) == 2
    assert circles[0]["i_channel"] == 1
    assert 0 <= circles[0]["center_h"] <= 100
    assert 0 <= circles[0]["center_w"] <= 200
    assert isinstance(circles[0]["radius"], float)
    assert circles[1]["i_channel"] == 2


def test_sample_circles_when_no_indices(generate) -> None:
    circles = generate.sample_circles()
    assert len(circles) == 3
    assert [c["i_channel"] for c in circles] == [0, 1, 2]


def test_add_circles(generate) -> None:
    generate.add_circles(
        [
            {"center_h": 50, "center_w": 100, "radius": 3, "i_channel": 0},
            {"center_h": 70, "center_w": 70, "radius": 3, "i_channel": 1},
        ]
    )
    assert len(generate._layers) == 1
    assert generate._layers[0].shape == (100, 200, 3)
    # check center of circle, and then 4 points away from the center
    layer = generate._layers[0]
    assert layer[50, 100, 0] == 1
    assert layer[70, 70, 1] == 1
    np.testing.assert_equal(layer[:, :, 2], 0)
    assert layer[53, 103, 0] == 0
    assert layer[73, 73, 1] == 0


def test_render(generate) -> None:
    generate._layers = [
        np.array(
            [
                [[1, 0, 0], [0, 1, 0]],
                [[0, 0, 1], [1, 1, 0]],
            ]
        )
    ]
    image = generate.render()
    assert image.n_channels == 3
    assert (2, 2) == image.dimensions
    assert ["synthetic_0", "synthetic_1", "synthetic_2"] == image.channel_names


if __name__ == "__main__":
    pytest.main()
