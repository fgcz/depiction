import polars as pl
import polars.testing
import pytest

from depiction_targeted_preproc.panel.standardize_input_panel import StandardizeConfig, standardize


@pytest.fixture
def config() -> StandardizeConfig:
    return StandardizeConfig(
        column_names={"mass": {"m/z"}, "label": {"label", "x"}},
        select_columns=["mass", "label", "type"],
        default_values={"type": "something"},
    )


def test_standardize(config: StandardizeConfig) -> None:
    raw_df = pl.DataFrame({"m/z": [1, 2, 3], "x": ["a", "b", "c"]})
    result = standardize(config=config, raw_df=raw_df)
    expected_df = pl.DataFrame(
        {"mass": [1, 2, 3], "label": ["a", "b", "c"], "type": ["something", "something", "something"]}
    )
    pl.testing.assert_frame_equal(result, expected_df)


def test_config_load_packaged() -> None:
    config = StandardizeConfig.load_packaged("standardize_main_config.yml")
    assert config.select_columns == ["mass", "label", "type"]
