import polars as pl
from pydantic import BaseModel


class StandardizeConfig(BaseModel):
    """Configuration for the input panel standardization."""

    column_names: dict[str, set[str]]
    select_columns: list[str]
    default_values: dict[str, str]


def _identify_column_correspondence(config: StandardizeConfig, raw_df: pl.DataFrame) -> dict[str, str]:
    """Identifies the correspondence between the columns in the raw dataframe and the standardized columns,
    returning an entry for each match from raw to standardized column name.

    If required columns are missing, raises a ValueError.
    """
    identified_columns = {}
    for column_name in raw_df.columns:
        for key, values in config.column_names.items():
            if column_name.lower() in values:
                if key not in identified_columns:
                    identified_columns[key] = column_name
                else:
                    raise ValueError(
                        f"Column {column_name} is ambiguous, it could be {key} or {identified_columns[key]}"
                    )
    required_columns = set(config.select_columns) - set(config.default_values.keys())
    missing_columns = required_columns - set(identified_columns.keys())
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    # reverse the mapping
    return {original: target for target, original in identified_columns.items()}


def standardize(config: StandardizeConfig, raw_df: pl.DataFrame) -> pl.DataFrame:
    """Standardizes the provided raw dataframe, according to the configuration."""
    column_correspondence = _identify_column_correspondence(config=config, raw_df=raw_df)
    renamed_df = raw_df.select(column_correspondence.keys()).rename(column_correspondence)
    full_df = renamed_df.with_columns(
        **{
            column: pl.lit(config.default_values[column])
            for column in config.default_values
            if column not in renamed_df.columns
        }
    )
    return full_df.select(config.select_columns)
