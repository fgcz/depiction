from pathlib import Path

import polars as pl


def standardize_table(input_df: pl.DataFrame) -> pl.DataFrame:
    # TODO this is a total hack for a quick setup
    mapping = {}
    for column in input_df.columns:
        if column.lower() in ["marker", "label"]:
            mapping[column] = "label"
        elif column.lower() in ["mass", "m/z", "pc-mt (m+h)+"]:
            mapping[column] = "mass"
        elif column.lower() in ["tol"]:
            mapping[column] = "tol"
    output_df = input_df.rename(mapping)

    if "tol" not in output_df:
        # TODO make configurable
        output_df = output_df.with_columns([pl.Series("tol", [0.2] * len(output_df))])
    return output_df


def copy_standardized_table(input_csv: Path, output_csv: Path):
    input_df = pl.read_csv(input_csv)
    write_standardized_table(input_df, output_csv)


def write_standardized_table(input_df: pl.DataFrame, output_csv: Path) -> None:
    output_df = standardize_table(input_df)
    output_df.write_csv(output_csv)
