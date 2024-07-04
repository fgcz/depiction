from __future__ import annotations

from pathlib import Path

import cyclopts
import polars as pl

app = cyclopts.App()

COLUMN_NAMES = {
    "mass": {"m/z", "mass", "pc-mt (m+h)+"},
    "label": {"marker", "label"},
    "tol": {"tol", "tolerance"},
}


def identify_column_correspondence(raw_df: pl.DataFrame) -> dict[str, str]:
    identified_columns = {}
    for column_name in raw_df.columns:
        for key, values in COLUMN_NAMES.items():
            if column_name.lower() in values:
                if key not in identified_columns:
                    identified_columns[key] = column_name
                else:
                    raise ValueError(
                        f"Column {column_name} is ambiguous, it could be {key} or {identified_columns[key]}"
                    )
    required_columns = {"mass", "label"}
    missing_columns = required_columns - set(identified_columns.keys())
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    # reverse the mapping
    return {original: target for target, original in identified_columns.items()}


@app.default
def mass_list_preparation(
    raw_csv: Path,
    out_csv: Path,
) -> None:
    raw_df = pl.read_csv(raw_csv)

    # identify columns
    column_correspondence = identify_column_correspondence(raw_df)

    # rename columns (and drop the rest)
    renamed = (
        raw_df.select(column_correspondence.values())
        .rename(column_correspondence)
        .select(sorted(column_correspondence.values()))
    )

    # add tol column if not present
    if "tol" not in renamed.columns:
        renamed = renamed.with_column("tol", pl.Null)

    # write the results
    renamed.write_csv(out_csv)


if __name__ == "__main__":
    app()
