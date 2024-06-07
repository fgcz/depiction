from pathlib import Path
from typing import Annotated

import altair as alt
import polars as pl
import typer
from typer import Option


def load_data(csv_paths: list[Path]) -> pl.DataFrame:
    collect = []
    for csv_path in csv_paths:
        df = pl.read_csv(csv_path)
        collect.append(df.with_columns(label=pl.lit(csv_path.parent.name)))
    return pl.concat(collect)


def compare_cluster_stats(input_csv_path: list[Path], output_pdf: Annotated[Path, Option()]) -> None:
    data = load_data(input_csv_path)
    chart = alt.Chart(data).mark_bar().encode(
        x="label",
        y="value",
        column="metric"
    ).resolve_scale(y="independent")
    chart.save(output_pdf)


if __name__ == "__main__":
    typer.run(compare_cluster_stats)

# print(sys.argv)
