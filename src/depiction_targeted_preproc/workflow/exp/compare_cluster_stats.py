import altair as alt
import cyclopts
import polars as pl
from pathlib import Path


def load_data(csv_paths: list[Path]) -> pl.DataFrame:
    collect = []
    for csv_path in csv_paths:
        df = pl.read_csv(csv_path)
        collect.append(df.with_columns(label=pl.lit(csv_path.parent.name)))
    return pl.concat(collect)


app = cyclopts.App()


@app.default
def compare_cluster_stats(input_csv_path: list[Path], output_pdf: Path) -> None:
    data = load_data(input_csv_path)
    chart = alt.Chart(data).mark_bar().encode(x="label", y="value", column="metric").resolve_scale(y="independent")
    chart.save(output_pdf)


if __name__ == "__main__":
    app()
