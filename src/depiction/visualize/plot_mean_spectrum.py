from pathlib import Path

import altair as alt
import polars as pl
import typer

from depiction.spectrum.evaluate_bins import EvaluateBins
from depiction.spectrum.evaluate_mean_spectrum import EvaluateMeanSpectrum
from depiction.parallel_ops import ParallelConfig
from depiction.persistence import ImzmlReadFile


def get_mean_spectrum(read_file: ImzmlReadFile, parallel_config: ParallelConfig) -> pl.DataFrame:
    with read_file.reader() as reader:
        mz_arr_first = reader.get_spectrum_mz(0)
    eval_bins = EvaluateBins.from_mz_values(mz_arr_first)
    evaluate_mean = EvaluateMeanSpectrum(parallel_config=parallel_config, eval_bins=eval_bins)
    mean_mz_arr, mean_int_arr = evaluate_mean.evaluate_file(read_file)
    return pl.DataFrame({"mz": mean_mz_arr, "intensity": mean_int_arr})


def get_sample_spec_data(read_file: ImzmlReadFile, n_specs: int):
    spec_indices = set(sorted(range(0, read_file.n_spectra, read_file.n_spectra // n_specs)))
    if len(spec_indices) > n_specs:
        spec_indices = sorted(spec_indices)[:n_specs]
    if len(spec_indices) != n_specs:
        raise ValueError(f"Could not get {n_specs} unique spectra indices.")
    collect = []
    with read_file.reader() as reader:
        for i_spectrum in spec_indices:
            mz_arr, int_arr = reader.get_spectrum(i_spectrum)
            collect.append({"mz": pl.Series(mz_arr), "intensity": pl.Series(int_arr), "i_spectrum": i_spectrum})
    return pl.DataFrame(collect).explode(["mz", "intensity"])


def main(input_imzml: Path, output_pdf: str, n_jobs: int = 20) -> None:
    read_file = ImzmlReadFile(input_imzml)
    parallel_config = ParallelConfig(n_jobs=n_jobs)
    mean_data = get_mean_spectrum(read_file, parallel_config=parallel_config)
    n_specs = 3
    sample_spec_data = get_sample_spec_data(read_file, n_specs=n_specs)
    title = f"Mean spectrum for {input_imzml.name} (n_bins={len(mean_data)})"
    chart_mean_spec = (
        alt.Chart(mean_data).mark_line().encode(x="mz", y="intensity").properties(width=800, height=300, title=title)
    )
    chart_sample_specs = (
        alt.Chart(sample_spec_data)
        .mark_point()
        .encode(x="mz", y="intensity", color="i_spectrum:N")
        .properties(width=800, height=300, title=f"{n_specs} sample spectra")
    )
    chart = (chart_mean_spec & chart_sample_specs).resolve_scale(x="shared")
    chart.save(output_pdf)


if __name__ == "__main__":
    typer.run(main)
