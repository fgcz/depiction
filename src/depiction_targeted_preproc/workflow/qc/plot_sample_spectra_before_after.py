import cyclopts
import polars as pl
from pathlib import Path

from depiction.persistence import ImzmlReadFile
from depiction.visualize.plot_mass_spectrum import PlotMassSpectrum

app = cyclopts.App()


@app.default
def qc_plot_sample_spectra_before_after(
    imzml_baseline: Path,
    imzml_calib: Path,
    mass_list: Path,
    output_pdf: Path,
) -> None:
    import holoviews as hv

    hv.extension("matplotlib")  # <- todo where to handle this nicely?

    baseline = ImzmlReadFile(imzml_baseline)
    calib = ImzmlReadFile(imzml_calib)
    # TODO this could be improved in the future
    spectra_indices = range(0, baseline.n_spectra, baseline.n_spectra // 5)

    mass_list_df = pl.read_csv(mass_list)

    plotter = PlotMassSpectrum()

    with baseline.reader() as baseline_reader, calib.reader() as calib_reader:
        for spectrum_index in spectra_indices:
            mz_arr_baseline, int_arr_baseline = baseline_reader.get_spectrum(spectrum_index)
            mz_arr_calib, int_arr_calib = calib_reader.get_spectrum(spectrum_index)

            df_spectrum = pl.concat(
                [
                    pl.DataFrame({"m/z": mz_arr_baseline, "intensity": int_arr_baseline, "variant": "baseline"}),
                    pl.DataFrame({"m/z": mz_arr_calib, "intensity": int_arr_calib, "variant": "calibrated"}),
                ]
            )

            plot_s_base = plotter.visualize_profile(df_spectrum, mass_list_df.rename({"mass": "m/z"}))
            plot_s_calib = plotter.visualize_profile(df_spectrum, mass_list_df.rename({"mass": "m/z"}))

            plot = plot_s_base.opts(color="") * plot_s_calib

            # plot = plotter.get_layer_peaks(df_spectrum, extra_cols=["variant"]).overlay("variant")
            plot *= plotter.get_layer_refs(mass_list_df.rename({"mass": "m/z"}))

            mz_ref = mass_list_df["mass"].median()
            plot = plot.opts(xlim=(mz_ref - 20, mz_ref + 20), width=800, height=300)

            hv.save(plot, output_pdf, fmt="pdf")

            print(df_spectrum)
            break


if __name__ == "__main__":
    app()
