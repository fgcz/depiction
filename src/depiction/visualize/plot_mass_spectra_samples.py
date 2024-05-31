import polars as pl
import altair as alt


class PlotMassSpectraSamples:
    """
    Scope:
    - non-interactive plotting (currently the only good solution for interactive plotting of mass spectra in my opinion
      is bokeh/hvplot because of box-zoom missing from altair, and better performance than plotly)
    - showing multiple distinct mass ranges in a grid
    - showing multiple mass spectra in a plot
    - annotating references in a plot

    Why:
    - to compare the effects of calibration for some actual examples

    Design:
    - we use altair for consistency with peak density plots
    - all data is added together in a single DataFrame with columns:
      - m/z: mass-to-charge ratio
      - int: intensity (for references we set this to 1 TODO or NA?)
      - spectrum_name: name of the spectrum
      - label_name: name of the label (if any)
    """

    def __init__(self, data_df: pl.DataFrame) -> None:
        self._data_df = data_df
        self._charts = []

    def plot_spectrum(self, spectrum_name: str):
        data = self._data_df.filter(spectrum_name=spectrum_name)
        chart = (
            alt.Chart(data)
            .mark_rule()
            .encode(
                x="m/z:Q",
                y="int:Q",
                color="label_name:N",
            )
        )
        self._charts.append(chart)


# def plot_profile(self, label: str = "profile"):
#    df = self.mass_spectra.filter(pl.col("label") == label)
