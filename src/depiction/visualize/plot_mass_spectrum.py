# TODO experimental for interactive visualization

from dataclasses import dataclass

import holoviews as hv
import polars as pl
from holoviews.core import ViewableTree


@dataclass
class PlotMassSpectrum:
    width: int = 800
    height: int = 300

    def get_layer_refs(self, df_references: pl.DataFrame) -> hv.NdOverlay:
        ds = hv.Dataset(df_references[["m/z", "label"]].to_pandas(), ["m/z", "label"])
        # return ds.to(hv.VLines, "m/z").opts(color="red", tools=["hover"], show_legend=False).overlay()
        return ds.to(hv.VLines, "m/z").opts(color="red", show_legend=False).overlay()

    def get_layer_profile(self, df_spectrum: pl.DataFrame, extra_cols: list[str] | None = None) -> hv.Curve:
        extra_cols = extra_cols or []
        ds = hv.Dataset(df_spectrum[["m/z", "intensity", *extra_cols]].to_pandas(), ["m/z", "intensity", *extra_cols])
        # return ds.to(hv.Curve, "m/z", "intensity").opts(color="black")
        return ds.to(hv.Curve, "m/z", "intensity")

    def get_layer_peaks(self, df_spectrum: pl.DataFrame, extra_cols: list[str] | None = None) -> hv.Spikes:
        extra_cols = extra_cols or []
        ds = hv.Dataset(df_spectrum[["m/z", "intensity", *extra_cols]].to_pandas(), ["m/z", "intensity", *extra_cols])
        return ds.to(hv.Spikes, "m/z", "intensity")

    def visualize_profile(self, df_spectrum: pl.DataFrame, df_references: pl.DataFrame) -> ViewableTree:
        layer_spec = self.get_layer_profile(df_spectrum)
        layer_refs = self.get_layer_refs(df_references)
        return (layer_refs * layer_spec).opts(width=self.width, height=self.height)

    def visualize_peaks(self, df_spectrum: pl.DataFrame, df_references: pl.DataFrame) -> ViewableTree:
        layer_spec = self.get_layer_peaks(df_spectrum)
        layer_refs = self.get_layer_refs(df_references)
        return (layer_refs * layer_spec).opts(width=self.width, height=self.height)
