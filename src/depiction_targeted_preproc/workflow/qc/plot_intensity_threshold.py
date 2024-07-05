from __future__ import annotations

from pathlib import Path

import cyclopts
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from numpy.typing import NDArray
from tqdm import tqdm

from depiction.image.multi_channel_image import MultiChannelImage

app = cyclopts.App()


@app.default
def main(image_hdf5: Path, output_all_pixels_pdf: Path, output_foreground_pixels_pdf: Path) -> None:
    image = MultiChannelImage.read_hdf5(image_hdf5)
    data_flat = image.data_flat.values.ravel()

    # compute thresholds to evaluate
    v_min = 0
    v_max = np.percentile(data_flat, 99)
    thresholds = np.logspace(np.log10(v_min + 1), np.log10(v_max + 1), 500) - 1
    plot_threshold_all_pixels(image=image, thresholds=thresholds, output_pdf=output_all_pixels_pdf)
    plot_threshold_foreground_only(image=image, thresholds=thresholds, output_pdf=output_foreground_pixels_pdf)


def plot_threshold_all_pixels(image: MultiChannelImage, thresholds: NDArray[float], output_pdf: Path) -> None:
    collect = []
    data_flat = image.data_flat
    for threshold in tqdm(thresholds):
        counts = (data_flat > threshold).sum("c")
        collect.append(
            {"threshold": threshold, "mean": counts.mean(), "p25": counts.quantile(0.25), "p75": counts.quantile(0.75)}
        )
    df = pl.DataFrame(collect)
    plt.figure()
    plt.title(f"Detected Targets by Intensity Threshold (N={np.prod(data_flat.shape):,})")
    plt.fill_between(df["threshold"], df["p25"], df["p75"], color="gray", alpha=0.5, label="p25-p75")
    plt.plot(df["threshold"], df["mean"], label="mean")
    plt.xlabel("Threshold")
    plt.ylabel("Detected Targets (agg. over pixels)")
    plt.grid()
    plt.legend()
    plt.savefig(output_pdf, bbox_inches="tight")


def plot_threshold_foreground_only(image: MultiChannelImage, thresholds: NDArray[float], output_pdf: Path) -> None:
    background_targets = 5
    threshold = 25
    data = image.data_spatial
    bg = (data > threshold).sum("c") <= background_targets
    fg = ~bg
    fg.plot.imshow(yincrease=False, cmap="gray")
    # TODO if this actually gets used it will have to be supplemented with information about the foreground mask
    collect = []
    data_fg_flat = data.where(fg).stack(i=("y", "x")).dropna("i", how="all")
    for threshold in tqdm(thresholds):
        counts = (data_fg_flat > threshold).sum("c")
        collect.append(
            {"threshold": threshold, "mean": counts.mean(), "p25": counts.quantile(0.25), "p75": counts.quantile(0.75)}
        )
    df = pl.DataFrame(collect)
    plt.figure()
    plt.title(f"Detected Targets by Intensity Threshold (N={np.prod(data_fg_flat.shape):,})")
    plt.fill_between(df["threshold"], df["p25"], df["p75"], color="gray", alpha=0.5, label="p25-p75")
    plt.plot(df["threshold"], df["mean"], label="mean")
    plt.xlabel("Threshold")
    plt.ylabel("Detected Targets (agg. over pixels)")
    plt.grid()
    plt.legend()
    plt.savefig(output_pdf, bbox_inches="tight")


if __name__ == "__main__":
    app()
