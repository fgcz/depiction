import altair as alt
import cyclopts
import numpy as np
import polars as pl
import scipy.ndimage
from pathlib import Path

from depiction.image.multi_channel_image import MultiChannelImage


def get_spatial_coherence_scores(image: MultiChannelImage) -> pl.DataFrame:
    fg_mask = image.fg_mask.values
    labels = image.data_spatial.coords["c"].values
    scores = []

    for c in labels:
        channel_2d = image.data_spatial.sel(c=c).values
        local_mean = scipy.ndimage.uniform_filter(channel_2d.astype(float), size=5)

        values_fg = channel_2d[fg_mask]
        local_mean_fg = local_mean[fg_mask]

        if len(values_fg) < 10:
            scores.append(float("nan"))
        else:
            scores.append(float(np.corrcoef(values_fg, local_mean_fg)[0, 1]))

    return pl.DataFrame({"label": labels, "spatial_coherence": scores})


def plot_spatial_distribution_qc(image: MultiChannelImage, out_path: Path) -> None:
    plot_df = get_spatial_coherence_scores(image)

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("spatial_coherence:Q").scale(domain=[0, 1]),
            y=alt.Y("label:N").sort("-x"),
        )
    )
    chart.save(out_path)


app = cyclopts.App()


@app.default
def qc_plot_spatial_distribution_qc(image_hdf5: Path, output_pdf: Path) -> None:
    image = MultiChannelImage.read_hdf5(image_hdf5)
    plot_spatial_distribution_qc(image=image, out_path=output_pdf)


if __name__ == "__main__":
    app()
