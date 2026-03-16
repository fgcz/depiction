import altair as alt
import cyclopts
import numpy as np
import polars as pl
import scipy.ndimage
from pathlib import Path

from depiction.image.multi_channel_image import MultiChannelImage

_DEFAULT_SCALES = [3, 7, 15, 31, 63]


def get_spatial_coherence_scores(
    image: MultiChannelImage,
    scales: list[int] | None = None,
    log_scale: bool = True,
) -> pl.DataFrame:
    """Compute a multi-scale spatial coherence score for each channel.

    For each scale (filter size), the score is the Pearson correlation between
    foreground pixel intensities and their local neighborhood mean. The final
    per-channel score is the area under the coherence-vs-scale curve, normalized
    by the scale range, giving a single value in roughly [-1, 1].

    Scales larger than the image are silently dropped. Channels with fewer than
    10 foreground pixels receive a NaN score.

    Args:
        image: Multi-channel image to score.
        scales: Filter sizes (pixels) to evaluate. Defaults to ``[3, 7, 15, 31, 63]``,
            which spans roughly one decade on a ~500×500 image.
        log_scale: If True (default), integrate over log(scale) so each decade
            of spatial scale contributes equally. If False, integrate linearly.

    Returns:
        DataFrame with columns ``label`` (channel name) and ``spatial_coherence``.
    """
    if scales is None:
        scales = _DEFAULT_SCALES

    fg_mask = image.fg_mask.values
    labels = image.data_spatial.coords["c"].values

    # Drop scales that exceed the smallest image dimension to avoid errors on small images.
    max_size = min(image.data_spatial.sizes["y"], image.data_spatial.sizes["x"])
    valid_scales = [s for s in scales if s < max_size]
    if len(valid_scales) < 2:
        # Fall back to a single scale of 3 (or 1 if even that is too big) so we
        # can still return a score rather than crashing.
        valid_scales = [min(3, max_size - 1)]

    x_values = np.log(valid_scales) if log_scale else np.array(valid_scales, dtype=float)

    scores = []
    for c in labels:
        channel_2d = image.data_spatial.sel(c=c).values.astype(float)
        values_fg = channel_2d[fg_mask]

        if len(values_fg) < 10:
            scores.append(float("nan"))
            continue

        correlations = []
        for size in valid_scales:
            local_mean = scipy.ndimage.uniform_filter(channel_2d, size=size)
            corr = float(np.corrcoef(values_fg, local_mean[fg_mask])[0, 1])
            correlations.append(corr)

        if len(valid_scales) == 1:
            auc = correlations[0]
        else:
            auc = float(np.trapz(correlations, x=x_values) / (x_values[-1] - x_values[0]))
        scores.append(auc)

    return pl.DataFrame({"label": labels, "spatial_coherence": scores})


def plot_spatial_distribution_qc(image: MultiChannelImage, out_pdf: Path, out_csv: Path) -> None:
    """Compute multi-scale spatial coherence scores, save the table to ``out_csv`` and the plot to ``out_pdf``."""
    plot_df = get_spatial_coherence_scores(image)
    plot_df.write_csv(out_csv)

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("spatial_coherence:Q").scale(domain=[0, 1]),
            y=alt.Y("label:N").sort("-x"),
        )
    )
    chart.save(out_pdf)


app = cyclopts.App()


@app.default
def qc_plot_spatial_distribution_qc(image_hdf5: Path, output_pdf: Path, output_csv: Path) -> None:
    image = MultiChannelImage.read_hdf5(image_hdf5)
    plot_spatial_distribution_qc(image=image, out_pdf=output_pdf, out_csv=output_csv)


if __name__ == "__main__":
    app()
