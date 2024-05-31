from pathlib import Path
from typing import Annotated

import altair as alt
import polars as pl
import typer
from typer import Option

from depiction.image.multi_channel_image import MultiChannelImage


# TODO when computing these statistics the big question is whether we should count 0 values as well?


def get_plot_df(image: MultiChannelImage) -> pl.DataFrame:
    im_data = image.data_flat

    im_data = im_data.where(im_data > 0)  # TODO removing 0 values makes a big difference on both metrics

    cv = im_data.std(dim="i") / im_data.mean(dim="i")
    # quartile coefficient of dispersion
    q1 = im_data.quantile(0.25, dim="i")
    q3 = im_data.quantile(0.75, dim="i")
    cqdisp = (q3 - q1) / (q3 + q1)

    return pl.DataFrame({"cv": cv.values, "cq": cqdisp.values, "label": image._data.coords["c"].values})


def plot_marker_presence_cv(image: MultiChannelImage, out_path: Path) -> None:
    # plot_df = pl.concat([
    #    get_cv_df(image_baseline).with_columns(variant=pl.lit("baseline_adj")),
    #    get_cv_df(image_calib).with_columns(variant=pl.lit("calibrated"))
    # ])
    plot_df = get_plot_df(image)

    # TODO figure out what to plot exactly

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            # TODO
            # x=alt.X("cv:Q"),
            x=alt.X("cq:Q"),
            y=alt.Y("label:N").sort("-x"),
        )
    )
    chart.save(out_path)


def qc_plot_marker_presence_cv(
    image_hdf5: Annotated[Path, Option()],
    output_pdf: Annotated[Path, Option()],
) -> None:
    image = MultiChannelImage.read_hdf5(image_hdf5)
    plot_marker_presence_cv(image=image, out_path=output_pdf)


if __name__ == "__main__":
    typer.run(qc_plot_marker_presence_cv)
