import altair as alt
import dash_bootstrap_components as dbc
import dash_vega_components as dvc
import h5py
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, callback, dcc, html

from depiction.image.sparse_image_2d import SparseImage2d

# ugly data loading for quick testing
df = (
    pd.read_csv("/Users/leo/Downloads/120-plex-EC-cohort_Updated.csv")
    .rename(columns={"PC-MT (M+H)+": "Mass"})
    .drop(columns=["No."])
)
h5_path = "/Users/leo/Documents/TmpData/20240122-out07-tonsil/data.hdf5"
with h5py.File(h5_path, "r") as file:
    sparse_values = file["ion_images/images_2d"][:]
    print(dict(file["ion_images/images_2d"].attrs).keys())
    channel_names = file["ion_images/images_2d"].attrs["labels"]
    offset = file["ion_images/images_2d"].attrs["offset"]
    coordinates = file["calibration/coordinates"][:]

    def better_from_dense(x):
        # TODO refactor the method in the future
        # flip y-axis
        x = np.flip(x, axis=0)

        dense_1 = SparseImage2d.from_dense_array(x, channel_names=channel_names, offset=offset)
        # scan for zeros
        is_missing = np.all(dense_1.sparse_values == 0, axis=-1)

        # create a new instance
        return SparseImage2d(
            dense_1.sparse_values[~is_missing],
            coordinates=dense_1.sparse_coordinates[~is_missing],
            channel_names=dense_1.channel_names,
        )

    # ion_images = SparseImage2d.from_dense_array(sparse_values, channel_names=channel_names, offset=offset)
    ion_images = better_from_dense(sparse_values)


# the actual app
alt.data_transformers.enable("vegafusion")

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
    [
        html.H3("Select a marker of interest"),
        dcc.Dropdown(df["Marker"].tolist(), id="marker-dropdown", value=df["Marker"].iloc[0], clearable=False),
        # dash_table.DataTable(
        #    id="table",
        #    columns=[{"name": i, "id": i} for i in df.columns],
        #    data=df.to_dict("records"),
        #    style_table={"height": "300px", "overflowY": "auto"},
        #    active_cell={"row": 0, "column": 0},
        # ),
        html.H3("Visualization"),
        dvc.Vega(id="altair-chart", opt={"actions": False}),
    ],
)


########@callback(Output("table", "style_data_conditional"), Input("table", "active_cell"))
########def highlight_selected_row(active_cell):
########    if not active_cell:
########        return []
########    return [{"if": {"row_index": active_cell["row"]}, "backgroundColor": "#3D9970"}]


# @callback(Output("altair-chart", "spec"), Input("table", "active_cell"))
# def display_altair_chart(active_cell):
@callback(Output("altair-chart", "spec"), Input("marker-dropdown", "value"))
def display_altair_chart(marker_name):
    # i_channel = active_cell["row"]
    i_channel = df[df["Marker"] == marker_name].index[0]
    coordinates_2d = ion_images.sparse_coordinates
    channel_df = pd.DataFrame(
        {
            "x": coordinates_2d[:, 0],
            "y": coordinates_2d[:, 1],
            "intensity": ion_images.sparse_values[:, i_channel],
        },
    )

    brush = alt.selection_interval(encodings=["x", "y"])

    color_intensity = alt.Color("intensity", scale=alt.Scale(scheme="viridis", domainMid=0))

    im_width, im_height = ion_images.dimensions
    im_aspect = im_width / im_height
    plot_img = (
        alt.Chart(channel_df)
        .mark_rect()
        .encode(
            x=alt.X("x:O", axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("y:O", axis=alt.Axis(labels=False, ticks=False)),
            color=color_intensity,
        )
        .properties(width=500, height=500 * im_aspect)
        .add_params(brush)
    )

    plot_hist = (
        alt.Chart(channel_df)
        .mark_bar()
        .encode(
            alt.X("intensity:Q").bin(maxbins=50),
            y="count()",
            color=color_intensity,
        )
        .properties(width=800, height=500 * im_aspect)
        .transform_filter(brush)
    )

    return (plot_img | plot_hist).to_dict(format="vega")

    # fake_x = np.linspace(0, 2, 100)
    # vis_data = pd.DataFrame({"x": fake_x, "y": fake_x + label * np.sin(fake_x)})
    # chart = alt.Chart(vis_data).mark_line().encode(x="x", y="y").interactive()
    # return chart.to_dict()


if __name__ == "__main__":
    app.run_server(debug=True)
