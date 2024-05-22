import argparse
import math
from typing import Callable

import seaborn
import vispy.color
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import napari
import napari.layers
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import (
    QGridLayout,
    QComboBox,
    QPushButton,
    QWidget,
    QTableView,
)
from napari_matplotlib import HistogramWidget

from depiction.image.sparse_image_2d import SparseImage2d
import h5py


class GeneralMultiSelectionWidget(QWidget):
    """A simple multi-selection widget based on a table that shows the selected items, a dropdown that allows
    to pick elements, and a pair of add/remove buttons to add or remove what is selected in the dropdown.
    """

    def __init__(
        self,
        entries_df: pd.DataFrame,
        dropdown_column: str = None,
        listeners: list[Callable] = None,
    ) -> None:
        super().__init__()

        self._entries_df = entries_df
        self._selection = np.zeros(len(entries_df), dtype=bool)
        self._dropdown_column = dropdown_column if dropdown_column is not None else entries_df.columns[0]
        self._listeners = listeners if listeners is not None else []

        table_view = QTableView()
        table_model = QStandardItemModel()
        table_model.setColumnCount(len(entries_df.columns))
        table_model.setHorizontalHeaderLabels(entries_df.columns)
        for i, row in entries_df.iterrows():
            for j, value in enumerate(row):
                table_model.setItem(i, j, QStandardItem(str(value)))
        table_view.setModel(table_model)
        self._table_model = table_model

        self._dropdown_widget = QComboBox()
        self._dropdown_widget.addItems(entries_df[self._dropdown_column].tolist())

        add_button = QPushButton("Add")
        add_button.clicked.connect(self.on_add)

        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.on_remove)

        layout = QGridLayout()
        layout.addWidget(table_view, 0, 0, 1, 3)
        layout.addWidget(self._dropdown_widget, 1, 0)
        layout.addWidget(add_button, 1, 1)
        layout.addWidget(remove_button, 1, 2)

        self.setLayout(layout)

    def on_add(self) -> None:
        selected_idx = self._dropdown_widget.currentIndex()
        if self._selection[selected_idx]:
            return
        self._selection[selected_idx] = True
        self._update_table()
        self._notify_listeners()

    def on_remove(self) -> None:
        selected_idx = self._dropdown_widget.currentIndex()
        if not self._selection[selected_idx]:
            return
        self._selection[selected_idx] = False
        self._update_table()
        self._notify_listeners()

    def add_listener(self, listener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener) -> None:
        self._listeners.remove(listener)

    def _notify_listeners(self) -> None:
        for listener in self._listeners:
            listener(self)

    @property
    def selected_indices(self) -> NDArray[int]:
        view = self._selection.view()
        view.flags.writeable = False
        return np.where(view)[0]

    @property
    def selected_rows(self) -> pd.DataFrame:
        return self._entries_df.iloc[self.selected_indices]

    def _update_table(self) -> None:
        for i, is_selected in enumerate(self._selection):
            self._table_model.item(i, 0).setCheckState(Qt.Checked if is_selected else Qt.Unchecked)


class VisualizationSelectionWidget(QWidget):
    def __init__(self, markers, file_variants) -> None:
        super().__init__()

        self._markers = markers
        self._file_variants = file_variants

        self._marker_selection = GeneralMultiSelectionWidget(markers, listeners=[self.on_update_selection])
        self._file_variant_selection = GeneralMultiSelectionWidget(file_variants, listeners=[self.on_update_selection])

        layout = QGridLayout()
        layout.addWidget(self._marker_selection, 0, 0)
        layout.addWidget(self._file_variant_selection, 0, 1)

        self.setLayout(layout)

    def on_update_selection(self, widget: GeneralMultiSelectionWidget) -> None:
        # get the selected indices
        pass

        # update the selection
        # TODO call main viewer


class ChannelSelectorWidget(QComboBox):
    def __init__(self, main_viewer: "NapariSparseImage2dViewer") -> None:
        super().__init__()
        self._main_viewer = main_viewer
        self.addItems(main_viewer.channel_names)
        self.currentIndexChanged.connect(self.on_update_selection)

    def on_update_selection(self, new_index: int) -> None:
        self._main_viewer.show_channel(new_index)


class GeneralSelectorWidget(QWidget):
    def __init__(self, main_viewer: "NapariSparseImage2dViewer") -> None:
        super().__init__()
        self._main_viewer = main_viewer

        layout = QGridLayout()
        self._channel_selector = ChannelSelectorWidget(main_viewer)
        layout.addWidget(self._channel_selector, 0, 0)

        arrange_grid_button = QPushButton("Arrange Grid")
        arrange_grid_button.clicked.connect(self.on_arrange_grid)
        layout.addWidget(arrange_grid_button, 1, 0)

        multi_sel = GeneralMultiSelectionWidget(pd.DataFrame({"channel": main_viewer.channel_names}))
        # multi_sel = GeneralMultiSelectionWidget(main_viewer.channel_names)
        layout.addWidget(multi_sel, 2, 0)

        self.setLayout(layout)

    def on_arrange_grid(self) -> None:
        # TODO this could be more expected from users, but it could also mess up someone's manual selection
        # self._main_viewer.show_channel(self._channel_selector.currentIndex())
        self._main_viewer.arrange_visible_layers_in_grid()


class NapariSparseImage2dViewer:
    def __init__(self) -> None:
        self._images = []
        self._viewer = napari.Viewer()
        self._channel_data = pd.DataFrame({"image_name": [], "channel_name": []})

    def add_image(self, image: SparseImage2d, name: str) -> None:
        self._images.append({"image": image, "name": name})
        self._channel_data = pd.concat(
            [
                self._channel_data,
                pd.DataFrame(
                    [{"image_name": name, "channel_name": channel_name} for channel_name in image.channel_names]
                ),
            ]
        ).reset_index()

    def add_all_images(self, file: h5py.File) -> None:
        for group_name, group in file.items():
            if SparseImage2d.is_valid_hdf5(group):
                self.add_image(SparseImage2d.load_from_hdf5(group), group_name)

    @property
    def channel_names(self) -> list[str]:
        return self._channel_data["channel_name"].unique().tolist()

    def show_channel(self, channel_idx: int) -> None:
        # identify the relevant channel indices
        self.channel_names[channel_idx]
        relevant_channels = self._channel_data.query("channel_name == @channel_name").index.tolist()
        print("relevant_channels", relevant_channels)

        # apply the selection
        for i_layer, layer in enumerate(self._viewer.layers):
            layer.visible = i_layer in relevant_channels

    def arrange_visible_layers_in_grid(self) -> None:
        visible_layers = [layer for layer in self._viewer.layers if layer.visible]
        n_visible_layers = len(visible_layers)

        n_per_row = 6
        math.ceil(n_visible_layers / n_per_row)
        n_cols = min(n_per_row, n_visible_layers)

        # in principle, we would assume the same height and width for all images, but to make it slightly more
        # robust, here we take the max of each over the visible layers
        max_height, max_width = 0, 0
        for layer in visible_layers:
            height = layer.data.shape[0]
            width = layer.data.shape[1]
            max_height = max(max_height, height)
            max_width = max(max_width, width)

        # add some minor padding
        max_height += 5
        max_width += 5

        # now arrange the layers
        for i_layer, layer in enumerate(visible_layers):
            i_row = i_layer // n_cols
            i_col = i_layer % n_cols

            # compute the position of the layer
            x = i_col * max_width
            y = i_row * max_height

            # set the position
            layer.translate = [y, x]

    def _setup_napari_viewer(self) -> None:
        selector_widget = GeneralSelectorWidget(self)
        self._viewer.window.add_dock_widget(selector_widget, area="left")
        histogram_widget = HistogramWidget(self._viewer)
        histogram_widget.n_layers_input = range(1, 10)
        self._viewer.window.add_dock_widget(histogram_widget, area="right")

    def run(self) -> None:
        self._setup_napari_viewer()
        cmap = vispy.color.Colormap(seaborn.color_palette("mako"))

        for image_data in self._images:
            image = image_data["image"]
            for i_channel, channel_name in enumerate(image.channel_names):
                layer = napari.layers.Image(
                    data=image.get_dense_array()[..., i_channel],
                    colormap=("mako", cmap),
                    rgb=False,
                    name=f"{image_data['name']} @ {channel_name}",
                )
                self._viewer.add_layer(layer)
        napari.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_file", type=str)
    args = parser.parse_args()

    viewer = NapariSparseImage2dViewer()
    with h5py.File(args.hdf5_file, "r") as file:
        viewer.add_all_images(file)
    viewer.run()


if __name__ == "__main__":
    main()
