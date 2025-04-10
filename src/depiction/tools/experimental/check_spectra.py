from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st
import polars as pl
from matplotlib import pyplot as plt

from depiction_io.parallel_ops import ParallelConfig
from depiction.persistence import ImzmlReadFile
from depiction.tools.generate_ion_image import GenerateIonImage


# @st.cache_data
# def get_imzml_reader(file: Path):
#    file = ImzmlReadFile(file)
#    return file.get_reader()
#
# @st.cache_data
# def get_ion_image(file: Path, )
#
#
# def main():
#    imzml_reader = get_imzml_reader(Path("/Users/leo/Documents/TmpData/20241016_01_timsconvert_data/work-2/S758504_CHCA_20um_bottom_timsconvert/calibrated.imzML"))
#    st.text(f"Loaded imzML file: {imzml_reader.imzml_path}")
#    st.text(f"Num spectra new: {imzml_reader.n_spectra}")


@st.cache_data
def get_ion_image(imzml_path: Path, mass: float, tolerance: float):
    read_file = ImzmlReadFile(imzml_path)
    gen_img = GenerateIonImage(ParallelConfig(n_jobs=8))
    return gen_img.generate_ion_images_for_file(
        input_file=read_file, mz_values=[mass], tol=[tolerance], channel_names=True
    )


@st.cache_data
def get_mass_list(path):
    return pl.read_csv(path)


def get_spectrum_indices(coordinates_flat, x0, x1, y0, y1):
    contained = (
        (x0 <= coordinates_flat.x)
        & (coordinates_flat.x <= x1)
        & (y0 <= coordinates_flat.y)
        & (coordinates_flat.y <= y1)
    )
    return np.where(contained.values)[0]


def get_spectrum_index(coordinates_flat, x0, x1, y0, y1):
    indices = get_spectrum_indices(coordinates_flat, x0, x1, y0, y1)
    if len(indices) == 0:
        return None
    return indices[0]


@st.cache_data
def get_spectrum(imzml_path: Path, spectrum_index: int):
    read_file = ImzmlReadFile(imzml_path)
    with read_file.reader() as reader:
        return reader.get_spectrum(spectrum_index)


def main():
    imzml_path = Path(
        "/Users/leo/Documents/TmpData/20241016_01_timsconvert_data/work-2/S758504_CHCA_20um_bottom_timsconvert/calibrated.imzML"
    )
    imzml_raw_path = Path(
        "/Users/leo/Documents/TmpData/20241016_01_timsconvert_data/work-2/S758504_CHCA_20um_bottom_timsconvert/raw.imzML"
    )
    # angiotensin
    img = get_ion_image(imzml_path, 1046.5, 0.5)
    st.text(str(img))

    mass_list = get_mass_list(
        Path(
            "/Users/leo/Documents/TmpData/20241016_01_timsconvert_data/work-2/S758504_CHCA_20um_bottom_timsconvert/mass_list.visualization.csv"
        )
    )
    st.write(mass_list)

    fig = px.imshow(img.data_spatial.isel(c=0))
    event_dict = st.plotly_chart(fig, on_select="rerun", selection_mode="box")

    if not event_dict["selection"]["box"]:
        st.write("Please select a ROI in the image.")
        return

    x0, x1 = event_dict["selection"]["box"][0]["x"]
    y0, y1 = event_dict["selection"]["box"][0]["y"]

    # get back a single spectrum index (TODO improve)
    spectrum_index = get_spectrum_index(img.coordinates_flat, x0, x1, y0, y1)
    if spectrum_index is None:
        st.write("No spectrum found in the selected region.")
        return

    # find points in this
    st.write(spectrum_index)

    # get the spectrum
    mz_arr, int_arr = get_spectrum(imzml_path, spectrum_index)
    int_arr = np.log(int_arr)
    fig = plt.figure()
    plt.vlines(mass_list["mass"].to_list(), -0.1 * int_arr.max(), 0, linestyle="--", color="red")
    plt.vlines(mz_arr, 0, int_arr)
    st.pyplot(fig)

    mz_arr, int_arr = get_spectrum(imzml_raw_path, spectrum_index)
    fig = plt.figure()
    int_arr = np.log(int_arr)
    plt.vlines(mass_list["mass"].to_list(), -0.1 * int_arr.max(), 0, linestyle="--", color="red")
    plt.vlines(mz_arr, 0, int_arr)
    st.pyplot(fig)
    # fig = px.line(x=mz_arr, y=int_arr)
    # st.write(int_arr)
    # st.plotly_chart(fig)
    # st.line_chart(mz_arr, int_arr)

    # fig = plt.figure()
    # img.data_spatial.isel(c=0).plot.imshow(ax=plt.gca(), x="x", y="y", yincrease=False, cmap="binary")
    # event_dict = st.pyplot(fig)


if __name__ == "__main__":
    main()
