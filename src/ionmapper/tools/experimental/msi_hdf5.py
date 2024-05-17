import argparse
import itertools

import awkward
import h5py
import numpy as np
from tqdm import tqdm
from ionmapper.persistence import ImzmlReader, ImzmlModeEnum, ImzmlReadFile


def batched(iterable, n):
    """See itertools.batched in Python 3.12, this is from the documentation:"""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


class MsiHdf5:
    """
    Implementation of conversions of MSI data to and from HDF5 files.
    NOTE: the current version might not be the most efficient one, i.e. RAM usage might be high.

    Schema:
    /coordinates: (n_spectra, 2) array of coordinates
    /mz_arr/offsets: (n_spectra+1,) array of offsets
    /mz_arr/values: (n_data_points,) array of mz values
    /int_arr/offsets: (n_spectra+1,) array of offsets
    /int_arr/values: (n_data_points,) array of intensity values
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def write_imzml(self, imzml_reader: ImzmlReader) -> None:
        self.write_coordinates(imzml_reader.coordinates)
        batch_size = 1000
        n_spectra = imzml_reader.n_spectra
        if imzml_reader.imzml_mode == ImzmlModeEnum.PROCESSED:
            for spectra_ids in tqdm(np.array_split(np.arange(n_spectra), n_spectra // batch_size)):
                mz_arr, int_arr = imzml_reader.get_spectra(spectra_ids)
                self.append_awkward_2d(awkward.Array(mz_arr), "mz_arr")
                self.append_awkward_2d(awkward.Array(int_arr), "int_arr")
        else:
            mz_arr = imzml_reader.get_spectrum(0)[0]
            self.write_awkward_2d(awkward.Array([mz_arr]), "mz_arr")
            # TODO optimize as above
            for batch in batched(
                (imzml_reader.get_spectrum(i)[1] for i in tqdm(range(imzml_reader.n_spectra))), batch_size
            ):
                self.append_awkward_2d(awkward.Array(batch), "int_arr")

    # TODO if giving it another chance: make use of the following method
    def count_total_datapoints(self, imzml_reader: ImzmlReader) -> int:
        """Returns the total number of data points in the imzML file."""
        return sum(imzml_reader.get_spectrum_n_points(i_spectrum) for i_spectrum in range(imzml_reader.n_spectra))

    def write_coordinates(self, coordinates: np.ndarray) -> None:
        with h5py.File(self._path, "a") as file:
            file.create_dataset("coordinates", data=coordinates)

    def read_coordinates(self) -> np.ndarray:
        with h5py.File(self._path, "r") as file:
            return np.asarray(file["coordinates"][:])

    def append_awkward_2d(self, array: awkward.Array, group_name: str) -> None:
        offsets, values = self._encode_awkward_2d(array)
        with h5py.File(self._path, "a") as file:
            if f"{group_name}/offsets" not in file:
                file.create_dataset(f"{group_name}/offsets", data=offsets, chunks=(10000,), maxshape=(None,))
                file.create_dataset(f"{group_name}/values", data=values, chunks=(10000,), maxshape=(None,))
                return

            offsets_len_old = file[f"{group_name}/offsets"].shape[0]
            offsets_len_new = offsets_len_old + len(offsets)
            file[f"{group_name}/offsets"].resize((offsets_len_new,))
            file[f"{group_name}/offsets"][offsets_len_old:] = offsets

            values_len_old = file[f"{group_name}/values"].shape[0]
            values_len_new = values_len_old + len(values)
            file[f"{group_name}/values"].resize((values_len_new,))
            file[f"{group_name}/values"][values_len_old:] = values

    def write_awkward_2d(self, array: awkward.Array, group_name: str) -> None:
        offsets, values = self._encode_awkward_2d(array)
        with h5py.File(self._path, "a") as file:
            file.create_dataset(f"{group_name}/offsets", data=offsets, chunks=True, maxshape=(None,))
            file.create_dataset(f"{group_name}/values", data=values, chunks=True, maxshape=(None,))

    def _encode_awkward_2d(self, array: awkward.Array) -> tuple[np.ndarray, np.ndarray]:
        lengths = np.array([len(row) for row in array], dtype=np.int64)
        offsets = np.cumsum(lengths, dtype=np.int64) - lengths[0]
        offsets = np.concatenate([offsets, np.array([offsets[-1] + lengths[-1]], dtype=np.int64)])
        values = np.concatenate(array)
        return offsets, values

    def read_awkward_2d(self, group_name: str) -> awkward.Array:
        with h5py.File(self._path, "r") as file:
            offsets = file[f"{group_name}/offsets"][:]
            values = file[f"{group_name}/values"][:]
            return awkward.Array([values[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)])


def main_imzml_to_hdf5(input_file: str, output_file: str) -> None:
    with ImzmlReadFile(input_file).reader() as reader:
        writer = MsiHdf5(output_file)
        writer.write_imzml(reader)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    sub.required = True
    parser_imzml_to_hdf5 = sub.add_parser("imzml_to_hdf5", help="Converts an imzML file to an HDF5 file.")
    parser_imzml_to_hdf5.add_argument("input_file", type=str, help="The input imzML file.")
    parser_imzml_to_hdf5.add_argument("output_file", type=str, help="The output HDF5 file.")
    args = parser.parse_args()
    if args.command == "imzml_to_hdf5":
        main_imzml_to_hdf5(input_file=args.input_file, output_file=args.output_file)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
