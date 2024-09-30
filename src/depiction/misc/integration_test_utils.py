from typing import Optional

from numpy.typing import ArrayLike

from depiction.persistence import ImzmlModeEnum, ImzmlWriteFile


class IntegrationTestUtils:
    @staticmethod
    def populate_test_file(
        path: str,
        mz_arr_list: list[ArrayLike] | ArrayLike,
        int_arr_list: list[ArrayLike] | ArrayLike,
        imzml_mode: ImzmlModeEnum,
        coordinates_list: Optional[list[tuple[int, ...]]] = None,
    ) -> None:
        if coordinates_list is None:
            coordinates_list = [(0, i) for i in range(len(mz_arr_list))]
        with ImzmlWriteFile(path, imzml_mode).writer() as writer:
            for mz_arr, int_arr, coordinates in zip(mz_arr_list, int_arr_list, coordinates_list):
                writer.add_spectrum(mz_arr, int_arr, coordinates)
