import warnings
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

    @staticmethod
    def treat_warnings_as_error(test_case) -> None:
        """To be called from setUp."""
        # TODO reuse or refactor (and move out of integration_test_utils and rather a more general test_utils?)
        warnings_ctx = warnings.catch_warnings()
        warnings_ctx.__enter__()
        warnings.simplefilter("error")
        test_case.addCleanup(warnings_ctx.__exit__, None, None, None)
