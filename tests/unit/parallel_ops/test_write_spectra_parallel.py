import unittest
from functools import cached_property
from pathlib import Path
from unittest.mock import MagicMock, ANY, patch, call

from depiction.parallel_ops import (
    WriteSpectraParallel,
    ReadSpectraParallel,
)
from depiction.persistence import ImzmlModeEnum, ImzmlWriteFile
from depiction.tools.merge_imzml import MergeImzml


class TestWriteSpectraParallel(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.mock_config = MagicMock(name="mock_config", n_jobs=2, task_size=None, verbose=0)
        self.mock_split_modes_and_paths = [
            (
                ImzmlModeEnum.CONTINUOUS,
                [
                    Path("/dev/null/mock/continuous_0.imzML"),
                    Path("/dev/null/mock/continuous_1.imzML"),
                    Path("/dev/null/mock/continuous_2.imzML"),
                ],
            ),
            (
                ImzmlModeEnum.PROCESSED,
                [
                    Path("/dev/null/mock/processed_0.imzML"),
                    Path("/dev/null/mock/processed_1.imzML"),
                    Path("/dev/null/mock/processed_2.imzML"),
                ],
            ),
        ]

    @cached_property
    def mock_parallel(self) -> WriteSpectraParallel:
        return WriteSpectraParallel.from_config(self.mock_config)

    @patch("depiction.parallel_ops.write_spectra_parallel.TemporaryDirectory")
    @patch.object(WriteSpectraParallel, "_get_split_modes_and_paths")
    @patch.object(WriteSpectraParallel, "_write_transformed_chunked_operation")
    @patch.object(WriteSpectraParallel, "_merge_results")
    @patch.object(ReadSpectraParallel, "from_config")
    def test_map_chunked_to_files(
        self,
        mock_from_config,
        mock_merge_results,
        mock_write_transformed_chunked_operation,
        mock_get_split_modes_and_paths,
        mock_temporary_directory,
    ) -> None:
        mock_temporary_directory.return_value.__enter__.return_value = "/dev/null/tmpdir"
        mock_get_split_modes_and_paths.return_value = self.mock_split_modes_and_paths

        mock_read_file = MagicMock(name="mock_read_file", spec=[])
        mock_write_files = MagicMock(name="mock_write_files", spec=[])
        mock_operation = MagicMock(name="mock_operation", spec=[])

        self.mock_parallel.map_chunked_to_files(
            read_file=mock_read_file,
            write_files=mock_write_files,
            operation=mock_operation,
        )

        mock_get_split_modes_and_paths.assert_called_once_with(
            work_directory=Path("/dev/null/tmpdir"),
            spectra_indices=None,
            read_file=mock_read_file,
            write_files=mock_write_files,
        )
        mock_from_config.assert_called_once_with(config=self.mock_config)
        mock_from_config.return_value.map_chunked.assert_called_once_with(
            read_file=mock_read_file,
            operation=ANY,
            spectra_indices=None,
            pass_task_index=True,
        )
        passed_operation = mock_from_config.return_value.map_chunked.mock_calls[0].kwargs["operation"]
        passed_operation("x")
        mock_write_transformed_chunked_operation.assert_called_once_with(
            "x",
            operation=ANY,
            open_write_files=True,
            split_modes_and_paths=self.mock_split_modes_and_paths,
        )
        # TODO properly test correct partial function application - here or with integration test
        mock_merge_results.assert_called_once_with(
            split_modes_and_paths=self.mock_split_modes_and_paths,
            write_files=mock_write_files,
        )

    @patch("depiction.parallel_ops.write_spectra_parallel.TemporaryDirectory")
    @patch("depiction.parallel_ops.write_spectra_parallel.ImzmlWriteFile")
    @patch.object(WriteSpectraParallel, "map_chunked_to_files")
    def test_map_chunked_external_to_files(
        self, mock_map_chunked_to_files, mock_imzml_write_file, mock_temporary_directory
    ) -> None:
        mock_read_file = MagicMock(name="mock_read_file", spec=[])
        mock_write_files = [
            MagicMock(name="mock_write_files_0", imzml_file="test1.imzML"),
            MagicMock(name="mock_write_files_1", imzml_file="test2.imzML"),
        ]
        mock_operation = MagicMock(name="mock_operation")
        mock_spectra_indices = MagicMock(name="mock_spectra_indices")
        mock_bind_args = MagicMock(name="mock_bind_args")

        self.mock_parallel.map_chunked_external_to_files(
            read_file=mock_read_file,
            write_files=mock_write_files,
            operation=mock_operation,
            spectra_indices=mock_spectra_indices,
            bind_args=mock_bind_args,
        )

        mock_map_chunked_to_files.assert_called_once_with(
            read_file=mock_read_file,
            write_files=mock_write_files,
            operation=ANY,
            spectra_indices=mock_spectra_indices,
            bind_args=mock_bind_args,
            open_write_files=False,
        )

        mock_temporary_directory.return_value.__enter__.return_value = "/dev/null/tmpdir"
        mock_reader = MagicMock(name="mock_reader")
        passed_operation = mock_map_chunked_to_files.mock_calls[0].kwargs["operation"]
        passed_operation(mock_reader, [3, 4, 5], mock_write_files)
        mock_operation.assert_called_once_with(
            Path("/dev/null/tmpdir/tmp_file.imzML"),
            ["test1.imzML", "test2.imzML"],
        )
        mock_imzml_write_file.assert_called_once_with(
            Path("/dev/null/tmpdir/tmp_file.imzML"), imzml_mode=mock_reader.imzml_mode
        )

    def test_get_split_modes_and_paths_when_spectra_indices_none(self) -> None:
        self.mock_config.get_splits_count.return_value = 3
        split_modes_and_paths = self.mock_parallel._get_split_modes_and_paths(
            work_directory=Path("/dev/null/mock"),
            read_file=MagicMock(name="mock_read_file", n_spectra=20),
            write_files=[
                MagicMock(name="mock_write_file_1", imzml_mode=ImzmlModeEnum.CONTINUOUS),
                MagicMock(name="mock_write_file_2", imzml_mode=ImzmlModeEnum.PROCESSED),
            ],
            spectra_indices=None,
        )
        self.assertEqual(2, len(split_modes_and_paths))
        self.assertTupleEqual(
            (
                ImzmlModeEnum.CONTINUOUS,
                [
                    Path("/dev/null/mock/chunk_f0_t0.imzML"),
                    Path("/dev/null/mock/chunk_f0_t1.imzML"),
                    Path("/dev/null/mock/chunk_f0_t2.imzML"),
                ],
            ),
            split_modes_and_paths[0],
        )
        self.assertTupleEqual(
            (
                ImzmlModeEnum.PROCESSED,
                [
                    Path("/dev/null/mock/chunk_f1_t0.imzML"),
                    Path("/dev/null/mock/chunk_f1_t1.imzML"),
                    Path("/dev/null/mock/chunk_f1_t2.imzML"),
                ],
            ),
            split_modes_and_paths[1],
        )
        self.mock_config.get_splits_count.assert_called_once_with(n_items=20)

    def test_get_split_modes_and_paths_when_spectra_indices_present(self) -> None:
        self.mock_config.get_splits_count.return_value = 3
        split_modes_and_paths = self.mock_parallel._get_split_modes_and_paths(
            work_directory=Path("/dev/null/mock"),
            read_file=MagicMock(name="mock_read_file", n_spectra=20),
            write_files=[
                MagicMock(name="mock_write_file_1", imzml_mode=ImzmlModeEnum.CONTINUOUS),
                MagicMock(name="mock_write_file_2", imzml_mode=ImzmlModeEnum.PROCESSED),
            ],
            spectra_indices=[0, 1, 2, 3, 4],
        )
        self.assertEqual(2, len(split_modes_and_paths))
        self.assertTupleEqual(
            (
                ImzmlModeEnum.CONTINUOUS,
                [
                    Path("/dev/null/mock/chunk_f0_t0.imzML"),
                    Path("/dev/null/mock/chunk_f0_t1.imzML"),
                    Path("/dev/null/mock/chunk_f0_t2.imzML"),
                ],
            ),
            split_modes_and_paths[0],
        )
        self.assertTupleEqual(
            (
                ImzmlModeEnum.PROCESSED,
                [
                    Path("/dev/null/mock/chunk_f1_t0.imzML"),
                    Path("/dev/null/mock/chunk_f1_t1.imzML"),
                    Path("/dev/null/mock/chunk_f1_t2.imzML"),
                ],
            ),
            split_modes_and_paths[1],
        )
        self.mock_config.get_splits_count.assert_called_once_with(n_items=5)

    def test_write_transformed_chunked_operation_when_not_open_write_files(self) -> None:
        mock_reader = MagicMock(name="mock_reader")
        mock_operation = MagicMock(name="mock_operation")
        mock_spectra_indices = MagicMock(name="mock_spectra_indices")
        mock_task_index = 1
        self.mock_parallel._write_transformed_chunked_operation(
            reader=mock_reader,
            spectra_indices=mock_spectra_indices,
            task_index=mock_task_index,
            operation=mock_operation,
            open_write_files=False,
            split_modes_and_paths=self.mock_split_modes_and_paths,
        )
        mock_operation.assert_called_once_with(
            mock_reader,
            mock_spectra_indices,
            ANY,
        )
        self.assertEqual(2, len(mock_operation.mock_calls[0].args[2]))
        self.assertEqual(
            Path("/dev/null/mock/continuous_1.imzML"),
            mock_operation.mock_calls[0].args[2][0].imzml_file,
        )
        self.assertEqual(ImzmlModeEnum.CONTINUOUS, mock_operation.mock_calls[0].args[2][0].imzml_mode)
        self.assertIsInstance(mock_operation.mock_calls[0].args[2][1], ImzmlWriteFile)
        self.assertEqual(
            Path("/dev/null/mock/processed_1.imzML"),
            mock_operation.mock_calls[0].args[2][1].imzml_file,
        )
        self.assertEqual(ImzmlModeEnum.PROCESSED, mock_operation.mock_calls[0].args[2][1].imzml_mode)
        self.assertIsInstance(mock_operation.mock_calls[0].args[2][1], ImzmlWriteFile)

    @patch.object(ImzmlWriteFile, "writer")
    def test_write_transformed_chunked_operation_when_open_write_files(self, mock_writer) -> None:
        mock_writer_instances = [MagicMock(name=f"mock_writer_{i}") for i in range(2)]
        mock_writer.return_value.__enter__.side_effect = mock_writer_instances
        mock_reader = MagicMock(name="mock_reader")
        mock_operation = MagicMock(name="mock_operation")
        mock_spectra_indices = MagicMock(name="mock_spectra_indices")
        mock_task_index = 1
        self.mock_parallel._write_transformed_chunked_operation(
            reader=mock_reader,
            spectra_indices=mock_spectra_indices,
            task_index=mock_task_index,
            operation=mock_operation,
            open_write_files=True,
            split_modes_and_paths=self.mock_split_modes_and_paths,
        )
        mock_operation.assert_called_once_with(
            mock_reader,
            mock_spectra_indices,
            ANY,
        )
        self.assertEqual(2, len(mock_operation.mock_calls[0].args[2]))
        self.assertEqual(mock_writer_instances[0], mock_operation.mock_calls[0].args[2][0])
        self.assertEqual(mock_writer_instances[1], mock_operation.mock_calls[0].args[2][1])

    @patch.object(MergeImzml, "merge")
    @patch("depiction.parallel_ops.write_spectra_parallel.ImzmlReadFile")
    def test_merge_results(self, mock_read_file, method_merge) -> None:
        mock_read_file.side_effect = lambda path: {"read_file": path}
        mock_write_file_0 = MagicMock(
            name="mock_write_file_0",
            imzml_file="outputC.imzML",
            imzml_mode=ImzmlModeEnum.CONTINUOUS,
        )
        mock_write_file_1 = MagicMock(
            name="mock_write_file_1",
            imzml_file="outputP.imzML",
            imzml_mode=ImzmlModeEnum.PROCESSED,
        )
        self.mock_parallel._merge_results(
            split_modes_and_paths=self.mock_split_modes_and_paths,
            write_files=[mock_write_file_0, mock_write_file_1],
        )
        self.assertListEqual(
            [
                call(
                    input_files=[
                        {"read_file": Path("/dev/null/mock/continuous_0.imzML")},
                        {"read_file": Path("/dev/null/mock/continuous_1.imzML")},
                        {"read_file": Path("/dev/null/mock/continuous_2.imzML")},
                    ],
                    output_file=mock_write_file_0,
                ),
                call(
                    input_files=[
                        {"read_file": Path("/dev/null/mock/processed_0.imzML")},
                        {"read_file": Path("/dev/null/mock/processed_1.imzML")},
                        {"read_file": Path("/dev/null/mock/processed_2.imzML")},
                    ],
                    output_file=mock_write_file_1,
                ),
            ],
            method_merge.mock_calls,
        )


if __name__ == "__main__":
    unittest.main()
