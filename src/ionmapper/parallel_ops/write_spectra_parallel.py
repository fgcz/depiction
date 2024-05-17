import contextlib
import functools
import os
from typing import Optional, Callable, Any, Union
from tempfile import TemporaryDirectory
from numpy.typing import NDArray

from ionmapper.parallel_ops import ReadSpectraParallel
from ionmapper.parallel_ops.parallel_config import ParallelConfig
from ionmapper.persistence import (
    ImzmlReadFile,
    ImzmlWriteFile,
    ImzmlReader,
    ImzmlWriter,
    ImzmlModeEnum,
)
from ionmapper.tools.merge_imzml import MergeImzml


class WriteSpectraParallel:
    def __init__(self, config: ParallelConfig):
        self._config = config

    @classmethod
    def from_config(cls, config: ParallelConfig):
        return cls(config)

    @classmethod
    def from_params(cls, n_jobs: int, task_size: Optional[int], verbose: int = 1):
        return cls(ParallelConfig(n_jobs=n_jobs, task_size=task_size, verbose=verbose))

    def map_chunked_to_files(
        self,
        read_file: ImzmlReadFile,
        write_files: list[ImzmlWriteFile],
        operation: Union[
            Callable[[ImzmlReader, list[int], list[ImzmlWriter], ...], None],
            Callable[[ImzmlReader, list[int], list[ImzmlWriteFile], ...], None],
        ],
        spectra_indices: Optional[NDArray[int]] = None,
        bind_args: Optional[dict[str, Any]] = None,
        open_write_files: bool = True,
    ):
        """Maps an operation over a file, in chunks, writing the results to a list of files.
        :param read_file: the file to read the spectra from
        :param write_files: the list of files to write the results to
        :param operation: the operation to apply to the spectra, the two signatures are controlled by open_write_files,
            as you can see while your function will have to add the results itself to the writer
        :param spectra_indices: the indices of the spectra to process, by default all spectra are processed
        :param bind_args: keyword arguments to bind to the operation
        :param open_write_files: whether to open the write files, if True the files will be provided as ImzmlWriter
            objects (the default), if False the files will be provided as ImzmlWriteFile objects which currently is
            only used internally (TODO maybe it could even become a private parameter, to be considered later, or
            refactored into a shared private method)
        """
        with TemporaryDirectory() as work_directory:
            split_modes_and_paths = self._get_split_modes_and_paths(
                work_directory=work_directory,
                spectra_indices=spectra_indices,
                read_file=read_file,
                write_files=write_files,
            )

            reader = ReadSpectraParallel.from_config(config=self._config)
            reader.map_chunked(
                read_file=read_file,
                operation=functools.partial(
                    self._write_transformed_chunked_operation,
                    operation=functools.partial(
                        operation,
                        **(bind_args or {}),
                    ),
                    open_write_files=open_write_files,
                    split_modes_and_paths=split_modes_and_paths,
                ),
                spectra_indices=spectra_indices,
                pass_task_index=True,
            )

            # merge the files
            self._merge_results(split_modes_and_paths=split_modes_and_paths, write_files=write_files)

    def map_chunked_external_to_files(
        self,
        read_file: ImzmlReadFile,
        write_files: list[ImzmlWriteFile],
        operation: Callable[[str, list[str]], None],
        spectra_indices: Optional[NDArray[int]] = None,
        bind_args: Optional[dict[str, Any]] = None,
    ):
        def op(
            reader: ImzmlReader,
            spectra_ids: list[int],
            write_files: list[ImzmlWriteFile],
            **kwargs,
        ):
            # TODO maybe kwarg handling could be done a bit more clean here in the future
            # TODO also it's currently untested
            # copy the relevant spectra to a temporary file
            with TemporaryDirectory() as work_directory:
                split_input_file = os.path.join(work_directory, "tmp_file.imzML")
                with ImzmlWriteFile(
                    split_input_file,
                    imzml_mode=reader.imzml_mode,
                ).writer() as tmp_writer:
                    tmp_writer.copy_spectra(reader, spectra_indices=spectra_ids)
                operation(
                    split_input_file,
                    [file.imzml_file for file in write_files],
                    **kwargs,
                )

        self.map_chunked_to_files(
            read_file=read_file,
            write_files=write_files,
            operation=op,
            spectra_indices=spectra_indices,
            bind_args=bind_args,
            open_write_files=False,
        )

    def _get_split_modes_and_paths(
        self,
        work_directory: str,
        read_file: ImzmlReadFile,
        write_files: list[ImzmlWriteFile],
        spectra_indices: Optional[NDArray[int]],
    ) -> list[tuple[ImzmlModeEnum, list[str]]]:
        # determine the number of tasks
        if spectra_indices is not None:
            n_tasks = self._config.get_splits_count(n_items=len(spectra_indices))
        else:
            n_tasks = self._config.get_splits_count(n_items=read_file.n_spectra)

        # for each write_file determine the imzml_mode and a list of file paths
        return [
            (
                write_file.imzml_mode,
                [os.path.join(work_directory, f"chunk_f{i_file}_t{i_task}.imzML") for i_task in range(n_tasks)],
            )
            for i_file, write_file in enumerate(write_files)
        ]

    @staticmethod
    def _write_transformed_chunked_operation(
        reader: ImzmlReader,
        spectra_indices: list[int],
        task_index: int,
        operation: Union[
            Callable[[ImzmlReader, list[int], list[ImzmlWriter], ...], None],
            Callable[[ImzmlReader, list[int], list[ImzmlWriteFile], ...], None],
        ],
        open_write_files: bool,
        split_modes_and_paths: list[tuple[ImzmlModeEnum, list[str]]],
    ):
        """Performs the operation for a chunk of spectra. To be called in a parallel context.
        :param reader: the reader to read the spectra from
        :param spectra_indices: the indices of the spectra to process
        :param task_index: the index of the task
        :param operation: the operation to apply to the spectra, the two signatures are controlled by open_write_files
        :param open_write_files: whether to open the write files
            if True, the operation signature is operation(reader, spectra_indices, write_files, ...)
            if False, the operation signature is operation(reader, spectra_indices, write_files, ...)
        :param split_modes_and_paths: the split modes and paths
        """
        write_files = [
            ImzmlWriteFile(path=split_file_paths[task_index], imzml_mode=imzml_mode)
            for imzml_mode, split_file_paths in split_modes_and_paths
        ]

        if open_write_files:
            # note: use ExitStack to support a dynamic number of context managers
            with contextlib.ExitStack() as stack:
                writers = [stack.enter_context(write_file.writer()) for write_file in write_files]
                operation(reader, spectra_indices, writers)
        else:
            operation(reader, spectra_indices, write_files)

    def _merge_results(
        self,
        split_modes_and_paths: list[tuple[ImzmlModeEnum, list[str]]],
        write_files: list[ImzmlWriteFile],
    ):
        """Merges the results of the parallel operations
        :param split_modes_and_paths: the split modes and paths
        :param write_files: the write files to write the merged results to
        """
        merger = MergeImzml()
        for i_file, write_file in enumerate(write_files):
            merger.merge(
                input_files=[ImzmlReadFile(f) for f in split_modes_and_paths[i_file][1]], output_file=write_file
            )

    def map_chunked_to_file(
        self,
        read_file: ImzmlReadFile,
        write_file: ImzmlWriteFile,
        operation: Callable[[ImzmlReader, list[int], ImzmlWriter], None],
        spectra_indices: Optional[NDArray[int]] = None,
        bind_args: Optional[dict[str, Any]] = None,
    ):
        def wrap_operation(reader: ImzmlReader, spectra_ids: list[int], writers: list[ImzmlWriter], **kwargs):
            return operation(reader, spectra_ids, writers[0], **kwargs)

        return self.map_chunked_to_files(
            read_file=read_file,
            write_files=[write_file],
            operation=wrap_operation,
            spectra_indices=spectra_indices,
            bind_args=bind_args,
            open_write_files=True,
        )
