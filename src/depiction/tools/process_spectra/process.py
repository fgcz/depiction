from depiction.parallel_ops import ParallelConfig, WriteSpectraParallel
from depiction.persistence.types import GenericReadFile, GenericWriteFile, GenericReader, GenericWriter
from depiction.tools.process_spectra.config import ProcessSpectraConfig
from depiction.tools.process_spectra.evaluators import get_combined_evaluator


def process_spectra(read_file: GenericReadFile, write_file: GenericWriteFile, config: ProcessSpectraConfig) -> None:
    parallel_config = ParallelConfig(n_jobs=config.n_jobs)
    write_parallel = WriteSpectraParallel.from_config(parallel_config)
    write_parallel.map_chunked_to_file(
        read_file=read_file, write_file=write_file, operation=_process_chunk, bind_args={"config": config}
    )


def _process_chunk(
    reader: GenericReader, spectra_indices: list[int], writer: GenericWriter, config: ProcessSpectraConfig
) -> None:
    evaluator = get_combined_evaluator(config=config)
    for spectrum_index in spectra_indices:
        mz_arr, int_arr, coords = reader.get_spectrum_with_coords(spectrum_index)
        mz_arr, int_arr = evaluator.evaluate(mz_arr, int_arr)
        writer.add_spectrum(mz_arr, int_arr, coords)
