from depiction.parallel_ops import ParallelConfig
from depiction.persistence.types import GenericReadFile, GenericWriteFile
from depiction.tools.process_spectra.config import ProcessSpectraConfig


def process_spectra(read_file: GenericReadFile, write_file: GenericWriteFile, config: ProcessSpectraConfig) -> None:
    parallel_config = ParallelConfig(n_jobs=config.n_jobs)
