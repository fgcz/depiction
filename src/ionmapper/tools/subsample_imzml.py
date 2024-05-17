import argparse
import enum
import json

import numpy as np
from ionmapper.persistence import ImzmlReadFile, ImzmlWriteFile


class SubsampleMode(enum.Enum):
    """Defines how the subsampling is performed."""

    randomized = enum.auto()
    linspaced = enum.auto()

    def sample(self, indices: np.ndarray, n_keep: int, seed: int = 0) -> np.ndarray[int]:
        """
        Returns the indices of the spectra to keep.
        :param indices: The indices of the spectra to subsample from.
        :param n_keep: The number of spectra to keep.
        :param seed: The seed to use for the random number generator.
        """
        if self == SubsampleMode.randomized:
            rng = np.random.default_rng(seed=seed)
            return rng.choice(indices, size=n_keep, replace=False)
        elif self == SubsampleMode.linspaced:
            # Note: This kind of is biased to always include the first and last spectrum,
            #       but an alternative approach would just as much be biased to exclude them.
            indices = np.round(np.linspace(0, len(indices) - 1, n_keep))
            return indices.astype(int)
        else:
            raise ValueError(f"Unknown subsample mode: {self} (should be unreachable)")


class SubsampleImzml:
    def __init__(self, ratio: float, mode: SubsampleMode, seed: int = 0):
        self._ratio = ratio
        self._mode = mode
        self._seed = seed

    def determine_spectra_to_keep(self, read_file: ImzmlReadFile):
        """Determines which spectra to keep."""
        all_spectra = np.arange(read_file.n_spectra)
        n_keep = int(read_file.n_spectra * self._ratio)
        return self._mode.sample(all_spectra, n_keep, seed=self._seed)

    def subsample(self, read_file: ImzmlReadFile, write_file: ImzmlWriteFile):
        spectra_to_keep = self.determine_spectra_to_keep(read_file)
        with read_file.reader() as reader, write_file.writer() as writer:
            writer.copy_spectra(reader=reader, spectra_indices=spectra_to_keep)

    def dump_subsample_info(
        self,
        read_file: ImzmlReadFile,
        output_imzml: str,
    ) -> None:
        """
        Dumps some information about the subsampling in an accompanying json file.
        """
        info = {
            "input_imzml": read_file.imzml_file,
            "output_imzml": output_imzml,
            "spectra_to_keep": self.determine_spectra_to_keep(read_file=read_file).tolist(),
            "ratio": self._ratio,
            "mode": self._mode.name,
            "seed": self._seed,
        }
        json_file_name = output_imzml.removesuffix(".imzML") + ".subsample_info.json"
        with open(json_file_name, "w") as f:
            json.dump(info, f, indent=1)


def main_subsample_imzml(input_imzml: str, output_imzml: str, ratio: float, mode: SubsampleMode) -> None:
    """
    Performs subsampling of the .imzML file at input_imzml, keeping ratio (0 to 1) of the original spectra,
    and writes the result to output_imzml.
    Currently, this will always also generate an .ibd file of reduced size.
    :param input_imzml: The input .imzML file.
    :param output_imzml: The output .imzML file.
    :param ratio: The ratio of spectra to keep.
    :param mode: The mode to use for subsampling.
    """
    read_file = ImzmlReadFile(input_imzml)
    write_file = ImzmlWriteFile(output_imzml, imzml_mode=read_file.imzml_mode)

    subsampler = SubsampleImzml(ratio=ratio, mode=mode)
    subsampler.subsample(read_file=read_file, write_file=write_file)
    subsampler.dump_subsample_info(
        read_file=read_file,
        output_imzml=output_imzml,
    )


def main() -> None:
    """Invokes CLI for main_subsample_imzml."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_imzml", type=str)
    parser.add_argument("output_imzml", type=str)
    parser.add_argument("ratio", type=float)
    parser.add_argument(
        "--mode",
        type=SubsampleMode,
        choices=list(SubsampleMode),
        default=SubsampleMode.linspaced,
    )
    args = vars(parser.parse_args())
    main_subsample_imzml(**args)


if __name__ == "__main__":
    main()
