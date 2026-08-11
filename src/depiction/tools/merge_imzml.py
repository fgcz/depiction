from depiction_io import GenericReadFile, ImzmlModeEnum, ImzmlWriteFile, get_read_file
from tqdm import tqdm
from collections.abc import Sequence
import argparse


class MergeImzml:
    def merge(self, input_files: Sequence[GenericReadFile], output_file: ImzmlWriteFile) -> None:
        with output_file.writer() as writer:
            for input_file in tqdm(input_files, desc=" input file", position=0):
                with input_file.reader() as reader:
                    writer.copy_spectra(reader=reader, spectra_indices=range(input_file.n_spectra))

    def merge_paths(self, input_files: Sequence[str], output_file: str, imzml_mode: ImzmlModeEnum):
        input_files = [get_read_file(f) for f in input_files]
        # Only when every input agrees: inputs with different rasters have no common pixel size,
        # and picking one of them would be a guess stated as a measurement.
        pixel_sizes = {input_file.pixel_size for input_file in input_files}
        output_file = ImzmlWriteFile(
            output_file,
            imzml_mode=imzml_mode,
            pixel_size=next(iter(pixel_sizes)) if len(pixel_sizes) == 1 else None,
        )
        return self.merge(input_files=input_files, output_file=output_file)


def main_merge_imzml(output_imzml: str, input_imzml: list[str], mode: str) -> None:
    """
    Merges the imzml files in ``input_imzml`` into a single imzml file at ``output_imzml``.
    The output file is written in ``mode``, if ``mode`` is 'continuous` and processed spectra are
    read the script might raise an error.
    :param output_imzml: The path to the output imzml file.
    :param input_imzml: The paths to the input imzml files.
    :param mode: The mode to write the output file in.
    """
    MergeImzml().merge_paths(input_imzml, output_imzml, ImzmlModeEnum.from_imzml_str(mode))


def main() -> None:
    """Invokes CLI for main_merge_imzml."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["continuous", "processed"], default="continuous", type=str)
    parser.add_argument("output_imzml", type=str)
    parser.add_argument("input_imzml", type=str, nargs="+")
    args = vars(parser.parse_args())
    main_merge_imzml(**args)


if __name__ == "__main__":
    main()
