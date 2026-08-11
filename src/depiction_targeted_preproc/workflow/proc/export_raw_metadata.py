import cyclopts
from pathlib import Path

from depiction_io.imzml.parser.parse_metadata import ParseMetadata

app = cyclopts.App()


@app.default
def proc_export_raw_metadata(
    input_imzml_path: Path,
    output_json_path: Path,
) -> None:
    # A file that declares no pixel size exports `"pixel_size": null`, and every consumer of this
    # JSON is expected to handle that. There used to be a fallback here that substituted a dummy
    # 1 um -- which the OME-TIFF then stated as the image's physical size, indistinguishable from
    # a genuine 1 um acquisition -- and discarded the software and checksum fields that had parsed
    # fine. Anything the parser now refuses outright is a file we do not understand; let it raise.
    metadata = ParseMetadata.from_file(input_imzml_path).parse()
    with output_json_path.open("w") as file:
        file.write(metadata.model_dump_json())


if __name__ == "__main__":
    app()
