import cyclopts
from loguru import logger
from pathlib import Path
from pydantic import ValidationError

from depiction.persistence.imzml.metadata import Metadata
from depiction.persistence.imzml.parser.parse_metadata import ParseMetadata
from depiction.persistence.pixel_size import PixelSize

app = cyclopts.App()


@app.default
def proc_export_raw_metadata(
    input_imzml_path: Path,
    output_json_path: Path,
) -> None:
    try:
        metadata = ParseMetadata.from_file(input_imzml_path).parse()
    except ValidationError:
        logger.error("Failed to extract metadata from {input_imzml_path}", input_imzml_path=input_imzml_path)
        logger.info("Using dummy metadata instead!")
        # TODO maybe this should be revisited in the future and handled as `None`
        metadata = Metadata(
            pixel_size=PixelSize(size_x=1, size_y=1, unit="micrometer"),
            data_processing=[],
            software=[],
            ibd_checksums={},
        )
    with output_json_path.open("w") as file:
        file.write(metadata.model_dump_json())


if __name__ == "__main__":
    app()
