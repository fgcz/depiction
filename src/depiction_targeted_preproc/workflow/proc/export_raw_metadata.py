from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from pydantic import ValidationError

from depiction.persistence.extract_metadata import ExtractMetadata, Metadata
from depiction.persistence.pixel_size import PixelSize


def proc_export_raw_metadata(
    input_imzml_path: Annotated[Path, typer.Option()],
    output_json_path: Annotated[Path, typer.Option()],
) -> None:
    try:
        metadata = ExtractMetadata.extract_file(input_imzml_path)
    except ValidationError:
        logger.error("Failed to extract metadata from {input_imzml_path}", input_imzml_path=input_imzml_path)
        logger.info("Using dummy metadata instead!")
        # TODO maybe this should be revisited in the future and handled as `None`
        metadata = Metadata(
            pixel_size=PixelSize(size_x=1, size_y=1, unit="micrometer"), data_processing=[], softwares=[]
        )
    with output_json_path.open("w") as file:
        file.write(metadata.model_dump_json())


def main() -> None:
    typer.run(proc_export_raw_metadata)


if __name__ == "__main__":
    main()
