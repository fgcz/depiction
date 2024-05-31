from pathlib import Path
from typing import Annotated

import typer

from depiction.persistence.extract_metadata import ExtractMetadata


def proc_export_raw_metadata(
    input_imzml_path: Annotated[Path, typer.Option()],
    output_json_path: Annotated[Path, typer.Option()],
) -> None:
    metadata = ExtractMetadata.extract_file(input_imzml_path)
    with output_json_path.open("w") as file:
        file.write(metadata.model_dump_json())


def main() -> None:
    typer.run(proc_export_raw_metadata)


if __name__ == "__main__":
    main()
