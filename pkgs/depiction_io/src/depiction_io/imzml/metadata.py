from pydantic import BaseModel

from depiction_io.pixel_size import PixelSize


class Metadata(BaseModel):
    pixel_size: PixelSize
    data_processing: list[str]
    software: list[str]
    ibd_checksums: dict[str, str]
