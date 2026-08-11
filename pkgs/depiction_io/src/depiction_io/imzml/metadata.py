from pydantic import BaseModel

from depiction_io.pixel_size import PixelSize


class Metadata(BaseModel):
    #: `None` when the file declares no `IMS:1000046`. Optional because an absent pixel size is
    #: recoverable and a confidently wrong one is not: this field used to be required, so the one
    #: caller that parsed metadata substituted a dummy 1 um that reached the exported images.
    pixel_size: PixelSize | None = None
    data_processing: list[str]
    software: list[str]
    ibd_checksums: dict[str, str]
