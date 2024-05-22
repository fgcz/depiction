from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import os
import pandas as pd
from rpy2 import robjects as robjects
from rpy2.robjects import pandas2ri

from depiction.image.sparse_image_2d import SparseImage2d


@dataclass
class GutenTagLabeledOutputs:
    """Represents gutenTAG results."""

    metapeak_df: pd.DataFrame
    panel_df: pd.DataFrame

    # TODO consider ideal naming
    expanded_correspondence_df: pd.DataFrame | None
    untargeted_correspondence_df: pd.DataFrame | None

    intensities: SparseImage2d | None = None

    @classmethod
    def from_rdata_file(cls, path: str) -> GutenTagLabeledOutputs:
        """Reads the RData file at `path` and returns a `GutenTagLabeledOutputs` instance."""
        entries = cls._read_rdata_entries(path=path)
        return cls.from_rdata_entries(entries=entries)

    @classmethod
    def from_rdata_entries(cls, entries: dict[str, Any], load_intensities: bool = False) -> GutenTagLabeledOutputs:
        return cls(
            metapeak_df=cls._parse_metapeak_df(entries=entries),
            panel_df=cls._parse_panel_df(entries=entries),
            expanded_correspondence_df=cls._parse_expanded_correspondence_df(entries=entries),
            untargeted_correspondence_df=cls._parse_untargeted_correspondence_df(entries=entries),
            intensities=(
                cls._parse_intensities(entries=entries, image_key="IntensityDF") if load_intensities else None
            ),
        )

    @staticmethod
    def _read_rdata_entries(path: str) -> dict[str, Any]:
        """Reads the RData file at `path` and returns a dictionary of the contained objects."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")
        # TODO generic functionality that might be useful in other places
        # TODO if made generic, a safer / more isolated function should be used that will not pollute the whole R session
        entry_names = robjects.r["load"](path)
        if entry_names is None:
            raise FileNotFoundError(f"File {path} does not exist (or loading failed for other unknown reason).")
        return {name: robjects.r[name] for name in list(entry_names)}

    @staticmethod
    def _parse_metapeak_df(entries: dict[str, Any]) -> pd.DataFrame:
        """Parses the metapeak data from the `entries` dictionary."""
        metapeaks_rdata = _get_rlist_key(entries["metapeaks"], "metapeaks")
        arrays = {
            key: pandas2ri.rpy2py(metapeaks_rdata[metapeaks_rdata.names.index(key)]) for key in metapeaks_rdata.names
        }
        return pd.DataFrame(
            {
                "center": arrays["center"],
                "max": arrays["max"],
                "width": arrays["width"],
                "limit_min": arrays["limits"][:, 0],
                "limit_max": arrays["limits"][:, 1],
            }
        )

    @staticmethod
    def _parse_panel_df(entries: dict[str, Any]) -> pd.DataFrame:
        """Parses the panel data from the `entries` dictionary."""
        return pandas2ri.rpy2py(entries["panel"]).rename(columns={"Name": "marker", "FeatureMass": "mass"})

    @staticmethod
    def _parse_expanded_correspondence_df(entries: dict[str, Any]) -> pd.DataFrame:
        """Parses the expanded/default correspondence data from the `entries` dictionary."""
        return pandas2ri.rpy2py(_get_rlist_key(entries["processed"], "CorrespondenceMatrix"))

    @staticmethod
    def _parse_untargeted_correspondence_df(entries: dict[str, Any]) -> pd.DataFrame:
        """Parses the untargeted correspondence data from the `entries` dictionary."""
        df = pandas2ri.rpy2py(
            _get_rlist_key(_get_rlist_key(entries["processed"], "Untargeted"), "UntargetedCorrespondence")
        )
        df["marker"] = df["marker"].apply(lambda x: str(x) if str(x) != "NA_character_" else pd.NA)
        return df

    @staticmethod
    def _parse_intensities(entries: dict[str, Any], image_key: str) -> SparseImage2d:
        """Parses the intensities from the `entries` dictionary for `image_key`.
        Valid values are currently "IntensityDF" and "FilteredDF".
        """
        # TODO do we need image_key "FilteredDF"?

        # TODO very slow ~30s for a small image
        intensity_df = pandas2ri.rpy2py(_get_rlist_key(entries["processed"], image_key))
        coords_df = pandas2ri.rpy2py(_get_rlist_key(entries["processed"], "SpatialCoords"))

        # convert to SparseImage2d
        sparse_values = intensity_df.values
        channel_names = list(intensity_df.columns)
        coordinates = coords_df[["x", "y"]].values.astype(int)
        return SparseImage2d(values=sparse_values, coordinates=coordinates, channel_names=channel_names)


def _get_rlist_key(r_list, name: str):
    """Returns the R list entry with the given `name` from `r_list`."""
    # TODO generic functionality that might be useful in other places
    return r_list[r_list.names.index(name)]
