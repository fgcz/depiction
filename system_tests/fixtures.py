"""The acquisitions the pipeline system tests run against.

Two of them, and the difference between them is the point:

- `TONSIL` is the FGCZ acquisition this test was written for. It is 1.26 GB, it is not
  redistributable, and its panel is a B-Fabric dataset. It runs on one laptop.
- `MOUSE_KIDNEY` is public, MIT-licensed and 59 MB, so it runs in CI. Its imzML/ibd come
  from the manifest in `tests/real_data/datasets.py` -- imported, never copied, so there is
  one place where a URL or a checksum lives.

Deliberately absent: any expected geometry, channel count or pixel count. Those used to be
constants in the test and are now derived from whatever the pipeline was actually given; see
`calibration/test_pipeline_calibration_only.py`. The single expectation that survives here is
`expect_every_pixel_has_signal`, because it is a property of the acquisition and its panel
rather than something the inputs state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tests.real_data.datasets import CACHE_DIR_ENV_VAR, MOUSE_KIDNEY as _MOUSE_KIDNEY_DEPOSIT

_SYSTEM_TESTS_DIR = Path(__file__).parent
_INPUTS_DIR = _SYSTEM_TESTS_DIR / "inputs"


@dataclass(frozen=True)
class PipelineFixture:
    """One acquisition plus the panel to run it against."""

    name: str
    #: The imzML/ibd pair, as a callable rather than two paths: the public fixture's cache
    #: directory is overridable at run time through `DEPICTION_TEST_DATA_DIR`, which is also
    #: how these tests are made to skip.
    files: Callable[[], tuple[Path, Path]]
    panel_path: Path
    #: What to tell someone who does not have the files.
    how_to_obtain: str
    #: Whether every acquired pixel is expected to carry signal in at least one panel
    #: channel. When true the foreground mask must equal the acquisition's coordinate set
    #: exactly; when false only the subset relation is asserted.
    expect_every_pixel_has_signal: bool

    @property
    def imzml_path(self) -> Path:
        return self.files()[0]

    @property
    def ibd_path(self) -> Path:
        return self.files()[1]

    @property
    def missing(self) -> list[Path]:
        return [path for path in (self.imzml_path, self.ibd_path, self.panel_path) if not path.is_file()]

    @property
    def skip_reason(self) -> str:
        names = ", ".join(str(path) for path in self.missing)
        return f"{self.name} fixture incomplete, missing {names}. {self.how_to_obtain}"


#: FGCZ targeted MSI acquisition: 10131 spectra, a 118-marker PC-MT panel, 50 um raster.
TONSIL = PipelineFixture(
    name="tonsil",
    files=lambda: (_INPUTS_DIR / "tonsil.imzML", _INPUTS_DIR / "tonsil.ibd"),
    panel_path=_INPUTS_DIR / "panel.csv",
    how_to_obtain=(
        "B-Fabric resources 2445579 (imzML) and 2445566 (ibd), and dataset 53798 (panel.csv); "
        "see system_tests/README.md."
    ),
    # 10131 spectra and 10131 foreground pixels: every acquired pixel carries at least one
    # marker. Checked against the file's own `spectrumList count`.
    expect_every_pixel_has_signal=True,
)

#: Public FFPE mouse kidney tryptic peptides, cropped: 1581 spectra, no declared pixel size,
#: and a panel generated from the acquisition itself rather than from a marker list.
MOUSE_KIDNEY = PipelineFixture(
    name="mouse_kidney",
    files=lambda: (_MOUSE_KIDNEY_DEPOSIT.imzml.local_path, _MOUSE_KIDNEY_DEPOSIT.ibd.local_path),
    panel_path=_SYSTEM_TESTS_DIR / "panels" / "mouse_kidney.csv",
    how_to_obtain=(
        f"Run `uv run python -m tests.real_data.fetch {_MOUSE_KIDNEY_DEPOSIT.name}`, "
        f"or point {CACHE_DIR_ENV_VAR} at an existing copy."
    ),
    # Its panel is the 20 strongest peaks of its own mean spectrum, so every spectrum hits
    # several of them; measured as 1581 of 1581.
    expect_every_pixel_has_signal=True,
)

FIXTURES: tuple[PipelineFixture, ...] = (MOUSE_KIDNEY, TONSIL)

FIXTURES_BY_NAME: dict[str, PipelineFixture] = {fixture.name: fixture for fixture in FIXTURES}
