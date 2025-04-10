from __future__ import annotations

import enum
from typing import Literal

from pydantic import BaseModel, PositiveInt, PositiveFloat


class BaselineVariants(str, enum.Enum):
    TopHat = "TopHat"
    LocMedians = "LocMedians"


class BaselineCorrectionConfig(BaseModel, use_enum_values=True, validate_default=True):
    n_jobs: PositiveInt | None = None
    baseline_variant: BaselineVariants = BaselineVariants.TopHat
    window_size: PositiveInt | PositiveFloat = 5000.0
    window_unit: Literal["ppm", "index"] = "ppm"
