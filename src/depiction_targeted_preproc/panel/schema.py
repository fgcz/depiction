import pandera.polars as pa


class PanelMainSchema(pa.DataFrameModel):
    label: str
    mass: float = pa.Field(gt=0)
    type: str = pa.Field(isin=["standard", "target"])


class PanelVisualizeSchema(pa.DataFrameModel):
    label: str
    mass: float = pa.Field(gt=0)
    tol: float = pa.Field(gt=0)
