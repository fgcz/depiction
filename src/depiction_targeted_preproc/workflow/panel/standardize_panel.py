import cyclopts
import polars as pl
from pathlib import Path
from rich.pretty import pprint

from depiction_targeted_preproc.panel.schema import PanelMainSchema
from depiction_targeted_preproc.panel.standardize_input_panel import StandardizeConfig, standardize

app = cyclopts.App()


def _standardize(input_panel_path: Path, config_name: str) -> pl.DataFrame:
    input_df = pl.read_csv(input_panel_path)
    config = StandardizeConfig.load_packaged(config_name)
    return standardize(config=config, raw_df=input_df)


def _validate(output_df: pl.DataFrame) -> None:
    results = PanelMainSchema.validate(output_df)
    pprint(results)


@app.default
def standardize_panel(input_panel_path: Path, config_name: str, output_panel_path: Path) -> None:
    output_df = _standardize(input_panel_path=input_panel_path, config_name=config_name)
    _validate(output_df=output_df)
    output_df.write_csv(output_panel_path)


if __name__ == "__main__":
    app()
