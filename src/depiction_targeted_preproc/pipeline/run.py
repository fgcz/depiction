from pathlib import Path

import cyclopts
from bfabric import Bfabric
from depiction_targeted_preproc.pipeline.prepare_inputs import prepare_inputs
from depiction_targeted_preproc.pipeline.prepare_params import prepare_params

app = cyclopts.App()


@app.default()
def run(workunit_id: int, sample_dir: Path, ssh_user: str | None = None) -> None:
    client = Bfabric.from_config()
    prepare_params(client=client, sample_dir=sample_dir, workunit_id=workunit_id)
    prepare_inputs(client=client, sample_dir=sample_dir, workunit_id=workunit_id, ssh_user=ssh_user)


if __name__ == "__main__":
    app()
