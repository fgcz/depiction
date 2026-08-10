from pydantic import BaseModel

from depiction_targeted_preproc.pipeline_config.model import PipelineArtifact


class Params(BaseModel):
    config_preset: str
    requested_artifacts: list[PipelineArtifact]
    # TODO handle this default value better as it can cause issues often
    n_jobs: int = 10
    mass_list_id: int | None = None
