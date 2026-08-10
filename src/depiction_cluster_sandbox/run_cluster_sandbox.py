from pathlib import Path

from snakemake_invoke import SnakemakeInvoke
from snakemake_invoke.config import SnakemakeInvokeConfig

work_dir = Path(__file__).parent / "data-sandbox"


def available_samples() -> list[str]:
    return [p.name for p in (work_dir / "raw").glob("*")]


def main():
    samples = available_samples()
    snakefile_path = Path(__file__).parent / "workflow" / "Snakefile"

    samples = []
    samples += ["concatenated"]

    cluster_algos = ["kmeans", "bisectingkmeans", "birch"]
    cluster_artifacts = [
        file
        for algo in cluster_algos
        for file in [
            f"cluster_{algo}_default.png",
            f"cluster-umap2-{algo}_default-cluster.png",
            f"cluster-umap2-{algo}_default-image_index.png",
        ]
    ]

    result_files = [work_dir / "work" / sample / artifact for sample in samples for artifact in cluster_artifacts]

    # Neither the import above nor this call has matched `snakemake_invoke` for some time:
    # it was `snakemake_invoke.snakemake_invoke`, a module that does not exist at the pinned
    # revision, and the keywords here predate the move to a config object. Fixed while
    # vendoring rather than left as the only import in the tree that cannot resolve; the
    # sandbox itself is still unsupported and untested.
    snakemake = SnakemakeInvoke(
        config=SnakemakeInvokeConfig(snakefile_path=snakefile_path, continue_on_error=False, n_cores=4)
    )
    snakemake.invoke(
        work_dir=work_dir,
        result_files=result_files,
    )


if __name__ == "__main__":
    main()
