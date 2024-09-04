from pathlib import Path

from depiction_targeted_preproc.workflow.snakemake_invoke import SnakemakeInvoke

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

    snakemake = SnakemakeInvoke(continue_on_error=False, snakefile_name=snakefile_path, n_cores=4)
    snakemake.invoke(
        work_dir=work_dir,
        result_files=result_files,
    )


if __name__ == "__main__":
    main()
