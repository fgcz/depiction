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
    cluster_variants = [
        "cluster_kmeans_default.png",
        "cluster_bisectingkmeans_default.png",
        "cluster_kmeans_featscv.png",
        "cluster_bisectingkmeans_featscv.png",
        "cluster-umap_kmeans_default.png",
        "cluster-umap_kmeans_featscv.png",
    ]
    result_files = [
        work_dir / "work" / sample / cluster_variant for sample in samples for cluster_variant in cluster_variants
    ]

    snakemake = SnakemakeInvoke(continue_on_error=False, snakefile_name=snakefile_path)
    snakemake.invoke(
        work_dir=work_dir,
        result_files=result_files,
    )


if __name__ == "__main__":
    main()
