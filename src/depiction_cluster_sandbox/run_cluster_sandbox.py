from pathlib import Path

from depiction_targeted_preproc.workflow.snakemake_invoke import SnakemakeInvoke

work_dir = Path(__file__).parent / "data-sandbox"


def available_samples() -> list[str]:
    return [p.name for p in (work_dir / "raw").glob("*")]


def main():
    samples = available_samples()
    snakefile_path = Path(__file__).parent / "workflow" / "Snakefile"
    snakemake = SnakemakeInvoke(continue_on_error=False, snakefile_name=snakefile_path)
    snakemake.invoke(
        work_dir=work_dir,
        result_files=[work_dir / "work" / sample / "cluster_kmeans_default.png" for sample in samples],
    )


if __name__ == "__main__":
    main()
