from pathlib import Path

from depiction_targeted_preproc.pipeline_config.model import PipelineArtifact, PipelineParameters

ARTIFACT_FILES_MAPPING = {
    PipelineArtifact.CALIB_IMZML: ["calibrated.imzML", "calibrated.ibd"],
    # PipelineArtifact.CALIB_QC: ["calib_qc.pdf"],
    # TODO
    PipelineArtifact.CALIB_QC: [
        "qc/plot_marker_presence.pdf",
        "qc/plot_peak_density_combined.pdf",
        "qc/plot_peak_density_grouped.pdf",
    ],
    PipelineArtifact.CALIB_IMAGES: ["images_default.ome.tiff", "images_default_norm.ome.tiff"],
    # PipelineArtifact.CALIB_HEATMAP: ["images_calib_heatmap.ome.tiff"],
    # TODO
    PipelineArtifact.CALIB_HEATMAP: [
        "qc/plot_calibration_map.pdf",
        "qc/plot_calibration_map_v2.pdf",
    ],
    PipelineArtifact.DEBUG: [
        "qc/plot_marker_presence_cv.pdf",
        # "qc/plot_spectra_for_marker.pdf",
        # "qc/plot_sample_spectra_before_after.pdf",
        "qc/plot_peak_counts.pdf",
        "qc/plot_scan_direction.pdf",
        "cluster_default_kmeans.hdf5",
        #        "cluster_default_stats_kmeans.csv",
        "cluster_default_kmeans.png",
        "cluster_default_hdbscan.png",
        # "exp_plot_map_comparison.pdf",
        # "qc/plot_marker_presence_mini.pdf",
    ],
}


def get_all_output_files(folders: list[Path]) -> list[Path]:
    artifacts = [PipelineArtifact.CALIB_QC, PipelineArtifact.CALIB_IMAGES, PipelineArtifact.DEBUG]
    filenames = [name for artifact in artifacts for name in ARTIFACT_FILES_MAPPING[artifact]]

    all_files = []
    for folder in folders:
        for filename in filenames:
            all_files.append(folder / filename)

    return all_files


def get_result_files(params: PipelineParameters, work_dir: Path, sample_name: str):
    result_files = list(
        {
            work_dir / sample_name / file
            for artifact in params.requested_artifacts
            for file in ARTIFACT_FILES_MAPPING[artifact]
        }
    )
    return result_files
