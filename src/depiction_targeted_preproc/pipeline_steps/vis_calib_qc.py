import shutil
from pathlib import Path
from typing import Annotated

import typer

from benchmark_202404.workflow.process.plot_calibration_peaks_qc import plot_calibration_peaks_qc
from depiction.misc.render_quarto import RenderQuarto


def vis_calib_qc(
    imzml_peaks_before_path: Annotated[Path, typer.Option()],
    imzml_peaks_calib_path: Annotated[Path, typer.Option()],
    mass_list_path: Annotated[Path, typer.Option()],
    calib_data_path: Annotated[Path, typer.Option()],
    output_pdf_path: Annotated[Path, typer.Option()],
) -> None:
    # TODO clean implement (i.e. consider if these intermediary pdfs should be deleted or even part of rules)
    pdf_dir = output_pdf_path.parent / "pdfs"
    pdf_dir.mkdir(exist_ok=True, parents=True)
    plot_calibration_peaks_qc(
        baseline_adj_imzml_path=str(imzml_peaks_before_path),
        calibrated_imzml_path=str(imzml_peaks_calib_path),
        mass_list_vend_path=str(mass_list_path),
        out_peak_counts=pdf_dir / "peak_counts.pdf",
        out_peak_density_combined=pdf_dir / "peak_density_combined.pdf",
        out_peak_density_standards=pdf_dir / "peak_density_standards.pdf",
        out_peak_density_ranges=pdf_dir / "peak_density_ranges.pdf",
        out_marker_presence=pdf_dir / "marker_presence.pdf",
        key_mass="mass",
    )

    # merge these into one
    quarto_source_path = Path(__file__).parent / "vis_calib_qc.qmd"
    output_file = RenderQuarto().render(
        document=quarto_source_path,
        output_dir=output_pdf_path.parent,
        parameters=None,
        output_format="pdf",
        delete_qmd=True,
    )
    shutil.move(output_file, output_pdf_path)


def main() -> None:
    typer.run(vis_calib_qc)


if __name__ == "__main__":
    main()
