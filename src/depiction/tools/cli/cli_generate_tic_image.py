from pathlib import Path

import cyclopts

from depiction.parallel_ops import ParallelConfig
from depiction_io import ImzmlReadFile
from depiction.tools.generate_ion_image import GenerateIonImage

app = cyclopts.App()


@app.default
def generate_tic_image(imzml_path: Path, output_hdf5_path: Path, *, n_jobs: int = 16) -> None:
    parallel_config = ParallelConfig(n_jobs=n_jobs)
    gen_image = GenerateIonImage(parallel_config=parallel_config)
    image = gen_image.generate_tic_image_for_file(ImzmlReadFile(imzml_path))
    image.write_hdf5(output_hdf5_path)


if __name__ == "__main__":
    app()
