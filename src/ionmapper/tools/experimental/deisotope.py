from ionmapper.peak_picking.basic_peak_picker import BasicPeakPicker
import pyopenms
from numpy.typing import NDArray
from ionmapper.persistence import ImzmlReadFile, ImzmlWriteFile, ImzmlReader, ImzmlWriter
from ionmapper.parallel_ops import WriteSpectraParallel, ParallelConfig


class DeisotopeSpectra:
    def __init__(self, parallel_config: ParallelConfig):
        self._deisotope_config = dict(
            fragment_tolerance=0.1,
            fragment_unit_ppm=False,
            min_charge=1,
            max_charge=1,
            keep_only_deisotoped=True,
            min_isopeaks=2,
            max_isopeaks=10,
            make_single_charged=False,
            annotate_charge=True,
            annotate_iso_peak_count=True,
            use_decreasing_model=False,
            start_intensity_check=3,
            add_up_intensity=True,
        )
        self._parallel_config = parallel_config

    def process_file(self, input_file: ImzmlReadFile, output_file: ImzmlWriteFile):
        parallelize = WriteSpectraParallel.from_config(self._parallel_config)
        parallelize.map_chunked_to_file(
            read_file=input_file,
            write_file=output_file,
            operation=self._process_chunk,
            spectra_indices=None,
            bind_args={
                "deisotope_config": self._deisotope_config,
            },
        )

    @classmethod
    def _process_chunk(
        cls,
        reader: ImzmlReader,
        spectra_ids: list[int],
        writer: ImzmlWriter,
        deisotope_config: dict,
    ):
        picker = BasicPeakPicker()
        for spectrum_id in spectra_ids:
            mz_arr_in, int_arr_in = reader.get_spectrum(spectrum_id)
            mz_arr_out, int_arr_out = cls._deisotoped_spectrum_peaks(
                mz_arr_in,
                int_arr_in,
                deisotope_config,
                picker,
            )
            writer.add_spectrum(mz_arr_out, int_arr_out, coordinates=reader.coordinates[spectrum_id])

    @staticmethod
    def _deisotoped_spectrum_peaks(
        mz_arr, int_arr, deisotope_config, peak_picker
    ) -> tuple[NDArray[float], NDArray[float]]:
        # pick the initial peaks
        peak_indices = peak_picker.pick_peaks_index(int_arr)
        peak_mz = mz_arr[peak_indices]
        peak_int = mz_arr[peak_indices]

        # create pyopenms object
        spectrum = pyopenms.MSSpectrum()
        spectrum.set_peaks([peak_mz, peak_int])

        # deisotope
        pyopenms.Deisotoper.deisotopeAndSingleCharge(spectrum, **deisotope_config)

        # get the peaks
        return spectrum.get_peaks()
