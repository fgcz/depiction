from dataclasses import dataclass

import numpy as np


@dataclass
class CalibratePeptideMassClustersV1:
    """Experimental implementation of the method presented in [1].

    CAVEATS:
    - e.g. formula (2) shows how the simpler formula changes for peptides with protein cleavage, at least for the data
      that i'm mainly working with this is not relevant yet

    [1] Wolski, W.E., Farrow, M., Emde, AK. et al.
        Analytical model of peptide mass cluster centres with applications.
    Proteome Sci 4, 18 (2006). https://doi.org/10.1186/1477-5956-4-18
    """

    #amino_freqs: dict
    cleavage_probability: float = 0.
    average_protein_seq_length: float = np.nan


    wavelength_lambda: float = 1.0 + 4.95e-4

    # from their results section (which is empirical, but very similar to the previously reported value)
    slope_coefficient: float = 4.98e-4

    def get_monoisotopic_mass(self, nominal_mass: float) -> float:
        return self.wavelength_lambda * nominal_mass

    def get_nominal_mass(self, monoisotopic_mass: float) -> float:
        return monoisotopic_mass / self.wavelength_lambda

    def get_peptide_mass_cluster_center(self) -> None:
        # formula (25) in the paper
        pass
