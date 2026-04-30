import numpy as np


def basicEMSCsolver(RawSpectra, EMSCModel):
    """
    Solves the basic EMSC correction.

    Input:
    RawSpectra   - Raw spectra (matrix containing one spectrum per row)
    EMSCmodel    - Matrix containing the elements of the EMSC model as column vectors:
                   Column number 1: constant baseline
                   Column number 2: linear
                   Column number 3: quadratic
                   Column number 4: reference spectrum

    Output:
    Corrected    - Corrected spectra (matrix containing one spectrum per row)
    Residuals    - Residuals after correction (matrix containing one spectrum per row)
    Parameters   - EMSC parameters
    """

    Model = EMSCModel
    Parameters = np.linalg.lstsq(Model, RawSpectra.T, rcond=None)[0].T

    Corrected = RawSpectra - np.dot(Parameters[:, 0:3], EMSCModel[:, 0:3].T)
    Corrected = Corrected / Parameters[:, 3].reshape(-1, 1)

    Residuals = RawSpectra - np.dot(Parameters, EMSCModel.T)

    return Corrected, Residuals, Parameters
