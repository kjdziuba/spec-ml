import numpy as np


def ME_EMSCsolver(RawSpectra, EMSCModel):
    """
    Solves the ME-EMSC correction.

    Input:
    RawSpectra - Raw spectra (matrix containing one spectrum per row)
    EMSCmodel  - Matrix containing the elements of the EMSC model as column vectors:
                 Column number 1: constant baseline
                 Column number 2: reference spectrum
                 Column number 3 - end: loadings from PCA on Mie extinction curves

    Output:
    Corrected  - Corrected spectra (matrix containing one spectrum per row)
    Residuals  - Residuals after correction (matrix containing one spectrum per row)
    Parameters - EMSC parameters
    """

    Model = EMSCModel
    # Solve for Parameters using least squares
    Parameters = np.linalg.lstsq(Model, RawSpectra.T, rcond=None)[0].T

    # Calculate corrected spectra
    Corrected = RawSpectra - np.dot(Parameters, EMSCModel.T) + np.dot(Parameters[:, 1].reshape(-1, 1), EMSCModel[:, 1].reshape(1, -1))
    Corrected = Corrected / Parameters[:, 1].reshape(-1, 1)  # correct for multiplicative effects

    # Calculate residuals
    Residuals = RawSpectra - np.dot(Parameters, EMSCModel.T)

    return Corrected, Residuals, Parameters
