import numpy as np

from . import basicEMSCsolver
from . import make_ME_EMSCmodel
from . import ME_EMSCsolver


def iterationSteps(spectrumRowForCorrection, spectraNumber, correctedSpectraForIteration,
                   EMSCScaleModel, maxIterationNumber, weights, wn, alpha0, gamma,
                   options, PCnumber):
    """
    Perform iteration steps for ME-EMSC correction.

    Input:
    spectrumRowForCorrection - Spectrum row for correction (1D array)
    spectraNumber            - Spectrum number (int)
    correctedSpectraForIteration - Corrected spectra for iteration (2D array)
    EMSCScaleModel           - EMSC scale model (2D array)
    maxIterationNumber       - Maximum number of iterations (int)
    weights                  - Weights (1D array)
    wn                       - Wavenumbers (1D array)
    alpha0                   - Alpha0 parameter (1D array)
    gamma                    - Gamma parameter (1D array)
    options                  - Options (dict)
    PCnumber                 - Number of principal components (int)

    Output:
    correctedSpectraForIteration - Corrected spectra after iteration (2D array)
    residualsFromIteration        - Residuals after iteration (2D array)
    parameters                   - EMSC parameters (2D array)
    numberOfIterations           - Number of iterations performed (int)
    """

    RMSE = np.zeros(options['maxIterationNumber'])

    for iterationNumber in range(2, maxIterationNumber + 1):
        # Scale the reference spectrum in each iteration
        if options['scaleRef']:
            correctedSpectraForIteration, _, _ = basicEMSCsolver.basicEMSCsolver(correctedSpectraForIteration, EMSCScaleModel)

        # Apply weights to the corrected spectra
        correctedSpectraForIteration *= weights

        # Reference spectrum for EMSC with weights applied
        referenceSpectrum = correctedSpectraForIteration.copy()

        # Set negative parts to zero
        correctedSpectraForIteration[correctedSpectraForIteration < 0] = 0

        # Handle positive reference spectrum option
        if options['PositiveRefSpectrum']:
            referenceSpectrum = correctedSpectraForIteration.copy()

        # Create ME-EMSC model
        MieEMSCmodel, PCnumber = make_ME_EMSCmodel.make_ME_EMSCmodel(referenceSpectrum, correctedSpectraForIteration, wn, alpha0, gamma, options, PCnumber)

        # Calculate corrected spectrum and parameters from EMSC
        correctedSpectraForIteration, residualsFromIteration, parameters = ME_EMSCsolver.ME_EMSCsolver(spectrumRowForCorrection, MieEMSCmodel)

        # Calculate root mean square error
        if not options['fixIterationNumber']:
            RMSE[iterationNumber - 1] = np.round(np.sqrt(np.mean(residualsFromIteration ** 2)), 4)

        # Stop criterion
        if iterationNumber == options['maxIterationNumber']:
            print(f"Spectrum no. {spectraNumber}: Number of iterations (maxIterationNumber) should be bigger.")
            numberOfIterations = options['maxIterationNumber']
        elif options['fixIterationNumber'] and iterationNumber < options['fixIterationNumber']:
            continue
        elif iterationNumber == options['fixIterationNumber']:
            numberOfIterations = iterationNumber
            break
        elif iterationNumber > 2 and not options['fixIterationNumber']:
            if (RMSE[iterationNumber - 1] == RMSE[iterationNumber - 2] == RMSE[iterationNumber - 3]) or (RMSE[iterationNumber - 1] > RMSE[iterationNumber - 2]):
                numberOfIterations = iterationNumber
                break
    return correctedSpectraForIteration, residualsFromIteration, parameters, numberOfIterations
