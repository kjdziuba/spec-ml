import numpy as np
from scipy.special import expit
import warnings

from . import calcWeightFunction
from . import make_ME_EMSCmodel
from . import make_basicEMSCmodel
from . import ME_EMSCsolver
from . import iterationSteps


def ME_EMSC(normalizedReferenceSpectrum, spectraForCorrection, wn, options, custom_weights=None):
    # Correcting Mie scattering in infrared spectra.

    # Initialization with default options
    options_default = {
        #Int: Maximum number of iterations
        'maxIterationNumber': 45,
        #Scales reference spectrum
        'scaleRef': True,
        #Number of loadings used in EMSC-model, if set overwrites Explained
        #Variance
        'PCnumber': False,
        #True/False: Set negative parts of the reference spectrum to zero.
        'PositiveRefSpectrum': True,
        #int, set fixed number of iterations.
        'fixIterationNumber': False,
        'mode': 'Correction',
        'Weights': True,
        'Weights_InflectionPoints': [[3700, 2550], [1900, 0]],
        'Weights_Kappa': [[1, 1], [1, 0]],
        'minRadius': 2,
        'maxRadius': 7.1,
        'minRefractiveIndex': 1.1,
        'maxRefractiveIndex': 1.4,
        'samplingSteps': 10,
        'radius': np.linspace(2, 7.1, 10),
        'n_zero': np.linspace(1.1, 1.4, 10),
        'h': 0.25,
        'ExplainedVariance': 99.96,
        'plotResults': False
    }

    options_all = options_default.keys()
    for option in options_all:
        if option not in options:
            options[option] = options_default[option]

    maxIterationNumber = options['maxIterationNumber']
    if options['fixIterationNumber']:
        maxIterationNumber = options['fixIterationNumber']

    # Errors and warnings
    if options['mode'] != 'PreRun' and options['mode'] != 'Correction':
        raise ValueError("Choose options.mode either 'Correction' or 'PreRun'")

    if options['maxIterationNumber'] == 1:
        warnings.warn("Using only one iteration will not result in a proper correction.")

    if options['mode'] == 'PreRun':
        options['PositiveRefSpectrum'] = False
        options['Weights'] = False
        options['plotResults'] = True
        if len(spectraForCorrection) > 40:
            msg = "Number of spectra exceeds 40 in options.mode 'PreRun'. A fewer number of spectra is advised for this mode. Do you wish to proceed? [Y/N]: "
            answer = input(msg)
            if answer.upper() == 'N':
                return

    # Build weights for the reference spectrum
    if options.get('Weights') == "Custom":
        if custom_weights is not None:
            # Ensure the provided weights array has the correct shape
            if len(custom_weights) != len(wn):
                raise ValueError(f"custom_weights length ({len(custom_weights)}) does not match wavenumber length ({len(wn)}).")
            weights = custom_weights
            print("Using provided custom weights.")
        else:
            # User set "Custom" but didn't pass an array
            raise ValueError("options['Weights'] is 'Custom', but no custom_weights array was provided.")

    elif options.get('Weights'): # This means options['Weights'] is True
        print("Calculating weights from inflection points.")
        weights = calcWeightFunction.calcWeightFunction(wn, options)

    else: # This means options['Weights'] is False or None
        print("No weights applied (using flat weight=1).")
        weights = np.ones(len(wn))

    # Calculate alpha0 and gamma
    alpha0 = (4 * np.pi * options['radius'] * (options['n_zero'] - 1)) * (1e-6)
    gamma = options['h'] * np.log(10) / (4 * np.pi * 0.5 * np.pi * (options['n_zero'] - 1) * options['radius'] * (1e-6))

    # Initialize the correction
    numberOfSpectra = spectraForCorrection.shape[0]
    numberOfIterations = np.ones(numberOfSpectra)

    # First iteration
    pureAbsorbanceSpectrum = normalizedReferenceSpectrum

    MieEMSCmodelForFirstIteration, PCnumber = make_ME_EMSCmodel.make_ME_EMSCmodel(normalizedReferenceSpectrum, pureAbsorbanceSpectrum, wn, alpha0, gamma, options)

    if options['scaleRef'] and maxIterationNumber > 1:
        EMSCScaleModel = make_basicEMSCmodel.make_basicEMSCmodel(normalizedReferenceSpectrum, wn, 1)
    else:
        EMSCScaleModel = 0

    correctedSpectra, residuals, EMSCparameters = ME_EMSCsolver.ME_EMSCsolver(spectraForCorrection, MieEMSCmodelForFirstIteration)

    # Iterations loop
    if maxIterationNumber > 1:
        for spectrumNumber in range(numberOfSpectra):
            print("Correcting spectrum " + str(spectrumNumber) + " of " +
                  str(numberOfSpectra) + "...")
            correctedSpectraForIteration, residualsFromIteration, parameters, numberOfIterationsForSpectra = iterationSteps.iterationSteps(
                spectraForCorrection[spectrumNumber, :].reshape(1,-1), spectrumNumber, correctedSpectra[spectrumNumber, :].reshape(1,-1), EMSCScaleModel,
                maxIterationNumber, weights, wn, alpha0, gamma, options, PCnumber)
            correctedSpectra[spectrumNumber, :] = correctedSpectraForIteration
            EMSCparameters[spectrumNumber, :] = parameters
            residuals[spectrumNumber, :] = residualsFromIteration
            numberOfIterations[spectrumNumber] = numberOfIterationsForSpectra

    return correctedSpectra, residuals, EMSCparameters, numberOfIterations, options
