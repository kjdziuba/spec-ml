import numpy as np
import numpy.linalg as LA

from . import make_MieExtinction_curves
from . import make_basicEMSCmodel


def make_ME_EMSCmodel(referenceSpectrum, pureAbsorbanceSpectrum, wn, alpha0, gamma, options, PCnumber=None):
    """
    Establishes the ME-EMSC model.

    Output columns of MieEMSCmodel:
        1: baseline
        2: reference spectrum
        3 - end: loadings from PCA on Mie extinction curves
    """

    MieExtinctionCurves = make_MieExtinction_curves.make_MieExtinction_curves(pureAbsorbanceSpectrum, wn, alpha0, gamma)

    # Remove the mean-centered reference spectrum from the model spectra
    m = np.dot(referenceSpectrum, referenceSpectrum.T)
    norm = np.sqrt(m)
    rnorm = referenceSpectrum / norm
    s = np.dot(MieExtinctionCurves, rnorm.T)
    MieExtinctionCurves = MieExtinctionCurves - np.dot(s.reshape(-1, 1), rnorm.reshape(1, -1))

    # Decompose the set of Mie functions (manual SVD, no centering)
    U, S, Vt = LA.svd(MieExtinctionCurves, full_matrices=False)

    lds = Vt.T
    latent = (S ** 2) / (MieExtinctionCurves.shape[0] - 1)

    EMSCfunctions = make_basicEMSCmodel.make_basicEMSCmodel(referenceSpectrum, wn, 2)

    if PCnumber is None:
        cumulative_variance = np.cumsum(latent)
        t = (cumulative_variance / np.sum(latent)) * 100
        cumulative_variance = np.cumsum(latent)[:-1]
        t = (cumulative_variance / np.sum(latent)) * 100

        if options.get('PCnumber'):
            PCnumber = options['PCnumber']
        elif options.get('ExplainedVariance'):
            id2 = np.where(t > options['ExplainedVariance'])[0]
            PCnumber = min(id2) + 1
        else:
            raise ValueError("Set options['PCnumber'] or options['ExplainedVariance']")

    nRow, nCol = EMSCfunctions.shape
    PCzeros = np.zeros((nRow, PCnumber))
    MieEMSCmodel = np.hstack([EMSCfunctions, PCzeros])
    for i in range(PCnumber):
        MieEMSCmodel[:, nCol + i] = lds[:, i]

    return MieEMSCmodel, PCnumber
