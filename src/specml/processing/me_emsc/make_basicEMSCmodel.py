import numpy as np


def make_basicEMSCmodel(referenceSpectra, wn, EMSCoption=1):
    """
    Establishes the basic EMSC model (linear + quadratic baseline + reference).

    EMSCoption = 1 -> basic EMSC (baseline, linear, quadratic, mean reference)
    EMSCoption = 2 -> MSC        (baseline, mean reference)
    """

    _, Ny = referenceSpectra.shape

    Start = wn[0]
    End = wn[-1]

    C = 0.5 * (Start + End)
    M0 = 2.0 / (Start - End)
    M = 4.0 / ((Start - End) * (Start - End))

    WaveNumT = wn.reshape(-1, 1)
    Baseline = np.ones((1, Ny))
    Mean = np.mean(referenceSpectra, axis=0)

    if EMSCoption == 1:
        Linear = M0 * (Start - WaveNumT) - 1
        Quadratic = M * (WaveNumT - C) ** 2
        MModel = np.vstack([Baseline, Linear.T, Quadratic.T, Mean])
    elif EMSCoption == 2:
        MModel = np.vstack([Baseline, Mean])

    EMSCmodel = MModel.T

    return EMSCmodel
