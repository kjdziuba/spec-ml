import numpy as np

from . import Mie_hulst_complex_scaled
from . import kkre_hilbert


def make_MieExtinction_curves(absorbanceSpectrum, wn, alpha0, gamma):
    """
    Calculate Mie extinction curves from a given imaginary part of the complex
    refractive index.

    Output:
    ExtinctionCurves - Matrix where each row is one Mie extinction curve.
    """

    RefIndexABS = absorbanceSpectrum.T

    xts = 200  # number of points used to extend the spectrum on each side

    dx = wn[1] - wn[0]
    wn = np.concatenate([((dx * np.linspace(1, xts, xts)) + (wn[0] - dx * (xts + 1))), wn,
                         ((dx * np.linspace(1, xts, xts)) + wn[-1])])
    RefIndexABS = np.concatenate([np.full(xts, RefIndexABS[0]), RefIndexABS.flatten(),
                                  np.full(xts, RefIndexABS[-1])])

    # Real fluctuating part via Kramers-Kronig
    RefIndexN = kkre_hilbert.kkre_hilbert(RefIndexABS / (wn * 100))

    RefIndexN = RefIndexN[xts:-xts]
    RefIndexABS = RefIndexABS[xts:-xts]
    wn = wn[xts:-xts]

    if abs(min(RefIndexN)) > 1:
        RefIndexIN = RefIndexN / abs(min(RefIndexN))
        RefIndexABS = RefIndexABS / (wn * 100) / abs(min(RefIndexN))
    else:
        RefIndexIN = RefIndexN
        RefIndexABS = RefIndexABS / (wn * 100)

    refIndexComplex = RefIndexIN + 1j * RefIndexABS

    ExtinctionCurves = np.zeros((len(gamma) * len(alpha0), len(wn)))

    ifunc = 0
    for i in range(len(gamma)):
        for j in range(len(alpha0)):
            ExtinctionCurves[ifunc, :] = Mie_hulst_complex_scaled.Mie_hulst_complex_scaled(refIndexComplex, wn, gamma[i], alpha0[j])
            ifunc += 1

    return ExtinctionCurves
