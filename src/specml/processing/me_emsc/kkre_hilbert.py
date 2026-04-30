import numpy as np
from scipy.signal import hilbert


def kkre_hilbert(absorbance):
    """
    Real part of the refractive index from the Hilbert transform of absorbance
    (Kramers-Kronig relation, real part).
    """
    rindex = -np.imag(hilbert(absorbance))
    return rindex
