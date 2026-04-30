import numpy as np
from scipy.interpolate import interp1d


def adjustWavenumbers(RefSpec, wnRefSpec, RawSpec, wnRawSpec):
    """
    Adjust the wavenumbers of a reference spectrum to match the measured data.

    Returns (RefSpecFitted, RawSpecFitted, wn) where the reference is
    interpolated onto the raw spectrum's grid over the overlapping range.
    """
    minWavenumber = max(min(wnRefSpec), min(wnRawSpec))
    maxWavenumber = min(max(wnRefSpec), max(wnRawSpec))

    i1 = np.argmin(np.abs(wnRefSpec - minWavenumber))
    i2 = np.argmin(np.abs(wnRefSpec - maxWavenumber))
    RefSpec = RefSpec[0, i1:i2 + 1].reshape(1, -1)
    wnRefSpec = wnRefSpec[i1:i2 + 1].reshape(1, -1)

    j1 = np.argmin(np.abs(wnRawSpec - minWavenumber))
    j2 = np.argmin(np.abs(wnRawSpec - maxWavenumber))
    RawSpec = RawSpec[:, j1:j2 + 1]
    wnRawSpec = wnRawSpec[j1:j2 + 1].reshape(1, -1)

    RefSpecFitted = interp1d(wnRefSpec.flatten(), RefSpec.flatten(), kind='linear', fill_value="extrapolate")(wnRawSpec.flatten())

    RawSpecFitted = RawSpec
    wn = wnRawSpec

    if np.isnan(RefSpecFitted).any():
        if np.isnan(RefSpecFitted[0]):
            RefSpecFitted[0] = RefSpecFitted[1]
        if np.isnan(RefSpecFitted[-1]):
            RefSpecFitted[-1] = RefSpecFitted[-2]

    return RefSpecFitted.reshape(1, -1), RawSpecFitted, wn.flatten()
