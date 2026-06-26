"""specml — FTIR / QCL infrared spectroscopy ML toolkit.

Public surface is intentionally narrow: the registration engine in
:mod:`xfer_anno` and downstream training code depend on
:class:`SpectroscopicData` (constructor, ``to_image()``, ``wavenumbers``)
and ``specml.__version__``. See ``rat_lung_project/reports/audit_specml.md``
for the documented contract.
"""

import logging as _logging

__version__ = "0.1.0"
__author__ = "Krzysztof Dziuba"

# Library logging best practice: attach a NullHandler so we never emit
# uncaught records, and never call basicConfig() from library code (that
# was hijacking downstream pipelines' log routing — see Lane A.12).
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from .data import SpectroscopicData
from .visualization import plotting

__all__ = [
    "SpectroscopicData",
    "plotting",
    "__version__",
]
