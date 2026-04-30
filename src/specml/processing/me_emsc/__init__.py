"""
ME-EMSC (Mie Extinction Extended Multiplicative Signal Correction).

Vendored from Dougal Ferguson's `ME_EMSC_Python`
(https://github.com/dougalferg/ME_EMSC_Python, MIT licensed), which is in turn
a Python port of Solheim/Konevskikh's MATLAB implementation
(https://github.com/BioSpecNorway/ME-EMSC).

See `LICENSE-FERGUSON` for the upstream license. Modifications from upstream:
  * removed Windows-specific `sys.path` hacks at the top of `ME_EMSC.py`
  * converted flat `import foo` statements to package-relative imports

Use `specml.processing.advanced_preprocessing.me_emsc_correction` for a
friendly wrapper, or import `ME_EMSC` from this module directly for full
control over the options dict.
"""

from .ME_EMSC import ME_EMSC

__all__ = ["ME_EMSC"]
