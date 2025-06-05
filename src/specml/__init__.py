# /c:/Projects/spec-ml/spec-ml/src/__init__.py

"""
This is the initialization file for the src package.
"""

__version__ = "0.1.0"
__author__ = "Krzysztof Dziuba"

from .data import SpectroscopicData
from .visualization import plotting
# from .feature_engineering import dim_reduction
# from .ml_models import RandomForestModel, CNN1DModel
# from .evaluation import metrics
# from .utils import helpers

__all__ = [
    "SpectroscopicData",
    "plotting",
    # "dim_reduction",
    # "RandomForestModel",
    # "CNN1DModel",
    # "metrics",
    # "helpers"
]
