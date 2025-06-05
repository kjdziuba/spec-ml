# spec-ml

**spec-ml** is a toolkit for spectroscopic machine learning workflows. It
provides utilities to load and preprocess spectral images, run
feature engineering steps such as PCA or UMAP and train machine learning
models. The goal is to make experimentation with infrared or Raman data
simple and reproducible.

## Installation
Clone the repository and install in editable mode. Required packages are
listed in `requirements.txt`.

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick start
The snippet below loads a dataset, performs a few preprocessing steps and
runs PCA.

```python
from specml.data import SpectroscopicData
from specml.feature_engineering.dim_reduction import apply_pca

# Load an HDF5, Mosaic or Zarr dataset
spec = SpectroscopicData(file_path="path/to/data.h5")
# Flatten to 2D (pixels x wavenumbers)
spec = spec.flatten()
# Preprocess spectra
spec = spec.adjust_min_to_zero().mean_center().normalize(method="vector")

# Dimensionality reduction
pca_res = apply_pca(spec, n_components=3, scale_data=True)
print("Explained variance:", pca_res.explained_variance_ratio())
```

## Optional components
- **GUI**: The PyQt5 interface in `src/specml/gui` allows interactive
  exploration and visualization.
- **Notebooks**: The `notebooks/` folder contains example notebooks for
  exploration and model experiments.

## Project structure
Key modules reside under `src/specml`:
- `data` – loaders and the `SpectroscopicData` class
- `processing` – preprocessing and masking utilities
- `feature_engineering` – PCA, UMAP and clustering
- `ml_models` – classical and neural network models
- `visualization` – plotting helpers
- `gui` – optional interactive viewer

Contributions and issues are welcome.
