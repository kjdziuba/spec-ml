# specml/data/loaders.py

import h5py
import numpy as np
from specml.utils.agilentirformats.agilentirformats.agilentmosaic import AgilentMosaic
from specml.utils.agilentirformats.agilentirformats.agilenttile import AgilentTile
import logging
import os
from PIL import Image
import zarr

logger = logging.getLogger(__name__)

class DataLoader:
    """
    A class to load spectroscopic data from various file formats.
    """

    @staticmethod
    def load_data(file_path):
        """
        Loads data from the given file path.

        :param file_path: Path to the data file.
        :return: Dictionary containing data, wavenumbers, mask, and metadata.
        """
        # Error handling for file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        file_type = DataLoader.determine_file_type(file_path)
        reader = DataLoader.get_reader(file_type)
        data_dict = reader.read(file_path)
        
        # Process data
        if file_type == 'zarr':
            data = data_dict['data']
            wavenumbers = data_dict['wavenumbers']
            metadata = data_dict['metadata']
        else:
            data = np.transpose(data_dict['ydata'], (1, 2, 0))  # Shape: (y, x, wavenumbers)
            wavenumbers = data_dict['xdata']
            metadata = data_dict['info']
        xpixels = metadata.get('xpixels', data.shape[1])
        ypixels = metadata.get('ypixels', data.shape[0])
        
        # Handle mask
        if 'mask' in data_dict:
            mask = data_dict['mask']
        else:
            mask = np.ones((ypixels, xpixels), dtype=bool)

        logger.info(f"Data loaded from {file_path} with shape {data.shape}")
        return {
            'data': data,
            'wavenumbers': wavenumbers,
            'mask': mask,
            'metadata': metadata
        }
    
    @staticmethod
    def load_hdf5(file_path):
        """Loads the data using external h5py module for hdf5 file formatted 
        data. Data is loaded in as form x * y by wavenumber h1.dataset format.
        Assume read only access to be given.
        
        :returns: object.items; data, wavenumbers, is_loaded, xpixels, ypixels, dict of metadata
        
        """
        # open with 'with' to ensure file is closed after loading
        with h5py.File(file_path, 'r', locking=False) as temp:
            data = temp['data']['intensities'][:]
            wavenumbers = temp['data']['xvalues'][0]
            xpixels = temp['data']['xpixels'][0]
            ypixels = temp['data']['ypixels'][0]
            mask = None #np.array(temp['data']['tissue_mask']) for aberdeen mask is wrong
            metadata = dict(temp['data'].attrs)
            file_type = 'hdf5'
            data_loaded = True
        metadata['xpixels'] = xpixels
        metadata['ypixels'] = ypixels
        logger.info(f"Data loaded from {file_path} with shape {data.shape}")
        return {
            'data': data,
            'wavenumbers': wavenumbers,
            'mask': mask,
            'metadata': metadata
        }

    @staticmethod
    def determine_file_type(file_path):
        """
        Determines the file type based on the file extension.

        :param file_path: Path to the data file.
        :return: String indicating file type.
        """
        if os.path.isdir(file_path):
        # Try to open as Zarr
            try:
                zarr.open(file_path, mode='r')
                return 'zarr'
            except (zarr.errors.PathNotFoundError, KeyError):
                pass  # Not a Zarr directory
        extension = file_path[-4:].lower()
        if extension in (".dmd", ".drd", ".dms", ".dmt"):
            return 'mosaic'
        else:
            return 'tile'

    @staticmethod
    def get_reader(file_type):
        """
        Returns the appropriate reader class based on file type.

        :param file_type: Type of the file ('mosaic' or 'tile').
        :return: Reader class instance.
        """
        if file_type == 'mosaic':
            return AgilentMosaic()
        elif file_type == 'tile':
            return AgilentTile()
        elif file_type == 'zarr':
            return ZarrReader()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    @staticmethod
    def load_labels(label_image_path: str, xpixels: int, ypixels: int, mask: np.ndarray = None, label_mapping: dict = None):
        """
        Loads labels from a PNG image.

        Args:
            label_image_path (str): Path to the label image file (PNG).
            xpixels (int): Number of pixels in x-direction.
            ypixels (int): Number of pixels in y-direction.
            mask (np.ndarray, optional): Mask applied to the data.
            label_mapping (dict, optional): Mapping from pixel values to class labels.
                                            If not provided, unique pixel values will be used as labels.

        Returns:
            labels (np.ndarray): 1D array of labels corresponding to unmasked data points.
            label_mapping (dict): Mapping from pixel values to class names.

        Raises:
            ValueError: If the label image dimensions do not match the specified dimensions.
        """
        # Load the label image
        label_image = Image.open(label_image_path)
        label_array = np.array(label_image)[:,:,:3]  # Remove alpha channel if present
        # Check dimensions
        if label_array.shape[:2] != (ypixels, xpixels):
            raise ValueError("Label image dimensions do not match the specified data dimensions.")
        
        # Flatten the label array
        labels_flat = label_array.reshape(-1, label_array.shape[2]) # Shape: (ypixels * xpixels, channels)

        # Apply the mask if it exists
        if mask is not None:
            labels_flat = labels_flat[mask]
        
        # Map RGB values to numeric labels
        if label_mapping is None:
            # Generate a mapping from unique colors to numeric labels
            unique_colors, inverse_indices = np.unique(labels_flat, axis=0, return_inverse=True)
            rgb_to_label = {tuple(color): idx for idx, color in enumerate(unique_colors)}
            numeric_label_mapping = {idx: f"Class {idx}" for idx in range(len(unique_colors))}
            labels_numeric = inverse_indices
        else:
            # Use the provided label mapping
            rgb_to_label = {tuple(rgb): idx for idx, rgb in enumerate(label_mapping.keys())}
            numeric_label_mapping = {idx: label_mapping[rgb] for rgb, idx in rgb_to_label.items()}
            labels_numeric = np.array([rgb_to_label.get(tuple(color), -1) for color in labels_flat])
        
        return labels_numeric, numeric_label_mapping

class ZarrReader:
    """Reads QCL/IR hypercube data from a Zarr directory.

    Two layouts are supported:

    * **Pyramid** (rat-lung QCL output)::

          <store>/0/hypercube   shape (y, x, n_wavenumbers)
          <store>/0/wvnm        shape (n_wavenumbers,)
          <store>/0/mask        shape (y, x)            # optional

    * **Flat** (earlier exports / breast tokenisation)::

          <store>/qcl_data      shape (y, x, n_wavenumbers)
          <store>/wavenumbers   shape (n_wavenumbers,)
          <store>/mask          shape (y, x)            # optional

    The loader auto-detects which layout is present and reads accordingly.
    Output shape is always ``(y, x, wavenumbers)`` — the same convention the
    Agilent mosaic path produces after its transpose in
    :py:meth:`DataLoader.load_data`.
    """

    HYPERCUBE_KEYS = ("hypercube", "qcl_data")
    WAVENUMBER_KEYS = ("wvnm", "wavenumbers")

    def read(self, file_path):
        zarr_group = zarr.open(file_path, mode="r")
        sub = self._select_subgroup(zarr_group)

        data = self._load_required(sub, self.HYPERCUBE_KEYS, file_path, "hypercube")
        wavenumbers = self._load_required(sub, self.WAVENUMBER_KEYS, file_path, "wavenumbers")
        wavenumbers = np.asarray(wavenumbers, dtype=np.float64).ravel()

        if data.ndim != 3:
            raise ValueError(
                f"Zarr at {file_path}: expected 3-D hypercube (y, x, wavenumbers), "
                f"got shape {data.shape}."
            )
        if data.shape[-1] != wavenumbers.size:
            raise ValueError(
                f"Zarr at {file_path}: hypercube last axis ({data.shape[-1]}) "
                f"does not match wavenumber count ({wavenumbers.size})."
            )

        ypixels, xpixels = int(data.shape[0]), int(data.shape[1])
        if "mask" in sub:
            raw_mask = np.asarray(sub["mask"])
            if raw_mask.shape != (ypixels, xpixels):
                raise ValueError(
                    f"Zarr at {file_path}: mask shape {raw_mask.shape} does not match "
                    f"hypercube spatial shape ({ypixels}, {xpixels})."
                )
            mask = raw_mask.astype(bool)
        else:
            mask = np.ones((ypixels, xpixels), dtype=bool)

        # Prefer subgroup attrs; fall back to root attrs. Always inject pixel sizes.
        metadata = dict(sub.attrs) if sub.attrs else {}
        if not metadata and sub is not zarr_group:
            metadata = dict(zarr_group.attrs)
        metadata.setdefault("xpixels", xpixels)
        metadata.setdefault("ypixels", ypixels)

        return {
            "data": data,
            "wavenumbers": wavenumbers,
            "mask": mask,
            "metadata": metadata,
        }

    def _select_subgroup(self, zarr_group):
        """Descend into the first numeric subgroup that holds a hypercube.

        Pyramid Zarrs (produced by the rat-lung QCL pipeline) wrap each
        resolution level under a numeric key — e.g. ``/0`` for the full-res
        cube. If the root looks empty but ``/0`` holds the data, descend.
        Otherwise stay at the root for flat-layout files.
        """
        root_has_cube = any(k in zarr_group for k in self.HYPERCUBE_KEYS)
        if root_has_cube:
            return zarr_group
        for level_key in ("0", "1", "2"):
            if level_key in zarr_group and any(
                k in zarr_group[level_key] for k in self.HYPERCUBE_KEYS
            ):
                return zarr_group[level_key]
        return zarr_group

    @staticmethod
    def _load_required(grp, keys, file_path, what):
        for key in keys:
            if key in grp:
                return np.asarray(grp[key])
        raise KeyError(
            f"Zarr at {file_path}: no {what} dataset found. "
            f"Tried keys {list(keys)}; present keys are {list(grp)}."
        )
