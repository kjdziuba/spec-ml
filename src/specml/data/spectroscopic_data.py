import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import scipy.integrate as sci_int
from specml.utils.helpers import arg_size_checker, get_wavenumber_indices
from specml.processing.masking import create_mask_from_area, apply_mask, rebuild_tissue_mask
from specml.processing.preprocessing import  (
                                                adjust_spectra_min_to_zero,
                                                vector_norm,
                                                apply_savitzky_golay,
                                                mean_center,
                                                baseline_correction,
                                                z_score_normalize)
from specml.ml_models.clustering import apply_clustering
from specml.data.loaders import DataLoader
from specml.visualization.plotting import display_image
from specml.feature_engineering.dim_reduction import apply_pca, apply_umap
from skimage.transform import resize

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inplace_method(func):
    def wrapper(self, *args, inplace=None, **kwargs):
        inplace = self.inplace if inplace is None else inplace
        if inplace:
            # Modify self and expect method to return self
            return func(self, *args, **kwargs)
        else:
            # Create a copy of self and modify it
            new_instance = self.from_existing()
            return func(new_instance, *args, **kwargs)
    return wrapper

class SpectroscopicData:
    """
    A class representing spectroscopic data with methods for processing.

    Attributes:
        data (np.ndarray): The spectroscopic data array.
        wavenumbers (np.ndarray): Array of wavenumbers.
        mask (np.ndarray): Boolean mask array.
        metadata (dict): Metadata information.
        xpixels (int): Number of pixels along x-axis.
        ypixels (int): Number of pixels along y-axis.
        labels (np.ndarray): Labels for the data, if any.
    """

    def __init__(self, file_path = None, inplace = True, data = None, wavenumbers = None, mask = None,
                metadata = None, labels = None, label_mapping=None, 
                annotations = None, type_of_data = None):
        self.file_path = file_path
        self.data = data
        self.wavenumbers = wavenumbers
        self.mask = mask
        self.metadata = metadata or {}
        if self.metadata:
            self.xpixels = self.metadata['info']['xpixels']
            self.ypixels = self.metadata['info']['ypixels']
        self.labels = labels
        self.label_mapping = label_mapping or {}  # Mapping from numeric labels to class names
        self.annotations = annotations

        self.inplace: bool = inplace  # Default inplace behavior

        if file_path:
            self.load_data(file_path, type_of_data=type_of_data)

### DATA HANDLING AND LOADING METHODS ###
    def load_data(self, file_path: str, type_of_data = None):
        """
        Loads the data using DataLoader.

        Args:
            file_path (str): Path to the data file.
        """
        if type_of_data == 'hdf5':
            loaded_data = DataLoader.load_hdf5(file_path)
        else:
            loaded_data = DataLoader.load_data(file_path)
        if loaded_data is None:
            raise ValueError(f"Failed to load data from {file_path}.")
        
        self.data = loaded_data['data']
        self.wavenumbers = loaded_data['wavenumbers']
        self.metadata = {'info': loaded_data['metadata']}
        self.xpixels = self.metadata['info']['xpixels']
        self.ypixels = self.metadata['info']['ypixels']
        # Always set mask from loaded data, flattening it if needed
        self.mask = loaded_data['mask'].flatten() if loaded_data['mask'] is not None else None
        self.metadata['original_shape'] = self.data.shape
        self.data = self.flatten()
        logger.info(self.metadata)
    
    def load_labels(self, label_image_path: str, label_mapping: dict = None):
        """
        Loads labels from a PNG image and stores them in the SpectroscopicData object.

        Args:
            label_image_path (str): Path to the label image file (PNG).
            label_mapping (dict, optional): Mapping from pixel values to class labels.

        Raises:
            ValueError: If the label image dimensions do not match the spectroscopic data dimensions.
        """
        labels, label_mapping = DataLoader.load_labels(
            label_image_path=label_image_path,
            xpixels=self.xpixels,
            ypixels=self.ypixels,
            mask=self.mask,
            label_mapping=label_mapping
        )
        self.labels = labels
        self.label_mapping = label_mapping
### ----------------------------- ###

    def flatten(self) -> np.ndarray:
        """
        Flattens the data for pixel-wise operations and applies the mask.

        Returns:
            np.ndarray: Flattened data array.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please load data before calling flatten().")
        # Check if data is already flattened
        if self.data.ndim == 2:
            return self.data  # Already flattened
        
        # Flatten spatial dimensions (y, x) to a single dimension
        flattened_data = self.data.reshape(-1, self.data.shape[-1])  # Shape: (n_pixels, n_wavenumbers)
        
        # Apply mask if it exists
        if self.mask is not None:
            masked_data = flattened_data[self.mask]
            logger.info(f"Data flattened and mask applied. Shape of processed data: {masked_data.shape}")
            return masked_data
        else:
            logger.info(f"Data flattened without mask. Shape of processed data: {flattened_data.shape}")
            return flattened_data
        
    def to_image(self, data_2d: np.ndarray = None):
        """
        Reshapes the flattened data back to 3D, filling masked pixels appropriately.

        Args:
            data_2d (np.ndarray, optional): 2D data array to reshape. If None, uses self.data.

        Returns:
            np.ndarray: 3D data array of shape (ypixels, xpixels, n_wavenumbers).
        """
        if data_2d is None:
            data_2d = self.data
        
        if data_2d.ndim == 1:
            data_2d = data_2d.reshape(-1, 1)
        elif data_2d.ndim != 2:
            raise ValueError("Input data must be a 2D array.")

        # Initialize a 3D array with the full image shape
        reshaped_data = np.zeros((self.ypixels * self.xpixels, data_2d.shape[1]))

        # Check that the number of unmasked pixels matches the number of data points
        if np.sum(self.mask) != self.data.shape[0]:
            raise ValueError("Mask and data size do not match. Check your mask or data.", )

        # Place the masked data back into the full array
        reshaped_data[self.mask] = data_2d
        reshaped_data = reshaped_data.reshape(self.ypixels, self.xpixels, -1)

        logger.info(f"Data reshaped back to 3D with shape {reshaped_data.shape}")
        return reshaped_data
        
    def get_spectrum_at(self, x: int, y: int) -> np.ndarray:
        """
        Retrieves the spectrum at the specified x and y coordinates.

        Args:
            x (int): The x-coordinate (column index).
            y (int): The y-coordinate (row index).

        Returns:
            np.ndarray: The spectrum at the specified coordinates.

        Raises:
            ValueError: If the coordinates are out of bounds or data is not loaded.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please load data before retrieving spectra.")

        if self.data.ndim == 2:
            # Data is flattened; reshape to 3D
            data_3d = self.to_image()
        else:
            data_3d = self.data

        if not (0 <= x < self.xpixels) or not (0 <= y < self.ypixels):
            raise ValueError(f"Coordinates (x={x}, y={y}) are out of bounds.")

        spectrum = data_3d[y, x, :]
        return spectrum    
    
### DATA PROCESSING METHODS ###
    @inplace_method
    def keep_range(self, lower, upper):
        """
        Clips the spectral data and wavenumbers to the specified wavenumber range.

        Args:
            lower (float): Lower wavenumber limit.
            upper (float): Upper wavenumber limit.
            inplace (bool, optional): If True, modifies the object in place.
                If False, returns a new object. If None, uses the object's default.

        Returns:
            SpectroscopicData: The updated object (self or a new instance).
        """
        lowerlim, upperlim = get_wavenumber_indices(self.wavenumbers, lower, upper)

        # Clip both wavenumbers and data to the specified range
        if self.data.ndim == 2:              
            self.data = self.data[:, lowerlim:upperlim]
        elif self.data.ndim == 3:
            self.data = self.data[:, :, lowerlim:upperlim]
        
        self.wavenumbers = self.wavenumbers[lowerlim:upperlim]

        logger.info(f"Kept wavenumber range from {lower} to {upper}")
        # Return the object itself for chaining
        return self
    
    @inplace_method
    def remove_range(self, upper, lower):
        """
        Removes the spectral data between the two specified wavenumber limits.

        :param lower: Lower wavenumber limit.
        :param upper: Upper wavenumber limit.
        :return: Returns the modified SpectroscopicData object for method chaining.
        """
        lowerlim, upperlim = get_wavenumber_indices(self.wavenumbers, lower, upper)

        # Check if there are any valid wavenumbers in the range to remove
        if lowerlim == upperlim:
            print(f"Warning: No valid wavenumbers found in the range ({lower}, {upper}).")
            return self  # Nothing to remove

        # Remove the range by concatenating the data outside the limits
        if self.data.ndim == 2:              
            self.data = np.concatenate((self.data[:, 0:lowerlim], self.data[:, upperlim:]), axis=1)
        elif self.data.ndim == 3:
            self.data = np.concatenate((self.data[:, :, 0:lowerlim], self.data[:, :, upperlim:]), axis=2)

        # Update the wavenumbers to reflect the removal
        self.wavenumbers = np.concatenate((self.wavenumbers[0:lowerlim], self.wavenumbers[upperlim:]))
        

        # Optional: replace the removed data with a straight line for continuity
        #  it can be useful in some cases for avoiding
        # sharp discontinuities that can confuse models (especially ML models based on continuous data).
        #
        # Example:
        # if self.data.ndim == 2:
        #     for i in range(self.data.shape[0]):
        #         self.data[i, lowerlim:upperlim] = np.linspace(self.data[i, lowerlim], self.data[i, upperlim], upperlim - lowerlim)
        # elif self.data.ndim == 3:
        #     for i in range(self.data.shape[0]):
        #         for j in range(self.data.shape[1]):
        #             self.data[i, j, lowerlim:upperlim] = np.linspace(self.data[i, j, lowerlim], self.data[i, j, upperlim], upperlim - lowerlim)
        logger.info(f"Removed wavenumber range from {lower} to {upper}")
         
        return self
    
    @inplace_method
    def remove_co2(self):
        """Removes the CO2 absorbance region found between 2450-2250 cm-1."""
        return self.remove_range(2450, 2250)
    
    @inplace_method
    def remove_wax(self):
        """Removes wax absorbance regions: 3000-2700, 2400-2300, 1490-1350 cm-1."""
        return (self.remove_range(1490, 1350)
                    .remove_range(2400, 2300)
                    .remove_range(3000, 2700))
        
    def calculate_area(self, lower, upper, method='simpson', data=None, wavenums=None, **kwargs):
        """
        Calculates the area under each spectral absorbance plot between two specified wavenumber positions.
        Allows choosing between 'trapezoid'/'trapz' and 'simpson' integration methods.

        :param lower: Lower wavenumber value.
        :param upper: Upper wavenumber value.
        :param method: Integration method ('trapezoid' or 'simpson'). Default is 'trapezoid'.
        :param data: Dataset input (optional). If not provided, uses self.data.
        :param wavenums: Wavenumber input (optional). If not provided, uses self.wavenumbers.
        :param kwargs: Additional arguments for the scipy integration functions.
        
        :returns: Array of areas corresponding to each spectrum.
        """
        upper, lower = arg_size_checker(upper, lower)

        # Step 2: Use provided data and wavenumbers, or fall back to self attributes
        if data is None:
            data = self.data
        if wavenums is None:
            wavenums = self.wavenumbers

        # Step 3: Calculate step size and find wavenumber indices
        step = np.abs(np.ravel(wavenums)[0] - np.ravel(wavenums)[1])  # Step size between wavenumbers
        
        lower_loc = np.where((np.ravel(wavenums) < lower + step / 2) & 
                            (np.ravel(wavenums) > lower - step / 2))[0][0]
        upper_loc = np.where((np.ravel(wavenums) < upper + step / 2) & 
                            (np.ravel(wavenums) > upper - step / 2))[0][0]
        
        # Step 4: Handle 3D data (flatten spatial dimensions if necessary)
        is_3D = False
        if data.ndim == 3:
            data = self.flatten()
            is_3D = True

        # Step 5: Extract the relevant spectral range and adjust spectra to have minimum 0
        spectral_range = data[:, lower_loc:upper_loc]
        spectral_range = adjust_spectra_min_to_zero(spectral_range)
        wavenumber_range = wavenums[lower_loc:upper_loc]

        # Step 6: Calculate the area using the chosen integration method
        if method == 'trapezoid' or method == 'trapz':
            areas = np.trapz(spectral_range, x=wavenumber_range, axis=1, **kwargs)
        elif method == 'simpson':
            areas = sci_int.simps(spectral_range, x=wavenumber_range, axis=1, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'trapezoid' or 'simpson'.")
        
        return areas.reshape(self.data.shape[0], self.data.shape[1]) if self.data.ndim == 3 else areas
    
    @inplace_method
    def easy_preprocess(self, wavenumber_range=(1000, 1800), area_range=(1600, 1700), enable_mask=True, area_threshold=1.5, deriv = 0):
        """
        Preprocess the spectroscopic data with a predefined pipeline.
        
        :param wavenumber_range: Tuple specifying the lower and upper wavenumber limits to keep.
        :param area_range: Tuple specifying the wavenumber limits for area calculation.
        :param area_threshold: Threshold for creating a mask from the calculated area.
        
        :returns: Preprocessed SpectroscopicData object
        """
        # Step 1: Refine the wavenumber range to 1000, 3500, wax and co2
        self = self.keep_range(950, 3550)
        self = self.remove_wax()
        self = self.remove_co2()

        # Step 2: Calculate the area under the spectral range
        area = self.calculate_area(*area_range)

        if enable_mask:
            # Step 3: Create mask based on area and apply it
            tissue_mask = create_mask_from_area(area=area, threshold=area_threshold)
            self.data = apply_mask(self.data, tissue_mask)
            self.mask = rebuild_tissue_mask(self, tissue_mask)

        # Step 4: Adjust spectra to bring each spectrum min to zero
        self.data = adjust_spectra_min_to_zero(self.data)

        # Step 5: Perform vector normalization
        self.data = vector_norm(self.data)

        # Step 6: Refine the wavenumber range to interest region
        self = self.keep_range(*wavenumber_range)

        # Step 7: Apply Savitzky-Golay derivative filter
        if deriv > 0:
            self = self.apply_derivative(deriv=deriv)

        return self
    
    @inplace_method
    def preprocess_qcl(self, area_range=(1600, 1700), enable_mask=True, area_threshold=1.5, deriv = 0):
        """
        Preprocess the spectroscopic data with a predefined pipeline.
        
        :param wavenumber_range: Tuple specifying the lower and upper wavenumber limits to keep.
        :param area_range: Tuple specifying the wavenumber limits for area calculation.
        :param area_threshold: Threshold for creating a mask from the calculated area.
        
        :returns: Preprocessed SpectroscopicData object
        """
        # Step 1: remove wax
        self = self.remove_range(1490, 1350)

        # Step 2: Calculate the area under the spectral range
        area = self.calculate_area(*area_range)

        if enable_mask:
            # Step 3: Create mask based on area and apply it
            tissue_mask = create_mask_from_area(area=area, threshold=area_threshold)
            self.data = apply_mask(self.data, tissue_mask)
            self.mask = rebuild_tissue_mask(self, tissue_mask)

        # Step 4: Adjust spectra to bring each spectrum min to zero
        self.data = adjust_spectra_min_to_zero(self.data)

        # Step 5: Perform vector normalization
        self.data = vector_norm(self.data)

        # Step 6: Refine the wavenumber range to interest region
        # data = data.keep_range(1000,1800)

        # Step 7: Apply Savitzky-Golay derivative filter
        if deriv > 0:
            self = self.apply_derivative(deriv=deriv)

        return self

    @inplace_method
    def adjust_min_to_zero(self) -> 'SpectroscopicData':
        """
        Adjusts each spectrum so that its minimum value is 0.

        Returns:
            SpectroscopicData: The modified instance.

        Raises:
            ValueError: If data is not loaded.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please load data before processing.")
        
        self.data = adjust_spectra_min_to_zero(self.data)
        return self

    @inplace_method  
    def normalize(self, method: str = 'vector') -> 'SpectroscopicData':
        """
        Normalizes the spectroscopic data using the specified method.

        Args:
            method (str): Normalization method ('vector' or 'z_score'). Default is 'vector'.

        Returns:
            SpectroscopicData: The modified instance.

        Raises:
            ValueError: If method is unknown or data is not loaded.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please load data before processing.")
            
        if method == 'vector':
            self.data = vector_norm(self.data)
        elif method == 'z_score':
            self.data = z_score_normalize(self.data)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return self  

    @inplace_method
    def apply_derivative(self, window_length: int = 23, polyorder: int = 5, deriv: int = 1, interp: bool = False):
        """
        Applies Savitzky-Golay derivative filter to the spectral data.

        Args:
            window_length (int): Length of the filter window (must be odd). Default is 11.
            polyorder (int): The order of the polynomial used to fit the samples. Default is 2.
            deriv (int): Derivative order to compute. Default is 1 (first derivative).
            interp (bool): If True, uses interpolation to handle edges and preserves data length.
                           If False, trims the data to remove edge effects. Default is False.
        """
        data_2d = self.flatten()
        processed_data, new_wavenumbers = apply_savitzky_golay(
            data_2d, self.wavenumbers, window_length, polyorder, deriv, interp=interp
        )
        self.data = processed_data
        self.wavenumbers = new_wavenumbers
        return self
    
    @inplace_method
    def smooth_data(self, window_length: int = 11, polyorder: int = 2, interp: bool = False):
        """
        Applies Savitzky-Golay smoothing to the spectral data.

        Args:
            window_length (int): Length of the filter window (must be odd). Default is 11.
            polyorder (int): The order of the polynomial used to fit the samples. Default is 2.
            interp (bool): If True, uses interpolation to handle edges and preserves data length.
                           If False, trims the data to remove edge effects. Default is False.
        """
        return self.apply_derivative(window_length, polyorder, deriv=0, interp=interp)

    @inplace_method
    def mean_center(self) -> 'SpectroscopicData':
        """
        Mean-centers the spectroscopic data.

        Returns:
            SpectroscopicData: The modified instance.

        Raises:
            ValueError: If data is not loaded.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please load data before processing.")
            
        self.data = mean_center(self.data)
        return self


    def cluster_data(self, clustering_algorithm):
        """
        Applies the specified clustering algorithm to the spectroscopic data.

        Args:
            clustering_algorithm: An instance of a clustering algorithm (e.g., KMeansClustering).

        Returns:
            ClusteringResult: The result of the clustering.
        """
        return apply_clustering(self, clustering_algorithm, mask=self.mask)

### VISUALIZATION METHODS ###  
    def display(self, lower=1600, upper=1700, title=None, color_bar=False, greyscale=False, cmap="viridis"):
        """
        Plots the summed spectral data within the specified wavenumber range.

        Args:
            spectroscopic_data (SpectroscopicData): The spectroscopic data object.
            lower (float): Lower wavenumber limit.
            upper (float): Upper wavenumber limit.
            title (str, optional): Title of the plot.
            color_bar (bool, optional): If True, adds a color bar.
            greyscale (bool, optional): If True, displays in greyscale.
            cmap (str, optional): Colormap to use.
        """
        summed_data = self.calculate_area(lower, upper)
        display_image(summed_data, self.ypixels, 
                      self.xpixels, mask=self.mask, title=title, 
                      color_bar=color_bar, greyscale=greyscale, cmap=cmap)
### ----------------------------- ###

### PCA ###
    def apply_pca(self, n_components=2, scale_data=False, random_state=None):
        """
        Applies PCA to the SpectroscopicData object and returns a DimReductionResult.

        :param n_components: Number of dimensions to reduce to. Default is 2.
        :param scale_data: Whether to scale data before applying PCA. Default is False.
        :param random_state: Set a random state for reproducibility. Default is None.
        :returns: DimReductionResult object.
        """
        return apply_pca(self, n_components, scale_data, random_state)


    def resize_spatial(self, target_h: int, target_w: int, 
                    preserve_range=True, anti_aliasing=True, 
                    mode='reflect'):
        """
        Resizes the spectral image to [target_h, target_w, n_wavenumbers].
        Applies interpolation to each wavenumber slice.

        Args:
            target_h (int): Desired height.
            target_w (int): Desired width.
            preserve_range (bool): If True, output pixels maintain the same intensity range.
            anti_aliasing (bool): Apply anti-aliasing filter when downsampling.
            mode (str): Parameter for skimage.transform.resize (e.g., 'reflect', 'constant', etc.).
        """
        if self.data is None:
            raise ValueError("Data not loaded.")

        # Convert to 3D: shape -> [H, W, #wavenumbers]
        data_3d = self.to_image()

        # data_3d.shape = (original_h, original_w, #wavenumbers)
        original_h, original_w, n_wavenumbers = data_3d.shape

        # We'll create a new 3D array resized along first 2 dims
        # We'll keep the #wavenumbers dimension as is
        resized_list = []
        for wn_idx in range(n_wavenumbers):
            single_slice = data_3d[:, :, wn_idx]  # shape: [H, W]
            resized_slice = resize(
                single_slice, 
                (target_h, target_w),
                preserve_range=preserve_range,
                anti_aliasing=anti_aliasing,
                mode=mode
            )
            resized_list.append(resized_slice[..., np.newaxis])

        # Stack back along the channel dimension
        resized_data_3d = np.concatenate(resized_list, axis=2)
        # resized_data_3d.shape = (target_h, target_w, #wavenumbers)

        # Flatten again if you prefer [n_pixels, n_wavenumbers]
        # or reorder to PyTorch style [channels, H, W]
        # We'll do PyTorch style below:
        # -> shape: (n_wavenumbers, target_h, target_w)
        resized_data_torch = np.transpose(resized_data_3d, (2, 0, 1))

        # Option 1: Keep it inside the SpectroscopicData object in "image" form
        # self.data = resized_data_3d   # shape (h,w,wn)
        #
        # Option 2: Replace self.data with the PyTorch-friendly shape (channels, H, W)
        self.data = resized_data_torch  

        # Update your metadata
        self.xpixels = target_w
        self.ypixels = target_h
        self.mask = np.ones((target_h, target_w), dtype=bool).flatten()
        self.metadata['info']['xpixels'] = target_w
        self.metadata['info']['ypixels'] = target_h

        return self

    def pad_or_cut_spatial(self, target_h: int, target_w: int, corner='center') -> 'SpectroscopicData':
        """
        Pads or cuts the spectral image data AND its corresponding mask
        to (target_h, target_w). Keeps final shape (H, W, channels) for data.

        Args:
            target_h (int): Desired spatial height.
            target_w (int): Desired spatial width.
            corner (str): Cropping/padding positioning ('top-left' or 'center').

        Returns:
            SpectroscopicData: Self (inplace).
        """
        # --- Get Data and Current Mask ---
        data_3d = self.to_image()  # shape: (H, W, n_wavenumbers)
        H, W, n_channels = data_3d.shape

        # Get current mask and reshape to 2D (H, W)
        if hasattr(self, 'mask') and self.mask is not None:
            try:
                current_mask_2d = self.mask.reshape(H, W)
            except ValueError:
                # Handle case where mask might already be 2D or has unexpected shape
                if self.mask.shape == (H, W):
                    current_mask_2d = self.mask
                else:
                    logging.warning(f"Could not reshape mask with shape {self.mask.shape} to ({H}, {W}). Generating default mask.")
                    current_mask_2d = np.ones((H, W), dtype=bool) # Fallback
        else:
            logging.warning("No mask found before pad/cut. Creating default mask of ones.")
            current_mask_2d = np.ones((H, W), dtype=bool) # Create default if no mask exists

        if corner not in ('top-left', 'center'):
            raise ValueError("Corner must be 'top-left' or 'center'.")

        # --- 1) Crop Data and Mask if bigger ---
        crop_h = min(H, target_h)
        crop_w = min(W, target_w)

        if corner == 'top-left':
            cropped_data_3d = data_3d[0:crop_h, 0:crop_w, :]
            cropped_mask_2d = current_mask_2d[0:crop_h, 0:crop_w]
        elif corner == 'center':
            start_h = (H - crop_h) // 2
            start_w = (W - crop_w) // 2
            cropped_data_3d = data_3d[start_h:start_h + crop_h, start_w:start_w + crop_w, :]
            cropped_mask_2d = current_mask_2d[start_h:start_h + crop_h, start_w:start_w + crop_w]

        # --- 2) Pad Data and Mask if smaller ---
        final_h = target_h
        final_w = target_w

        # Initialize padded arrays
        padded_data_3d = np.zeros((final_h, final_w, n_channels), dtype=cropped_data_3d.dtype)
        padded_mask_2d = np.zeros((final_h, final_w), dtype=bool) # Initialize with False (background)

        # Placement logic for padding
        if corner == 'top-left':
            pad_start_h = 0
            pad_start_w = 0
        elif corner == 'center':
            pad_start_h = (final_h - crop_h) // 2
            pad_start_w = (final_w - crop_w) // 2
        
        # Place cropped data and mask in padded arrays
        padded_data_3d[pad_start_h:pad_start_h + crop_h, pad_start_w:pad_start_w + crop_w, :] = cropped_data_3d
        padded_mask_2d[pad_start_h:pad_start_h + crop_h, pad_start_w:pad_start_w + crop_w] = cropped_mask_2d

        # --- 3) Store Resized Data and Mask ---
        self.data = padded_data_3d  # Update data (shape target_h, target_w, n_channels)
        self.mask = padded_mask_2d  # Update mask (shape target_h, target_w), NOT FLATTENED

        # Update metadata
        self.xpixels = final_w
        self.ypixels = final_h
        if 'info' not in self.metadata: 
            self.metadata['info'] = {}
        self.metadata['info']['xpixels'] = final_w
        self.metadata['info']['ypixels'] = final_h

        return self
    
    def to_torch_format(self) -> np.ndarray:
        """
        Returns the data in (channels, height, width) format WITHOUT
        modifying self.data in place. If you want to modify self.data,
        do self.data = self.to_torch_format().
        
        Assumes current self.data is (H, W, channels).
        """
        if self.data is None or self.data.ndim != 3:
            raise ValueError("Data must be a 3D array (H, W, channels) to convert to torch format.")
        return np.transpose(self.data, (2, 0, 1))

    def from_torch_format(self) -> np.ndarray:
        """
        Returns the data in (H, W, channels) format WITHOUT
        modifying self.data in place. If you want to modify self.data,
        do self.data = self.from_torch_format().
        
        Assumes current self.data is (channels, height, width).
        """
        if self.data is None or self.data.ndim != 3:
            raise ValueError("Data must be a 3D array (channels, H, W) to convert from torch format.")
        return np.transpose(self.data, (1, 2, 0))

    @classmethod
    def combine(cls, data_objects):
        """
        Combines multiple SpectroscopicData objects into a single instance.

        :param data_objects: List of SpectroscopicData objects to combine.
        :return: A new SpectroscopicData object with concatenated data, labels, and mask.
        """
        # Check that all objects have the same wavenumbers
        wavenumbers_list = [obj.wavenumbers for obj in data_objects]
        if not all(np.array_equal(wavenumbers_list[0], wn) for wn in wavenumbers_list):
            raise ValueError("All SpectroscopicData objects must have the same wavenumbers to combine.")

        # Find the maximum ypixels (height) across all objects to pad smaller arrays
        max_ypixels = max([obj.ypixels for obj in data_objects])

        # Convert all flattened data to 3D using to_image() if needed and pad to max_ypixels
        data_3d_list = []
        mask_3d_list = []
        label_list = []
        for obj in data_objects:
            # Handle data and pad to max_ypixels
            if obj.data.ndim == 2:
                data_3d = obj.to_image()
            else:
                data_3d = obj.data
            
            # Pad the data along the y-axis if needed
            padding_needed = max_ypixels - obj.ypixels
            if padding_needed > 0:
                data_3d = np.pad(data_3d, ((0, padding_needed), (0, 0), (0, 0)), mode='constant', constant_values=0)
            
            data_3d_list.append(data_3d)

            # Handle masks and pad to max_ypixels
            if obj.mask.ndim == 1:
                mask_3d = obj.mask.reshape(obj.ypixels, obj.xpixels)
            else:
                mask_3d = obj.mask
            
            if padding_needed > 0:
                mask_3d = np.pad(mask_3d, ((0, padding_needed), (0, 0)), mode='constant', constant_values=False)

            mask_3d_list.append(mask_3d)

            # Handle labels and pad if necessary
            if obj.labels is not None:
                labels_padded = np.pad(obj.labels, (0, padding_needed * obj.xpixels), mode='constant', constant_values=0)
                label_list.append(labels_padded)

        # Concatenate data, mask, and labels along the x-axis (columns)
        combined_data = np.concatenate(data_3d_list, axis=1)
        combined_mask = np.concatenate(mask_3d_list, axis=1)
        combined_labels = np.concatenate(label_list) if label_list else None

        # Calculate the new xpixels and ypixels
        new_xpixels = sum([obj.xpixels for obj in data_objects])
        ypixels = max_ypixels

        # Create a new SpectroscopicData object and update its attributes
        combined_obj = cls()
        combined_obj.data = combined_data
        combined_obj.wavenumbers = data_objects[0].wavenumbers
        combined_obj.xpixels = new_xpixels
        combined_obj.ypixels = ypixels
        combined_obj.mask = combined_mask
        combined_obj.labels = combined_labels

        # Update metadata for the combined object
        combined_obj.metadata['original_shape'] = combined_data.shape
        combined_obj.metadata['info'] = {
            'xpixels': new_xpixels,
            'ypixels': ypixels,
            'wavenumbers': combined_obj.wavenumbers.shape[0]
        }

        return combined_obj

    def from_existing(self, new_data = None, wavenumbers = None, mask = None, labels = None, metadata = None) -> 'SpectroscopicData':
        """
        Creates a new SpectroscopicData instance from an existing one with new data.
        """
        new_instance = SpectroscopicData(inplace=self.inplace)
        new_instance.data = new_data if new_data is not None else self.data.copy()
        new_instance.wavenumbers = wavenumbers if wavenumbers is not None else self.wavenumbers.copy()
        new_instance.mask = mask if mask is not None else self.mask.copy()
        new_instance.metadata = metadata if metadata is not None else self.metadata.copy()
        new_instance.xpixels = self.xpixels
        new_instance.ypixels = self.ypixels
        new_instance.labels = labels if labels is not None else (self.labels.copy() if self.labels is not None else None)
        return new_instance
    
    def from_quality_control(self, clean_data, valid_indices) -> 'SpectroscopicData':
        """
        Creates a new SpectroscopicData instance after quality control.
        
        Args:
            clean_data (np.ndarray): The cleaned spectral data
            valid_indices (np.ndarray): Boolean mask or indices of valid spectra from original data
            
        Returns:
            SpectroscopicData: New instance with updated mask reflecting spatial locations
        """
        new_instance = SpectroscopicData(inplace=self.inplace)
        new_instance.data = clean_data
        new_instance.wavenumbers = self.wavenumbers.copy()
        new_instance.xpixels = self.xpixels
        new_instance.ypixels = self.ypixels
        new_instance.metadata = self.metadata.copy() if hasattr(self.metadata, 'copy') else self.metadata
        new_instance.labels = self.labels.copy() if self.labels is not None and hasattr(self.labels, 'copy') else self.labels
        
        # Create new mask that reflects which spatial pixels are still valid
        # Start with the original mask to get spatial pixel locations
        original_valid_pixels = np.where(self.mask)[0]  # Spatial indices of originally valid pixels
        
        # Convert valid_indices to boolean if needed
        if hasattr(valid_indices, 'dtype') and valid_indices.dtype == bool:
            final_valid_spectra = valid_indices
        else:
            final_valid_spectra = np.zeros(self.data.shape[0], dtype=bool)
            final_valid_spectra[valid_indices] = True
        
        # Map back to spatial coordinates
        final_valid_pixels = original_valid_pixels[final_valid_spectra]
        
        # Create new mask for the full spatial array
        new_mask = np.zeros(int(self.xpixels) * int(self.ypixels), dtype=bool)
        new_mask[final_valid_pixels] = True
        new_instance.mask = new_mask
        
        return new_instance